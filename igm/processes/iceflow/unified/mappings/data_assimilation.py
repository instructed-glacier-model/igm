#!/usr/bin/env python3
# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), see LICENSE

from __future__ import annotations

import tensorflow as tf
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from .mapping import Mapping
from .transforms import TRANSFORMS, ParameterTransform
from igm.processes.iceflow.utils.data_preprocessing import Y_to_UV


@dataclass
class VariableSpec:
    """
    Which state field we invert and which parameterization we use (in PHYSICAL space).
    Bounds are specified in PHYSICAL space, e.g., meters or Pa·s^n.

    Only entries where the resolved mask is True (or non-zero) are exposed to the
    optimizer; the complement keeps its initial physical values.
    """

    name: str  # e.g. "thk", "slidingco"
    transform: str = (
        "identity"  # key in TRANSFORMS
    )
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    mask: Optional[str] = None  # e.g. inverse_mask would use state.inverse_mask


class MappingDataAssimilation(Mapping):

    _DEFAULT_MASK_PATH = "icemask"

    def __init__(
        self,
        bcs: List[str],
        network: tf.keras.Model,
        Nz: tf.Tensor,
        output_scale,
        state,
        variables: List[VariableSpec],
        eps: float = 1e-12,
        field_to_channel: Optional[Dict[str, int]] = None,
        precision: str = "single",
        grad_precond_lambda: float = 0.0,
        grad_precond_p: int = 1,
        grad_precond_cg_max_iter: int = 15,
        grad_precond_cg_tol: float = 1e-4,
    ):
        super().__init__(bcs, precision)
        if not variables:
            raise ValueError(
                "❌ DataAssimilation mapping requires at least one variable."
            )

        self.network = network
        self.Nz = Nz
        self.output_scale = tf.cast(output_scale, self.precision)
        self.vars: List[VariableSpec] = variables
        self.eps = eps

        self.grad_precond_enable = grad_precond_lambda > 0.0
        self.grad_precond_lambda = float(grad_precond_lambda)
        self.grad_precond_p = int(grad_precond_p)
        self.grad_precond_cg_max_iter = int(grad_precond_cg_max_iter)
        self.grad_precond_cg_tol = float(grad_precond_cg_tol)

        if self.grad_precond_p not in (1, 2):
            raise ValueError(
                f"❌ precondition_grad_theta expects p in {{1, 2}}, got {self.grad_precond_p}."
            )

        # Scalar grid spacing for Laplacian-based gradient preconditioning
        dx0 = tf.reshape(tf.convert_to_tensor(state.dX), [-1])[0]
        self._dx = tf.cast(dx0, self.precision)
        self.cg_debug = tf.Variable(False, trainable=False, dtype=tf.bool)
        self.cg_warn = tf.Variable(True, trainable=False, dtype=tf.bool)

        self._da_step_callback = None  # python callable
        self._da_out_freq = 0  # python int

        for v in self.network.trainable_variables:
            if v.dtype != self.precision:
                raise TypeError(
                    f"[DataAssimilation] Network variable dtype is {v.dtype.name}, "
                    f"but requested precision is {self.precision.name}. "
                    "Please build/load the network in the same precision."
                )

        self.field_to_channel: Dict[str, int] = field_to_channel or {
            "thk": 0,
            "usurf": 1,
            "arrhenius": 2,
            "slidingco": 3,
            "dX": 4,
        }

        # Ensure state fields are tf.Variable and keep references (for initialization parity).
        self._field_refs: Dict[str, tf.Variable] = {}
        for spec in self.vars:
            field_val = getattr(state, spec.name)
            if isinstance(field_val, tf.Variable):
                self._field_refs[spec.name] = field_val
            else:
                field_var = tf.Variable(
                    field_val, trainable=False, name=f"ref_{spec.name}"
                )
                setattr(state, spec.name, field_var)
                self._field_refs[spec.name] = field_var

        # Build transform objects, initialize θ from physical fields, record shapes/sizes.
        self.transforms: List[ParameterTransform] = []
        self._theta: List[tf.Variable] = []
        self._shapes: List[tf.TensorShape] = []
        self._sizes: List[tf.Tensor] = []
        self._sizes_int: List[Optional[int]] = []
        self._full_shapes: List[tf.TensorShape] = []
        self._mask_bool: List[tf.Tensor] = []
        self._mask_flat_idx: List[tf.Tensor] = []
        self._background_phys_flat: List[tf.Tensor] = []

        # Map variable name -> index in self.vars / self._theta / etc.
        self._varname_to_idx: Dict[str, int] = {}
        for i, spec in enumerate(self.vars):
            if spec.name in self._varname_to_idx:
                raise ValueError(f"Duplicate variable name in DA mapping: {spec.name}")
            self._varname_to_idx[spec.name] = i

        for spec in self.vars:
            tname = (spec.transform or "identity").lower()
            if tname not in TRANSFORMS:
                raise ValueError(
                    f"❌ Unknown transform '{spec.transform}' for '{spec.name}'."
                )
            tform = TRANSFORMS[tname]()
            self.transforms.append(tform)

            x0_var = self._field_refs[spec.name]
            x0 = tf.cast(tf.convert_to_tensor(x0_var), self.precision)

            theta0_full = tform.to_theta(x0, eps=self.eps)
            full_shape_static = theta0_full.shape
            self._full_shapes.append(full_shape_static)

            if self.grad_precond_enable and full_shape_static.rank != 2:
                raise ValueError(
                    f"❌ Gradient preconditioner currently supports only 2D fields, "
                    f"but '{spec.name}' has shape {full_shape_static}."
                )

            mask_path = spec.mask or self._DEFAULT_MASK_PATH
            mask_tensor = self._resolve_mask(state, mask_path)
            mask_bool = tf.cast(mask_tensor, tf.bool)

            if mask_bool.shape != x0.shape:
                raise ValueError(
                    f"❌ Mask '{mask_path}' shape {mask_bool.shape} does not match "
                    f"field '{spec.name}' shape {x0.shape}."
                )

            flat_mask = tf.reshape(mask_bool, [-1])
            flat_idx = tf.where(flat_mask)[:, 0]
            flat_idx = tf.cast(flat_idx, tf.int32)
            flat_idx = tf.sort(flat_idx)  # deterministic ordering

            if int(tf.size(flat_idx).numpy()) == 0:
                raise ValueError(
                    f"❌ Mask '{mask_path}' for '{spec.name}' has no active elements."
                )

            theta0 = tf.gather(tf.reshape(theta0_full, [-1]), flat_idx)
            background_phys_flat = tf.reshape(x0, [-1])

            theta = tf.Variable(
                tf.cast(theta0, self.precision),
                trainable=True,
                name=f"theta_{spec.name}",
            )
            self._theta.append(theta)

            self._shapes.append(theta.shape)
            self._sizes.append(tf.size(theta))
            self._sizes_int.append(
                theta.shape.num_elements() if theta.shape.num_elements() is not None else None
            )

            self._mask_bool.append(mask_bool)
            self._mask_flat_idx.append(flat_idx)
            self._background_phys_flat.append(background_phys_flat)

        # Warm-start cache for the gradient preconditioner.
        self._pcg_warm: List[List[tf.Variable]] = []
        for spec, full_shape in zip(self.vars, self._full_shapes):
            self._pcg_warm.append(
                [
                    tf.Variable(
                        tf.zeros(full_shape, dtype=self.precision),
                        trainable=False,
                        name=f"pcg_warm_{spec.name}_1",
                    ),
                    tf.Variable(
                        tf.zeros(full_shape, dtype=self.precision),
                        trainable=False,
                        name=f"pcg_warm_{spec.name}_2",
                    ),
                ]
            )

        # Precompute θ-space bounds for optimizer consumption.
        self._L_list: List[tf.Tensor] = []
        self._U_list: List[tf.Tensor] = []
        for spec, theta, tform in zip(self.vars, self._theta, self.transforms):
            Ls, Us = tform.theta_bounds(
                spec.lower_bound, spec.upper_bound, dtype=theta.dtype, eps=self.eps
            )
            self._L_list.append(tf.fill(theta.shape, Ls))
            self._U_list.append(tf.fill(theta.shape, Us))

    @staticmethod
    def _resolve_mask(state, path: str) -> tf.Tensor:
        obj = state
        for attr in path.split("."):
            if not hasattr(obj, attr):
                raise ValueError(
                    f"❌ Mask path '{path}' could not be resolved on state."
                )
            obj = getattr(obj, attr)
        return tf.convert_to_tensor(obj)

    @tf.function(reduce_retracing=True, jit_compile=False)
    def _active_to_full_field(
        self,
        idx: int,
        active_values: tf.Tensor,
        background_flat: tf.Tensor,
    ) -> tf.Tensor:
        """
        Scatter an active-subspace vector back to the full 2D field.
        """
        flat_idx = self._mask_flat_idx[idx]
        updates = tf.reshape(active_values, [-1])

        tf.debugging.assert_equal(tf.shape(updates)[0], tf.shape(flat_idx)[0])

        field_flat = tf.tensor_scatter_nd_update(
            tf.cast(background_flat, updates.dtype),
            flat_idx[:, None],
            updates,
        )

        name = self.vars[idx].name
        shape_dyn = tf.shape(self._field_refs[name])
        field = tf.reshape(field_flat, shape_dyn)
        field.set_shape(self._full_shapes[idx])
        return field

    @tf.function(reduce_retracing=True, jit_compile=False)
    def _full_field_to_active(self, idx: int, field: tf.Tensor) -> tf.Tensor:
        """
        Gather a full field back to the active-subspace θ layout.
        """
        vec = tf.gather(tf.reshape(field, [-1]), self._mask_flat_idx[idx])
        vec = tf.reshape(vec, self._shapes[idx])
        return vec

    @tf.function(reduce_retracing=True, jit_compile=False)
    def _theta_to_field(self, idx: int) -> tf.Tensor:
        tform = self.transforms[idx]
        theta = self._theta[idx]
        updates = tform.to_physical(theta)
        return self._active_to_full_field(
            idx,
            updates,
            self._background_phys_flat[idx],
        )

    @tf.function(reduce_retracing=True)
    def synchronize_inputs(self, inputs: tf.Tensor) -> tf.Tensor:
        updated_inputs = self.apply_theta_to_inputs(inputs)
        return updated_inputs

    @tf.function(jit_compile=False)
    def get_UV(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        processed_inputs = self.synchronize_inputs(inputs)
        self.set_inputs(processed_inputs)
        U, V = self.get_UV_impl()
        for apply_bc in self.apply_bcs:
            U, V = apply_bc(U, V)
        return U, V

    @tf.function(jit_compile=True, reduce_retracing=True)
    def get_UV_impl(self) -> Tuple[tf.Tensor, tf.Tensor]:
        Y = self.network(self.inputs) * self.output_scale
        U, V = Y_to_UV(self.Nz, Y)
        return U, V

    def set_step_callback(self, callback, out_freq: int) -> None:
        self._da_step_callback = callback
        self._da_out_freq = int(out_freq)

    @tf.function(reduce_retracing=True)
    def on_step_end(self, it: tf.Tensor) -> tf.Tensor:
        if self._da_step_callback is None or self._da_out_freq <= 0:
            return tf.constant(0, dtype=tf.int32)

        of = tf.cast(self._da_out_freq, it.dtype)
        do_call = tf.equal(tf.math.floormod(it, of), 0)

        def _call():
            tf.py_function(self._da_step_callback, [it], Tout=[])
            return tf.constant(0, dtype=tf.int32)

        return tf.cond(do_call, _call, lambda: tf.constant(0, dtype=tf.int32))

    # ------- State update -----------------------------------------------------

    def update_state_fields(self, state):
        """Write current physical values back into `state` (post-optimization)."""
        for idx, spec in enumerate(self.vars):
            full_value = self._theta_to_field(idx)
            ref = self._field_refs[spec.name]
            ref.assign(tf.cast(full_value, ref.dtype))
            setattr(state, spec.name, ref)

    # ------- Bounds (θ-space) for optimizer ----------------------------------

    def get_box_bounds_flat(self) -> Tuple[tf.Tensor, tf.Tensor]:
        L_flat = tf.concat([tf.reshape(Li, [-1]) for Li in self._L_list], axis=0)
        U_flat = tf.concat([tf.reshape(Ui, [-1]) for Ui in self._U_list], axis=0)
        return L_flat, U_flat

    # ------- Parameter plumbing ----------------------------------------------

    def get_theta(self) -> List[tf.Variable]:
        return self._theta

    def set_theta(self, theta: List[tf.Tensor]) -> None:
        if len(theta) != len(self._theta):
            raise ValueError("❌ set_theta: length mismatch.")
        for var, val in zip(self._theta, theta):
            var.assign(val)

    def copy_theta(self, theta: List[tf.Variable]) -> List[tf.Tensor]:
        return [theta_i.read_value() for theta_i in theta]

    def copy_theta_flat(self, theta_flat: tf.Tensor) -> tf.Tensor:
        return tf.identity(theta_flat)

    def flatten_theta(self, theta: List[tf.Variable | tf.Tensor]) -> tf.Tensor:
        flats = []
        for i, theta_i in enumerate(theta):
            if theta_i is None:
                raise ValueError(f"❌ None gradient for parameter: {self.vars[i].name}")
            flats.append(tf.reshape(theta_i, [-1]))
        return tf.concat(flats, axis=0)

    def unflatten_theta(self, theta_flat: tf.Tensor) -> List[tf.Tensor]:
        if all(s is not None for s in self._sizes_int):
            vals: List[tf.Tensor] = []
            idx = 0
            for s_int, shp in zip(self._sizes_int, self._shapes):
                nxt = idx + int(s_int)  # type: ignore[arg-type]
                vals.append(tf.reshape(theta_flat[idx:nxt], shp))
                idx = nxt
            return vals
        else:
            splits = tf.split(theta_flat, self._sizes)
            return [tf.reshape(t, s) for t, s in zip(splits, self._shapes)]

    def reset_preconditioner_cache(self) -> None:
        for cache_per_var in self._pcg_warm:
            for x in cache_per_var:
                x.assign(tf.zeros_like(x))

    def on_minimize_start(self, iter_max: int) -> None:
        self.reset_preconditioner_cache()

    # ------- Input channel patching ------------------------------------------

    @tf.function(reduce_retracing=True, jit_compile=False)
    def apply_theta_to_inputs(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Patch BHWC inputs with current physical-space values for selected fields.
        Channel mapping follows the configured mapping.
        """
        updated = inputs
        B, H, W, C = tf.unstack(tf.shape(inputs))
        for idx, spec in enumerate(self.vars):
            ch = self.field_to_channel.get(spec.name, None)
            if ch is None:
                continue
            val = self._theta_to_field(idx)
            val = tf.cast(val, updated.dtype)
            phys_b = tf.tile(tf.reshape(val, [1, H, W, 1]), [B, 1, 1, 1])

            left = updated[:, :, :, :ch]
            right = updated[:, :, :, ch + 1 :]
            updated = tf.concat([left, phys_b, right], axis=-1)
        return updated

    def get_physical_field(self, name: str) -> tf.Tensor:
        """
        Differentiable physical field derived from current theta.
        Safe to call inside tf.function.
        """
        if name not in self._varname_to_idx:
            raise ValueError(
                f"Unknown field '{name}'. Available: {list(self._varname_to_idx.keys())}"
            )
        idx = self._varname_to_idx[name]
        return tf.cast(self._theta_to_field(idx), self.precision)

    # ------- Gradient preconditioning (Sobolev / inverse-Laplacian) ----------

    @tf.function(reduce_retracing=True)
    def _dot2(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        acc = tf.reduce_sum(tf.cast(a, tf.float64) * tf.cast(b, tf.float64))
        return tf.cast(acc, a.dtype)

    @tf.function(reduce_retracing=True, jit_compile=False)
    def _apply_A(self, v: tf.Tensor, mask: tf.Tensor, lam: tf.Tensor) -> tf.Tensor:
        """
        Apply A(v) = v + lam * L(v), where L is a masked graph Laplacian
        (Neumann-like at the mask boundary), scaled by dx^2.

        This uses the PSD graph Laplacian form:
            L(v) = (deg*v - sum_neighbors(v)) / dx^2
        so A is SPD for lam >= 0.
        """
        v = tf.where(mask, v, tf.zeros_like(v))
        mf = tf.cast(mask, v.dtype)

        vpad = tf.pad(v, [[1, 1], [1, 1]], mode="SYMMETRIC")
        mpad = tf.pad(mf, [[1, 1], [1, 1]], mode="CONSTANT", constant_values=0.0)

        c = vpad[1:-1, 1:-1]

        mu = mpad[0:-2, 1:-1]
        md = mpad[2:, 1:-1]
        ml = mpad[1:-1, 0:-2]
        mr = mpad[1:-1, 2:]

        fu = vpad[0:-2, 1:-1]
        fd = vpad[2:, 1:-1]
        fl = vpad[1:-1, 0:-2]
        fr = vpad[1:-1, 2:]

        neigh = mu * fu + md * fd + ml * fl + mr * fr
        deg = mu + md + ml + mr

        dx = tf.cast(self._dx, v.dtype)
        L = (deg * c - neigh) / (dx * dx)

        Av = v + tf.cast(lam, v.dtype) * L
        return tf.where(mask, Av, tf.zeros_like(Av))

    @tf.function(reduce_retracing=True, jit_compile=False)
    def _mask_degree(self, mask: tf.Tensor, dtype: tf.dtypes.DType) -> tf.Tensor:
        """
        Number of active 4-neighbors per cell on the masked domain.
        """
        mf = tf.cast(mask, dtype)
        mpad = tf.pad(mf, [[1, 1], [1, 1]], mode="CONSTANT", constant_values=0.0)

        deg = (
            mpad[0:-2, 1:-1]
            + mpad[2:, 1:-1]
            + mpad[1:-1, 0:-2]
            + mpad[1:-1, 2:]
        )
        return tf.where(mask, deg, tf.zeros_like(deg))

    @tf.function(reduce_retracing=True, jit_compile=False)
    def _cg_solve_A(
        self,
        rhs: tf.Tensor,
        mask: tf.Tensor,
        lam: tf.Tensor,
        x0: Optional[tf.Tensor] = None,
        *,
        max_iter: int = 15,
        tol: float = 1e-4,
    ) -> tf.Tensor:
        """
        Jacobi-preconditioned CG solve for A x = rhs on the masked domain.

        `tol` is a relative tolerance; a small absolute floor is also added so
        we do not oversolve tiny rhs late in the inversion.
        """
        lam = tf.cast(lam, rhs.dtype)

        b = tf.where(mask, rhs, tf.zeros_like(rhs))
        dtype = b.dtype

        dx = tf.cast(self._dx, dtype)
        dx2 = dx * dx

        deg = self._mask_degree(mask, dtype)
        diagA = tf.where(
            mask,
            tf.ones_like(b) + lam * deg / dx2,
            tf.ones_like(b),
        )

        if x0 is None:
            x = tf.where(mask, b / diagA, tf.zeros_like(b))
        else:
            x = tf.where(mask, tf.cast(x0, dtype), tf.zeros_like(b))

        r = b - self._apply_A(x, mask, lam)
        z = tf.where(mask, r / diagA, tf.zeros_like(r))
        p = tf.identity(z)

        rr = self._dot2(r, r)
        rz = self._dot2(r, z)
        rr0 = rr
        bs = self._dot2(b, b)

        b_inf = tf.cast(tf.reduce_max(tf.abs(b)), tf.float64)
        n_active = tf.reduce_sum(tf.cast(mask, tf.float64))
        eps_mach = tf.constant(
            1.1920928955078125e-7 if self.precision == tf.float32 else 2.220446049250313e-16,
            dtype=tf.float64,
        )
        atol = (
            10.0
            * tf.sqrt(tf.maximum(n_active, 1.0))
            * eps_mach
            * tf.maximum(b_inf, tf.cast(self.eps, tf.float64))
        )
        target = tf.maximum(
            tf.cast(tol * tol, tf.float64) * tf.cast(bs, tf.float64),
            atol * atol,
        )

        def _dbg_start():
            tf.print(
                "[PCG] start  lam=", lam,
                "  rr0=", rr0,
                "  target=", target,
                "  max_iter=", max_iter,
            )
            return tf.constant(0, dtype=tf.int32)

        _ = tf.cond(self.cg_debug, _dbg_start, lambda: tf.constant(0, dtype=tf.int32))

        eps = tf.cast(self.eps, dtype)

        def cond(i, x, r, z, p, rr, rz):
            return tf.logical_and(i < max_iter, tf.cast(rr, tf.float64) > target)

        def body(i, x, r, z, p, rr, rz):
            Ap = self._apply_A(p, mask, lam)
            denom = self._dot2(p, Ap) + eps
            alpha = rz / denom

            x_new = x + alpha * p
            r_new = r - alpha * Ap
            z_new = tf.where(mask, r_new / diagA, tf.zeros_like(r_new))

            rr_new = self._dot2(r_new, r_new)
            rz_new = self._dot2(r_new, z_new)
            beta = rz_new / (rz + eps)
            p_new = z_new + beta * p

            return i + 1, x_new, r_new, z_new, p_new, rr_new, rz_new

        i0 = tf.constant(0, dtype=tf.int32)
        i_end, x, _, _, _, rr_end, _ = tf.while_loop(
            cond,
            body,
            loop_vars=[i0, x, r, z, p, rr, rz],
            parallel_iterations=1,
        )

        rel = tf.sqrt(rr_end / (rr0 + eps))
        rel_b = tf.sqrt(rr_end / (bs + eps))
        finite = tf.math.is_finite(rr_end)
        converged = tf.cast(rr_end, tf.float64) <= target
        need_warn = tf.logical_or(tf.logical_not(finite), tf.logical_not(converged))

        def _warn():
            tf.print(
                "[PCG] WARNING  lam=", lam,
                "  iters=", i_end,
                "  rr_end=", rr_end,
                "  rel=", rel,
                "  rel_b=", rel_b,
                "  (finite=", finite, ", converged=", converged, ")",
            )
            return tf.constant(0, dtype=tf.int32)

        _ = tf.cond(
            tf.logical_and(self.cg_warn, need_warn),
            _warn,
            lambda: tf.constant(0, dtype=tf.int32),
        )

        def _dbg_end():
            tf.print(
                "[PCG] end    lam=", lam,
                "  iters=", i_end,
                "  rr_end=", rr_end,
                "  rel=", rel,
                "  rel_b=", rel_b,
            )
            return tf.constant(0, dtype=tf.int32)

        _ = tf.cond(self.cg_debug, _dbg_end, lambda: tf.constant(0, dtype=tf.int32))

        return tf.where(mask, x, tf.zeros_like(x))

    @tf.function(reduce_retracing=True, jit_compile=False)
    def precondition_grad_theta(
        self,
        grad_theta: list[tf.Tensor],
        lam: tf.Tensor,
        *,
        p: int = 1,
        cg_max_iter: int = 15,
        cg_tol: float = 1e-4,
    ) -> list[tf.Tensor]:
        """
        Apply (I + lam * L)^(-p) to each variable's gradient, respecting the
        variable mask. Warm starts are kept separately for stage 1 and stage 2,
        so p=2 also benefits.
        """
        lam = tf.cast(lam, self.precision)

        out: list[tf.Tensor] = []
        for i, g in enumerate(grad_theta):
            g_dtype = g.dtype
            g = tf.cast(g, self.precision)
            mask = self._mask_bool[i]

            rhs = self._active_to_full_field(
                i,
                g,
                tf.zeros_like(self._background_phys_flat[i], dtype=self.precision),
            )
            rhs = tf.where(mask, rhs, tf.zeros_like(rhs))

            x = rhs
            for stage in range(p):
                x = self._cg_solve_A(
                    x,
                    mask,
                    lam,
                    x0=self._pcg_warm[i][stage],
                    max_iter=cg_max_iter,
                    tol=cg_tol,
                )
                self._pcg_warm[i][stage].assign(x)

            out.append(tf.cast(self._full_field_to_active(i, x), g_dtype))

        return out

    @tf.function(reduce_retracing=True, jit_compile=False)
    def precondition_grad_theta_flat(
        self,
        grad_flat: tf.Tensor,
        lam: tf.Tensor,
        *,
        p: int = 1,
        cg_max_iter: int = 15,
        cg_tol: float = 1e-4,
    ) -> tf.Tensor:
        """
        Flat wrapper used by optimizers: unflatten -> precondition per variable -> flatten.
        """
        grads = self.unflatten_theta(grad_flat)
        grads_p = self.precondition_grad_theta(
            grads,
            lam,
            p=p,
            cg_max_iter=cg_max_iter,
            cg_tol=cg_tol,
        )
        return self.flatten_theta(grads_p)

    @tf.function(reduce_retracing=True, jit_compile=False)
    def precondition_direction_grad_flat(self, grad_flat: tf.Tensor) -> tf.Tensor:
        """
        Gradient transform used for search direction / curvature pairs.

        Returns the input unchanged if gradient preconditioning is disabled.
        """
        if not self.grad_precond_enable:
            return grad_flat

        return self.precondition_grad_theta_flat(
            grad_flat,
            tf.cast(self.grad_precond_lambda, self.precision),
            p=self.grad_precond_p,
            cg_max_iter=self.grad_precond_cg_max_iter,
            cg_tol=self.grad_precond_cg_tol,
        )