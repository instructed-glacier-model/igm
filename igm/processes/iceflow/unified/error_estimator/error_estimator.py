#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""A-posteriori velocity-error estimate from a truncated Newton correction.

At the current velocity ``u = (U, V)`` (typically the output of a network
mapping) the estimator evaluates the energy gradient ``g = dE/du`` and a
frozen banded Hessian ``H`` on a *shadow* identity mapping, then runs ``k``
preconditioned CG iterations on ``(H + damping I) delta = -g``. Since
``g(u*) = 0`` at the minimiser, ``-delta`` is the linearised error
``e = u - u*``; optional chord steps (gradient re-evaluated at ``u + delta``,
same frozen Hessian) remove the nonlinearity residual of Glen's flow law.

``delta`` is converted to the quantities used to score network solves:
the median relative surface-velocity error over grounded and floating ice,
the surface-velocity RMSE, and the Newton decrement ``1/2 <-g, delta>``.

The estimator is diagnostic only. It appends one JSON line per estimate to
``record_path`` and publishes the latest scalars as ``state.err_est_*`` (and
the estimated surface error field as ``state.err_est_surf``). When a
reference velocity is given it also logs the true error, which is how the
estimator was validated (see ``igm-ais/error_estimator_study.md``).
"""

import json
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

from igm.utils.math.precision import normalize_precision
from igm.processes.iceflow.utils.velocities import compute_node_ice_mask, get_velsurf

from ..mappings.identity import MappingIdentity
from ..operators import build_energy_operator
from ..optimizers.utils.pcg import pcg_init, pcg_relres, pcg_run
from ..preconditioners import build_preconditioner
from .metrics import grounded_mask, summarize_velocity_error


class ErrorEstimator:
    """Truncated-Newton velocity-error estimator on a shadow identity mapping."""

    def __init__(
        self,
        *,
        cost_fn,
        bcs,
        field_shape: Sequence[int],
        V_s: tf.Tensor,
        idx_thk: int,
        precision: str = "double",
        basis_vertical: str = "molho",
        idx_usurf: Optional[int] = None,
        idx_water_level: Optional[int] = None,
        topg: Optional[np.ndarray] = None,
        basin_mask: Optional[np.ndarray] = None,
        rho_ice: float = 910.0,
        rho_water: float = 1000.0,
        water_level: float = 0.0,
        freq: int = 250,
        estimate_at_start: bool = True,
        cg_iters: Sequence[int] = (10,),
        cg_tol: float = 1.0e-6,
        newton_steps: int = 2,
        hvp_mode: str = "banded",
        probe_mode: str = "fd",
        hvp_verify: bool = False,
        preconditioner: str = "barotropic_multigrid",
        preconditioner_options: Optional[dict] = None,
        damping: float = 1.0e-16,
        operator_update_freq: int = 0,
        operator_refresh_rel_change: float = 0.05,
        disable_xla: bool = True,
        record_path: Optional[str] = "error_estimate.jsonl",
        save_fields_dir: Optional[str] = None,
        publish_field: bool = True,
        reference: Optional[Dict[str, np.ndarray]] = None,
        state=None,
        verbose: bool = True,
    ):
        self.dtype = normalize_precision(precision)
        self.precision_name = "double" if self.dtype == tf.float64 else "single"
        self.field_shape = tuple(int(n) for n in field_shape)
        if len(self.field_shape) != 4 or self.field_shape[0] != 1:
            raise ValueError(
                "❌ error_estimator expects a single-patch velocity field of "
                f"shape (1, Nz, Ny, Nx); got {self.field_shape}."
            )
        _, self.Nz, self.Ny, self.Nx = self.field_shape

        cg_iters = sorted({int(k) for k in cg_iters if int(k) > 0})
        if not cg_iters:
            raise ValueError("❌ error_estimator.cg_iters needs at least one k > 0.")
        self.cg_iters: List[int] = cg_iters
        self.k_primary = self.cg_iters[-1]
        self.cg_tol = float(cg_tol)
        self.newton_steps = max(1, int(newton_steps))
        self.freq = int(freq)
        self.estimate_at_start = bool(estimate_at_start)
        self.operator_update_freq = int(operator_update_freq)
        if self.operator_update_freq < 0:
            raise ValueError("operator_update_freq must be non-negative.")
        self.operator_refresh_rel_change = float(operator_refresh_rel_change)
        self.disable_xla = bool(disable_xla)
        self.record_path = record_path
        self.save_fields_dir = save_fields_dir
        self.publish_field = bool(publish_field)
        self.state = state
        self.verbose = bool(verbose)

        self.idx_thk = int(idx_thk)
        self.idx_usurf = None if idx_usurf is None else int(idx_usurf)
        self.idx_water_level = None if idx_water_level is None else int(idx_water_level)
        self.rho_ice = float(rho_ice)
        self.rho_water = float(rho_water)
        self.water_level = tf.constant(float(water_level), self.dtype)
        self.V_s = tf.cast(V_s, self.dtype)
        self._damping = tf.constant(float(damping), self.dtype)
        self._topg = None if topg is None else tf.constant(np.asarray(topg), self.dtype)
        self._basin = (
            None
            if basin_mask is None
            else tf.constant(np.asarray(basin_mask, dtype=bool))
        )
        self._reference = None
        if reference is not None:
            self._reference = {
                key: tf.constant(np.asarray(value), self.dtype)
                for key, value in reference.items()
            }

        # Shadow identity mapping carrying the current iterate; the Hessian
        # operator and preconditioner are the ones used by cg_newton.
        zeros = tf.zeros(self.field_shape, self.dtype)
        self.shadow = MappingIdentity(list(bcs), zeros, zeros, self.precision_name)
        self.operator = build_energy_operator(
            hvp_mode=hvp_mode,
            probe_mode=probe_mode,
            basis_vertical=basis_vertical,
            precision=self.precision_name,
            cost_fn=cost_fn,
            mapping=self.shadow,
            verify_stencil=bool(hvp_verify),
            owner="error_estimator",
        )
        self.preconditioner = build_preconditioner(
            preconditioner,
            self.shadow,
            self.precision_name,
            self.operator.preconditioner_layout,
            preconditioner_options,
        )
        self.preconditioner.set_operator(self.operator)

        # Refresh bookkeeping for the frozen Hessian.
        self._u_prepared: Optional[Tuple[tf.Tensor, tf.Tensor]] = None
        self._inputs_prepared: Optional[tf.Tensor] = None
        self._calls_since_prepare = 0
        self._last_rel_change = 0.0
        self._last_fields: Optional[Dict[str, np.ndarray]] = None
        self.n_calls = 0
        self.last: Dict[str, float] = {}

        # One concrete trace for the solve: inputs are cast eagerly to the
        # estimator dtype and shapes are fixed by the grid (channels vary).
        field_spec = tf.TensorSpec(self.field_shape, self.dtype)
        inputs_spec = tf.TensorSpec([1, self.Ny, self.Nx, None], self.dtype)
        self._solve_graph = tf.function(
            self._solve_impl, input_signature=[field_spec, field_spec, inputs_spec]
        )

    # ------------------------------------------------------------------ #
    # Frozen Hessian                                                      #
    # ------------------------------------------------------------------ #

    def _needs_prepare(
        self,
        U: tf.Tensor,
        V: tf.Tensor,
        inputs: tf.Tensor,
    ) -> bool:
        """Refresh when inputs or velocity moved, or at an optional age cap.

        A Hessian frozen at an early iterate (an untrained network sits at the
        strain-rate regularisation floor) is useless a few hundred iterations
        later, so a fixed count alone is unsafe: the operator is rebuilt as soon
        as the RMS relative change of ``(U, V)`` since the last refresh exceeds
        ``operator_refresh_rel_change``. Any change in the energy inputs also
        rebuilds it. ``operator_update_freq > 0`` additionally sets a maximum
        operator age; zero disables that redundant cap for snapshot solves.
        """
        if self._u_prepared is None or self._inputs_prepared is None:
            return True
        if self.operator_update_freq == 1:
            return True
        inputs_changed = tf.reduce_any(tf.not_equal(inputs, self._inputs_prepared))
        if bool(inputs_changed.numpy()):
            return True
        if (
            self.operator_update_freq > 1
            and self._calls_since_prepare + 1 >= self.operator_update_freq
        ):
            return True
        U0, V0 = self._u_prepared
        change = tf.sqrt(
            tf.reduce_sum(tf.square(U - U0)) + tf.reduce_sum(tf.square(V - V0))
        )
        scale = tf.sqrt(tf.reduce_sum(tf.square(U0)) + tf.reduce_sum(tf.square(V0)))
        tiny = tf.cast(1e-30, self.dtype)
        self._last_rel_change = float((change / tf.maximum(scale, tiny)).numpy())
        return (
            not np.isfinite(self._last_rel_change)
            or self._last_rel_change > self.operator_refresh_rel_change
        )

    def _prepare(self, U: tf.Tensor, V: tf.Tensor, inputs: tf.Tensor) -> None:
        """Assign the iterate and rebuild the frozen Hessian and preconditioner.

        Deliberately eager, like ``OptimizerCGNewton``: each Hessian probe and
        the multigrid coarsening run as their own compiled functions. Wrapping
        them in one outer ``tf.function`` makes the graph optimizer inline every
        nested double-gradient graph, which takes minutes and tens of GB.
        """
        self.shadow.U.assign(U)
        self.shadow.V.assign(V)
        self.operator.prepare(inputs, self._damping)
        self.preconditioner.update(inputs, self._damping)
        token = self.preconditioner.synchronization_token()
        if token is None:
            token = self.operator.synchronization_token()
        if token is not None:
            tf.reshape(token, [-1])[0].numpy()

    # ------------------------------------------------------------------ #
    # Solve graph                                                         #
    # ------------------------------------------------------------------ #

    def _ice_mask(self, thk: tf.Tensor) -> tf.Tensor:
        ice = compute_node_ice_mask(thk)
        if self._basin is not None:
            ice = ice & self._basin
        return ice

    def _grounded(self, inputs: tf.Tensor, thk: tf.Tensor) -> Optional[tf.Tensor]:
        if self._topg is not None:
            topg = self._topg
        elif self.idx_usurf is not None:
            topg = inputs[0, :, :, self.idx_usurf] - thk
        else:
            return None
        if self.idx_water_level is not None:
            water_level = inputs[0, :, :, self.idx_water_level]
        else:
            water_level = self.water_level
        return grounded_mask(thk, topg, water_level, self.rho_ice, self.rho_water)

    def _delta_metrics(
        self,
        delta: tf.Tensor,
        u_s: tf.Tensor,
        v_s: tf.Tensor,
        ice: tf.Tensor,
        grounded: Optional[tf.Tensor],
        g: tf.Tensor,
        prefix: str,
    ) -> Tuple[Dict[str, tf.Tensor], tf.Tensor, tf.Tensor]:
        """Error statistics implied by a correction ``delta`` (flat theta)."""
        dU, dV = self.shadow.unflatten_theta(delta)
        node = ice[tf.newaxis, tf.newaxis]
        dU = tf.where(node, dU, tf.zeros_like(dU))
        dV = tf.where(node, dV, tf.zeros_like(dV))
        du_s, dv_s = get_velsurf(dU, dV, self.V_s)
        du_s, dv_s = du_s[0], dv_s[0]
        # -delta estimates the error and u + delta the reference velocity.
        out = summarize_velocity_error(
            du_s, dv_s, u_s + du_s, v_s + dv_s, ice, grounded, prefix
        )
        out[f"{prefix}_delta_l2"] = tf.norm(delta)
        out[f"{prefix}_decrement"] = -0.5 * tf.tensordot(delta, g, axes=1)
        return out, du_s, dv_s

    def _solve_impl(
        self, U: tf.Tensor, V: tf.Tensor, inputs: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        """Gradient, truncated PCG, chord steps and metrics (traced once)."""
        self.shadow.U.assign(U)
        self.shadow.V.assign(V)

        thk = inputs[0, :, :, self.idx_thk]
        ice = self._ice_mask(thk)
        grounded = self._grounded(inputs, thk)
        u_s, v_s = get_velsurf(U, V, self.V_s)
        u_s, v_s = u_s[0], v_s[0]

        cost, _, grad_theta = self.operator.cost_and_grad(inputs)
        g = self.shadow.flatten_theta(grad_theta)
        theta_flat = self.shadow.flatten_theta([U, V])

        def A(v: tf.Tensor) -> tf.Tensor:
            return self.operator.hvp(inputs, v, self._damping)

        pre = self.preconditioner.apply
        out: Dict[str, tf.Tensor] = {"cost": cost, "grad_l2": tf.norm(g)}

        # First Newton step: one PCG run, read off at every requested k.
        b = -g
        pcg = pcg_init(A, pre, b, tf.zeros_like(b))
        k_prev = 0
        for k in self.cg_iters:
            pcg = pcg_run(A, pre, pcg, k - k_prev, self.cg_tol)
            k_prev = k
            metrics, du_s, dv_s = self._delta_metrics(
                pcg.x, u_s, v_s, ice, grounded, g, f"k{k}"
            )
            out.update(metrics)
            out[f"k{k}_cg_iters"] = tf.cast(pcg.iters, self.dtype)
            out[f"k{k}_relres"] = pcg_relres(A, b, pcg.x)
        delta = pcg.x

        # Chord steps: gradient at u + delta, same frozen Hessian and
        # preconditioner. One Newton step contracts the shelf error of Glen
        # ice only by a factor ~0.2; the second removes most of the residual.
        for step in range(2, self.newton_steps + 1):
            _, g_step = self.operator.cost_grad_at(inputs, theta_flat + delta)
            pcg_step = pcg_init(A, pre, -g_step, tf.zeros_like(g_step))
            pcg_step = pcg_run(A, pre, pcg_step, self.k_primary, self.cg_tol)
            delta = delta + pcg_step.x
            metrics, du_s, dv_s = self._delta_metrics(
                delta, u_s, v_s, ice, grounded, g, f"step{step}"
            )
            out.update(metrics)
            out[f"step{step}_grad_l2"] = tf.norm(g_step)
            out[f"step{step}_cg_iters"] = tf.cast(pcg_step.iters, self.dtype)
            out[f"step{step}_relres"] = pcg_relres(A, -g_step, pcg_step.x)

        # Validation mode: true error of u and of the corrected u + delta.
        if self._reference is not None:
            u_ref, v_ref = self._reference["u_ref"], self._reference["v_ref"]
            out.update(
                summarize_velocity_error(
                    u_s - u_ref, v_s - v_ref, u_ref, v_ref, ice, grounded, "true"
                )
            )
            out.update(
                summarize_velocity_error(
                    u_s + du_s - u_ref,
                    v_s + dv_s - v_ref,
                    u_ref,
                    v_ref,
                    ice,
                    grounded,
                    "corr",
                )
            )

        out["field_err"] = tf.sqrt(tf.square(du_s) + tf.square(dv_s))
        out["field_u_s"] = u_s
        out["field_v_s"] = v_s
        out["field_du_s"] = du_s
        out["field_dv_s"] = dv_s
        return out

    # ------------------------------------------------------------------ #
    # Eager API                                                           #
    # ------------------------------------------------------------------ #

    @property
    def primary(self) -> str:
        """Key prefix of the reported estimate (last step, largest k)."""
        return (
            f"step{self.newton_steps}"
            if self.newton_steps > 1
            else f"k{self.k_primary}"
        )

    def estimate(self, U, V, inputs) -> Tuple[Dict[str, float], tf.Tensor, bool]:
        """Estimate the error of ``(U, V)``; returns ``(scalars, error_field, refreshed)``.

        XLA auto-clustering (switched on globally by the Adam module) is
        disabled while the estimator's own graphs are built and run: with it
        on, TensorFlow's pinned host-memory pool grows at every estimate until
        it hits its 64 GiB cap. The flag is read when a function's executor is
        first built, so the optimizer's graphs keep their setting.
        """
        jit_before = tf.config.optimizer.get_jit()
        if self.disable_xla:
            tf.config.optimizer.set_jit(False)
        try:
            return self._estimate(U, V, inputs)
        finally:
            if self.disable_xla:
                tf.config.optimizer.set_jit(jit_before)

    def _estimate(self, U, V, inputs) -> Tuple[Dict[str, float], tf.Tensor, bool]:
        t0 = time.perf_counter()
        U = tf.cast(U, self.dtype)
        V = tf.cast(V, self.dtype)
        inputs = tf.cast(inputs, self.dtype)
        first = self.n_calls == 0

        refreshed = self._needs_prepare(U, V, inputs)
        if refreshed:
            if first and self.verbose:
                print(
                    "[error_estimator] tracing Hessian probing and preconditioner ...",
                    flush=True,
                )
            self._prepare(U, V, inputs)
            self._u_prepared = (tf.identity(U), tf.identity(V))
            self._inputs_prepared = tf.identity(inputs)
            self._calls_since_prepare = 0
        else:
            self._calls_since_prepare += 1
        t1 = time.perf_counter()
        if first and self.verbose:
            print(
                f"[error_estimator] prepared in {t1 - t0:.1f}s; tracing PCG solve ...",
                flush=True,
            )

        out = self._solve_graph(U, V, inputs)
        fields = {
            key[6:]: out.pop(key) for key in list(out) if key.startswith("field_")
        }
        scalars = {key: float(value.numpy()) for key, value in out.items()}
        error_field = tf.constant(fields["err"].numpy(), tf.float32)  # synchronises
        if self.save_fields_dir:
            self._last_fields = {
                k: v.numpy().astype(np.float32) for k, v in fields.items()
            }
        t2 = time.perf_counter()

        scalars["wall_s"] = t2 - t0
        scalars["wall_prepare_s"] = t1 - t0
        scalars["wall_solve_s"] = t2 - t1
        scalars["rel_change_since_prepare"] = (
            0.0 if refreshed else self._last_rel_change
        )
        prefix = self.primary + "_"
        for key in list(scalars):
            if key.startswith(prefix):
                scalars["est_" + key[len(prefix) :]] = scalars[key]
        self.n_calls += 1
        return scalars, error_field, refreshed

    def record(
        self,
        iteration: int,
        scalars: Dict[str, float],
        field: tf.Tensor,
        refreshed: bool,
    ) -> None:
        """Log one estimate (JSONL, optional field files, ``state.err_est_*``)."""
        state = self.state
        step = getattr(state, "it", None) if state is not None else None
        sim_time = getattr(state, "t", None) if state is not None else None
        record = {
            "iter": int(iteration),
            "step": (
                int(step.numpy() if hasattr(step, "numpy") else step)
                if step is not None
                else None
            ),
            "t": float(sim_time.numpy()) if hasattr(sim_time, "numpy") else None,
            "refreshed": bool(refreshed),
            "k_primary": self.k_primary,
            "newton_steps": self.newton_steps,
            **scalars,
        }
        self.last = record

        if self.record_path:
            try:
                with open(self.record_path, "a") as fh:
                    fh.write(json.dumps(record) + "\n")
            except OSError:
                pass  # never let logging crash a run

        if self.save_fields_dir and self._last_fields is not None:
            try:
                os.makedirs(self.save_fields_dir, exist_ok=True)
                np.savez_compressed(
                    os.path.join(
                        self.save_fields_dir, f"fields_{int(iteration) + 1:06d}.npz"
                    ),
                    iteration=int(iteration),
                    **self._last_fields,
                )
            except OSError:
                pass

        if state is not None:
            for key in (
                "median_rel_grounded",
                "median_rel_floating",
                "median_rel_all",
                "rmse_all",
                "decrement",
            ):
                if "est_" + key in scalars:
                    setattr(state, "err_est_" + key, scalars["est_" + key])
            state.err_est_iter = int(iteration)
            if self.publish_field:
                state.err_est_surf = field

        if self.verbose:
            self._print_line(record)

    def _print_line(self, rec: Dict[str, float]) -> None:
        def fmt(key: str, spec: str = "6.2f") -> str:
            value = rec.get(key)
            return "   nan" if value is None else format(value, spec)

        line = (
            f"[error_estimator] iter={rec['iter']:>6d} "
            f"est(g/f/all)={fmt('est_median_rel_grounded')}/"
            f"{fmt('est_median_rel_floating')}/{fmt('est_median_rel_all')}% "
            f"rmse={fmt('est_rmse_all', '8.3f')} m/yr "
            f"decr={fmt('est_decrement', '9.3e')} "
            f"cg={int(rec.get(f'k{self.k_primary}_cg_iters', 0))} "
            f"relres={fmt(f'k{self.k_primary}_relres', '8.2e')} "
            f"{'refresh' if rec['refreshed'] else 'reuse  '} wall={rec['wall_s']:.2f}s"
        )
        if "true_median_rel_grounded" in rec:
            line += (
                f" | true(g/f)={fmt('true_median_rel_grounded')}/{fmt('true_median_rel_floating')}%"
                f" corr(g/f)={fmt('corr_median_rel_grounded')}/{fmt('corr_median_rel_floating')}%"
            )
        print(line, flush=True)

    def estimate_and_record(self, U, V, inputs, iteration) -> Dict[str, float]:
        if hasattr(iteration, "numpy"):
            iteration = int(iteration.numpy())
        scalars, field, refreshed = self.estimate(U, V, inputs)
        self.record(int(iteration), scalars, field, refreshed)
        return scalars

    def estimate_and_record_py(self, U, V, inputs, iteration) -> np.int32:
        """``tf.py_function`` body used by ``Optimizer._estimate_error``."""
        self.estimate_and_record(U, V, inputs, iteration)
        return np.int32(0)
