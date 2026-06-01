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

    name: str  # e.g. "thk", "tau_ref"
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
            "tau_ref": 3,
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

        # Precompute θ-space bounds for optimizer consumption.
        self._L_list: List[tf.Tensor] = []
        self._U_list: List[tf.Tensor] = []
        for spec, theta, tform in zip(self.vars, self._theta, self.transforms):
            Ls, Us = tform.theta_bounds(
                spec.lower_bound, spec.upper_bound, dtype=theta.dtype, eps=self.eps
            )
            self._L_list.append(tf.fill(theta.shape, Ls))
            self._U_list.append(tf.fill(theta.shape, Us))

        # Bounds are static during a DA phase. Flatten once; the bounded optimizer
        # queries these inside every accepted iteration and line search setup.
        self._L_flat = tf.concat([tf.reshape(Li, [-1]) for Li in self._L_list], axis=0)
        self._U_flat = tf.concat([tf.reshape(Ui, [-1]) for Ui in self._U_list], axis=0)

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
        """
        Register a Python callback to be run every ``out_freq`` accepted iterations.

        The DA-specific optimizer runs its outer minimize loop eagerly, so it can call
        this callback directly from Python via ``maybe_run_step_callback`` instead of
        crossing the host/device boundary through ``tf.py_function``.
        """
        self._da_step_callback = callback
        self._da_out_freq = int(out_freq)

    def clear_step_callback(self) -> None:
        self._da_step_callback = None
        self._da_out_freq = 0

    def maybe_run_step_callback(self, iter_zero_based: int) -> None:
        """
        Eager-only accepted-step callback runner.

        ``iter_zero_based`` is the optimizer iteration index of the accepted step.
        The registered callback receives a one-based accepted-iteration count so that
        output iteration numbering is intuitive and aligns with stored cost histories.
        """
        if self._da_step_callback is None or self._da_out_freq <= 0:
            return

        accepted_iter = int(iter_zero_based) + 1
        if accepted_iter % int(self._da_out_freq) != 0:
            return

        self._da_step_callback(accepted_iter)

    @tf.function(reduce_retracing=True)
    def on_step_end(self, it: tf.Tensor) -> tf.Tensor:
        """
        Compatibility no-op for code paths that still call ``on_step_end`` inside a
        traced optimizer loop. The DA optimizer uses ``maybe_run_step_callback``.
        """
        del it
        return tf.constant(0, dtype=tf.int32)

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
        return self._L_flat, self._U_flat

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

    # ------- Input channel patching ------------------------------------------

    @tf.function(reduce_retracing=True, jit_compile=False)
    def apply_theta_to_inputs(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Patch BHWC inputs with current physical-space values for selected fields.

        This builds the output channels once instead of repeatedly slicing and
        concatenating the full BHWC tensor for each inverted variable.
        """
        patches = []
        for idx, spec in enumerate(self.vars):
            ch = self.field_to_channel.get(spec.name, None)
            if ch is not None:
                patches.append((idx, int(ch)))

        if not patches:
            return inputs

        n_channels = inputs.shape[-1]
        if n_channels is None:
            n_channels = max(self.field_to_channel.values()) + 1

        n_channels = int(n_channels)
        channels = tf.unstack(inputs, num=n_channels, axis=-1)

        shape = tf.shape(inputs)
        B, H, W = shape[0], shape[1], shape[2]

        for idx, ch in patches:

            val = tf.cast(self._theta_to_field(idx), inputs.dtype)

            channels[ch] = tf.broadcast_to(
                tf.reshape(val, [1, H, W]),
                [B, H, W],
            )

        return tf.stack(channels, axis=-1)

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

