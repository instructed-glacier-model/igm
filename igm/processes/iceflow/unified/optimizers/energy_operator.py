#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Energy operators used by the Newton-CG ice-flow solver."""

import warnings
from abc import ABC, abstractmethod
from typing import Dict, Optional

import tensorflow as tf

from .banded import (
    COMPONENT_BANDS_KEY,
    OFFSETS,
    ComponentBandedOperator,
    as_dtype,
    build_component_selectors,
    extract_component_bands,
    periodic_axes,
)


class Operator(ABC):
    """Gradient, trial-gradient, and Hessian-vector product contract."""

    name = "operator"

    @abstractmethod
    def cost_and_grad(self, inputs: tf.Tensor):
        """Return cost, velocity gradient, and parameter gradient."""

    @abstractmethod
    def cost_grad_at(self, inputs: tf.Tensor, theta_flat: tf.Tensor):
        """Return cost and flat gradient at an arbitrary parameter vector."""

    @abstractmethod
    def hvp(
        self,
        inputs: tf.Tensor,
        v_flat: tf.Tensor,
        damping: tf.Tensor,
    ) -> tf.Tensor:
        """Return ``(H + damping I) v`` flattened like theta."""

    def prepare(self, inputs: tf.Tensor, damping: tf.Tensor) -> None:
        del inputs, damping

    def assemble_bands(
        self,
        inputs: tf.Tensor,
        damping: tf.Tensor,
    ) -> Optional[Dict[str, tf.Tensor]]:
        del inputs, damping
        return None


class ADOperator(Operator):
    """Exact gradient and Hessian-vector products from autodiff."""

    name = "ad"

    def __init__(self, cost_fn, mapping, precision: str = "float32"):
        self.cost_fn = cost_fn
        self.map = mapping
        self.precision = as_dtype(precision)

    def cost_and_grad(self, inputs: tf.Tensor):
        theta = self.map.get_theta()
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            for theta_i in theta:
                tape.watch(theta_i)
            U, V = self.map.get_UV(inputs)
            cost = self.cost_fn(U, V, inputs)
        grads = tape.gradient(cost, [U, V] + theta)
        return cost, tuple(grads[:2]), grads[2:]

    @tf.function(reduce_retracing=True)
    def cost_grad_at(self, inputs: tf.Tensor, theta_flat: tf.Tensor):
        with tf.GradientTape() as tape:
            tape.watch(theta_flat)
            U, V = self.map.unflatten_theta(theta_flat)
            for apply_bc in self.map.apply_bcs:
                U, V = apply_bc(U, V)
            cost = self.cost_fn(U, V, inputs)
        return cost, tape.gradient(cost, theta_flat)

    def hvp(
        self,
        inputs: tf.Tensor,
        v_flat: tf.Tensor,
        damping: tf.Tensor,
    ) -> tf.Tensor:
        damping = tf.convert_to_tensor(damping, self.precision)
        return self._hvp(inputs, v_flat, damping)

    @tf.function(reduce_retracing=True)
    def _hvp(
        self,
        inputs: tf.Tensor,
        v_flat: tf.Tensor,
        damping: tf.Tensor,
    ) -> tf.Tensor:
        theta_flat = self.map.flatten_theta(self.map.get_theta())

        with tf.GradientTape() as outer_tape:
            outer_tape.watch(theta_flat)
            with tf.GradientTape() as inner_tape:
                inner_tape.watch(theta_flat)
                U, V = self.map.unflatten_theta(theta_flat)
                for apply_bc in self.map.apply_bcs:
                    U, V = apply_bc(U, V)
                cost = self.cost_fn(U, V, inputs)
            grad_flat = inner_tape.gradient(cost, theta_flat)

        hv_flat = outer_tape.gradient(
            grad_flat,
            theta_flat,
            output_gradients=v_flat,
        )
        return hv_flat + damping * v_flat


class BandedADOperator(Operator):
    """Hessian frozen as a periodic-aware component-block 9-point stencil."""

    name = "banded_ad"

    def __init__(
        self,
        cost_fn,
        mapping,
        precision: str = "float32",
        verify_stencil: bool = False,
        probe_mode: str = "fd",
    ):
        self.cost_fn = cost_fn
        self.map = mapping
        self.precision = as_dtype(precision)
        self._ad = ADOperator(cost_fn, mapping, precision)

        field_shape = getattr(mapping, "shape", None)
        if (
            getattr(mapping, "name", "") != "identity"
            or field_shape is None
            or len(field_shape) != 4
        ):
            raise ValueError(
                "hvp_mode='banded' requires an identity velocity mapping with "
                f"shape (B, Nz, Ny, Nx); got mapping "
                f"'{getattr(mapping, 'name', '?')}' with shape {field_shape}."
            )

        self.B, self.Nz, self.Ny, self.Nx = tuple(field_shape)
        self.n_components = 2 * self.Nz
        self.periodic_y, self.periodic_x = periodic_axes(mapping)
        self._colors, self._neighbour_colors, self._n_colors = (
            build_component_selectors(
                self.Ny,
                self.Nx,
                periodic_y=self.periodic_y,
                periodic_x=self.periodic_x,
            )
        )

        band_shape = (
            len(OFFSETS),
            self.B,
            self.n_components,
            self.n_components,
            self.Ny,
            self.Nx,
        )
        self._component_bands = tf.Variable(
            tf.zeros(band_shape, self.precision),
            trainable=False,
            name="band_components",
        )
        self._banded = ComponentBandedOperator(
            self._component_bands,
            periodic_y=self.periodic_y,
            periodic_x=self.periodic_x,
        )
        self._zero = tf.constant(0.0, self.precision)
        self._verify_stencil = bool(verify_stencil)
        self._verified = False
        self._prepared = False

        if probe_mode not in ("fd", "autodiff"):
            raise ValueError(
                f"Unknown cg_newton.probe_mode: <{probe_mode}>. Use 'fd' or "
                "'autodiff'."
            )
        self._probe_mode = probe_mode
        if probe_mode == "fd" and self.precision != tf.float64:
            warnings.warn(
                "probe_mode='fd' differences the gradient and requires "
                "numerics.precision=double for a reliable stencil.",
                RuntimeWarning,
            )

    def cost_and_grad(self, inputs: tf.Tensor):
        return self._ad.cost_and_grad(inputs)

    def cost_grad_at(self, inputs: tf.Tensor, theta_flat: tf.Tensor):
        return self._ad.cost_grad_at(inputs, theta_flat)

    def prepare(self, inputs: tf.Tensor, damping: tf.Tensor) -> None:
        bands = extract_component_bands(
            lambda components: self._component_apply(inputs, components),
            self.B,
            self.n_components,
            self.Ny,
            self.Nx,
            self.precision,
            self._colors,
            self._neighbour_colors,
            self._n_colors,
        )
        self._component_bands.assign(bands)
        self._prepared = True

        if self._verify_stencil and not self._verified:
            self.verify(inputs, damping)
            self._verified = True

    def hvp(
        self,
        inputs: tf.Tensor,
        v_flat: tf.Tensor,
        damping: tf.Tensor,
    ) -> tf.Tensor:
        del inputs
        if not self._prepared:
            raise RuntimeError("BandedADOperator.prepare() must be called before hvp().")
        result = self._join_components(
            self._banded.apply(self._split_components(v_flat))
        )
        return result + tf.cast(damping, self.precision) * v_flat

    def assemble_bands(self, inputs: tf.Tensor, damping: tf.Tensor):
        del inputs
        if not self._prepared:
            raise RuntimeError(
                "BandedADOperator.prepare() must be called before assemble_bands()."
            )
        bands = tf.convert_to_tensor(self._component_bands)
        identity = tf.eye(self.n_components, dtype=self.precision)
        identity = identity[tf.newaxis, :, :, tf.newaxis, tf.newaxis]
        damping = tf.cast(damping, self.precision)
        bands = tf.concat([bands[0:1] + damping * identity, bands[1:]], axis=0)
        return {COMPONENT_BANDS_KEY: bands}

    def _split_components(self, flat: tf.Tensor) -> tf.Tensor:
        u, v = self.map.unflatten_theta(flat)
        return tf.concat([u, v], axis=1)

    def _join_components(self, components: tf.Tensor) -> tf.Tensor:
        return self.map.flatten_theta(
            [components[:, : self.Nz], components[:, self.Nz :]]
        )

    def _hvp_fd(self, inputs: tf.Tensor, v_flat: tf.Tensor) -> tf.Tensor:
        theta_flat = self.map.flatten_theta(self.map.get_theta())
        eps = tf.cast(1e-6, self.precision) * (
            tf.cast(1.0, self.precision) + tf.norm(theta_flat)
        ) / (tf.norm(v_flat) + tf.cast(1e-30, self.precision))
        _, grad_plus = self._ad.cost_grad_at(inputs, theta_flat + eps * v_flat)
        _, grad_minus = self._ad.cost_grad_at(inputs, theta_flat - eps * v_flat)
        return (grad_plus - grad_minus) / (2.0 * eps)

    def _component_apply(
        self,
        inputs: tf.Tensor,
        components: tf.Tensor,
    ) -> tf.Tensor:
        v_flat = self._join_components(components)
        if self._probe_mode == "fd":
            h_flat = self._hvp_fd(inputs, v_flat)
        else:
            h_flat = self._ad.hvp(inputs, v_flat, self._zero)
        return self._split_components(h_flat)

    def verify(self, inputs: tf.Tensor, damping: tf.Tensor) -> float:
        n = 2 * self.B * self.Nz * self.Ny * self.Nx
        vector = tf.random.normal((n,), dtype=self.precision)
        exact = self._ad.hvp(inputs, vector, damping)
        approximate = self.hvp(inputs, vector, damping)
        relative_error = float(
            tf.norm(exact - approximate) / (tf.norm(exact) + 1e-30)
        )
        if relative_error > 1e-6:
            warnings.warn(
                "BandedADOperator differs from the exact Hessian by relative "
                f"{relative_error:.2e}; use hvp_mode='autodiff' if the energy "
                "couples beyond the 9-point stencil.",
                RuntimeWarning,
            )
        return relative_error
