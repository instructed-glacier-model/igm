#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Energy operators used by the Newton-CG ice-flow solver."""

import warnings
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import tensorflow as tf

from .banded import (
    COMPONENT_CENTER_KEY,
    OFFSETS,
    ComponentBandedOperator,
    as_dtype,
    build_component_selectors,
    extract_component_bands,
    periodic_axes,
)
from .ssa_banded import (
    SSA_BAND_KEYS,
    SSABandedStencil,
    allocate_ssa_bands,
    build_ssa_selectors,
    extract_ssa_bands,
    supports_compact_ssa,
)
from .molho_banded import (
    MOLHO_STENCIL_KEY,
    allocate_molho_bands,
    build_molho_stencil,
    extract_symmetric_bands,
    supports_compact_molho,
)


class Operator(ABC):
    """Gradient, trial-gradient, and Hessian-vector product contract."""

    name = "operator"
    preconditioner_layout = "component"

    @abstractmethod
    def cost_and_grad(self, inputs: tf.Tensor):
        """Return cost, velocity gradient, and parameter gradient."""

    @abstractmethod
    def cost_grad_at(self, inputs: tf.Tensor, theta_flat: tf.Tensor):
        """Return cost and flat gradient at an arbitrary parameter vector."""

    def cost_at(self, inputs: tf.Tensor, theta_flat: tf.Tensor) -> tf.Tensor:
        """Return cost at an arbitrary parameter vector.

        Operators may override this to avoid constructing a gradient tape in
        value-only line-search evaluations.
        """
        cost, _ = self.cost_grad_at(inputs, theta_flat)
        return cost

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

    def synchronization_token(self) -> Optional[tf.Tensor]:
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

    @tf.function(reduce_retracing=True)
    def cost_at(self, inputs: tf.Tensor, theta_flat: tf.Tensor) -> tf.Tensor:
        U, V = self.map.unflatten_theta(theta_flat)
        for apply_bc in self.map.apply_bcs:
            U, V = apply_bc(U, V)
        return self.cost_fn(U, V, inputs)

    def hvp(
        self,
        inputs: tf.Tensor,
        v_flat: tf.Tensor,
        damping: tf.Tensor,
    ) -> tf.Tensor:
        damping = tf.convert_to_tensor(damping, self.precision)
        return self._hvp(inputs, v_flat, damping)

    def forward_hvp(
        self,
        inputs: tf.Tensor,
        v_flat: tf.Tensor,
        damping: tf.Tensor,
    ) -> tf.Tensor:
        damping = tf.convert_to_tensor(damping, self.precision)
        return self._forward_hvp(inputs, v_flat, damping)

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

    @tf.function(reduce_retracing=True)
    def _forward_hvp(
        self,
        inputs: tf.Tensor,
        v_flat: tf.Tensor,
        damping: tf.Tensor,
    ) -> tf.Tensor:
        theta_flat = self.map.flatten_theta(self.map.get_theta())
        with tf.autodiff.ForwardAccumulator(theta_flat, v_flat) as accumulator:
            with tf.GradientTape() as tape:
                tape.watch(theta_flat)
                U, V = self.map.unflatten_theta(theta_flat)
                for apply_bc in self.map.apply_bcs:
                    U, V = apply_bc(U, V)
                cost = self.cost_fn(U, V, inputs)
            grad_flat = tape.gradient(cost, theta_flat)
        return accumulator.jvp(grad_flat) + damping * v_flat


class _BandedADOperatorBase(Operator):
    """Shared autodiff probing for frozen banded Hessians."""

    def __init__(
        self,
        cost_fn,
        mapping,
        precision: str,
        verify_stencil: bool,
        probe_mode: str,
    ):
        self.map = mapping
        self.precision = as_dtype(precision)
        self._ad = ADOperator(cost_fn, mapping, precision)
        self._zero = tf.constant(0.0, self.precision)
        self._verify_stencil = bool(verify_stencil)
        self._verified = False

        if probe_mode not in ("fd", "autodiff", "forward"):
            raise ValueError(
                f"Unknown cg_newton.probe_mode: <{probe_mode}>. Use 'fd', "
                "'autodiff', or 'forward'."
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

    def cost_at(self, inputs: tf.Tensor, theta_flat: tf.Tensor) -> tf.Tensor:
        return self._ad.cost_at(inputs, theta_flat)

    def _probe_hvp(self, inputs: tf.Tensor, v_flat: tf.Tensor) -> tf.Tensor:
        if self._probe_mode == "autodiff":
            return self._ad.hvp(inputs, v_flat, self._zero)
        if self._probe_mode == "forward":
            return self._ad.forward_hvp(inputs, v_flat, self._zero)

        theta_flat = self.map.flatten_theta(self.map.get_theta())
        epsilon = tf.cast(1e-6, self.precision) * (
            tf.cast(1.0, self.precision) + tf.norm(theta_flat)
        ) / (tf.norm(v_flat) + tf.cast(1e-30, self.precision))
        _, grad_plus = self._ad.cost_grad_at(
            inputs, theta_flat + epsilon * v_flat
        )
        _, grad_minus = self._ad.cost_grad_at(
            inputs, theta_flat - epsilon * v_flat
        )
        return (grad_plus - grad_minus) / (2.0 * epsilon)

    def _verify_if_requested(
        self, inputs: tf.Tensor, damping: tf.Tensor
    ) -> None:
        if self._verify_stencil and not self._verified:
            self.verify(inputs, damping)
            self._verified = True

    def verify(self, inputs: tf.Tensor, damping: tf.Tensor) -> float:
        theta_flat = self.map.flatten_theta(self.map.get_theta())
        vector = tf.random.normal(tf.shape(theta_flat), dtype=self.precision)
        exact = self._ad.hvp(inputs, vector, damping)
        approximate = self.hvp(inputs, vector, damping)
        relative_error = float(
            tf.norm(exact - approximate) / (tf.norm(exact) + 1e-30)
        )
        if relative_error > 1e-6:
            warnings.warn(
                f"{type(self).__name__} differs from the exact Hessian by "
                f"relative {relative_error:.2e}; use hvp_mode='autodiff' if "
                "the energy couples beyond the 9-point stencil.",
                RuntimeWarning,
            )
        return relative_error


class BandedADOperator(_BandedADOperatorBase):
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
        super().__init__(
            cost_fn,
            mapping,
            precision,
            verify_stencil,
            probe_mode,
        )

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
        self._prepared = False

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

        self._verify_if_requested(inputs, damping)

    def hvp(
        self,
        inputs: tf.Tensor,
        v_flat: tf.Tensor,
        damping: tf.Tensor,
    ) -> tf.Tensor:
        del inputs
        if not self._prepared:
            raise RuntimeError(
                "BandedADOperator.prepare() must be called before hvp()."
            )
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
        damping = tf.cast(damping, self.precision)
        identity = tf.eye(self.n_components, dtype=self.precision)
        identity = identity[tf.newaxis, :, :, tf.newaxis, tf.newaxis]
        center = self._component_bands[0] + damping * identity
        return {COMPONENT_CENTER_KEY: center}

    def synchronization_token(self) -> tf.Tensor:
        return self._component_bands

    def _split_components(self, flat: tf.Tensor) -> tf.Tensor:
        u, v = self.map.unflatten_theta(flat)
        return tf.concat([u, v], axis=1)

    def _join_components(self, components: tf.Tensor) -> tf.Tensor:
        return self.map.flatten_theta(
            [components[:, : self.Nz], components[:, self.Nz :]]
        )

    def _component_apply(
        self,
        inputs: tf.Tensor,
        components: tf.Tensor,
    ) -> tf.Tensor:
        v_flat = self._join_components(components)
        h_flat = self._probe_hvp(inputs, v_flat)
        return self._split_components(h_flat)


class MOLHOBandedADOperator(_BandedADOperatorBase):
    """MOLHO Hessian stored as a compact symmetric component stencil."""

    name = "molho_banded_ad"
    preconditioner_layout = "compact_molho"

    def __init__(
        self,
        cost_fn,
        mapping,
        precision: str = "float32",
        verify_stencil: bool = False,
        probe_mode: str = "autodiff",
    ):
        if not supports_compact_molho(mapping):
            raise ValueError(
                "MOLHOBandedADOperator requires an identity mapping with "
                "shape (B, 2, Ny, Nx)."
            )
        super().__init__(
            cost_fn,
            mapping,
            precision,
            verify_stencil,
            probe_mode,
        )

        self.B, self.Nz, self.Ny, self.Nx = tuple(mapping.shape)
        self.n_components = 4
        self.periodic_y, self.periodic_x = periodic_axes(mapping)
        self._center, self._edges = allocate_molho_bands(
            mapping,
            self.precision,
        )
        self._banded = build_molho_stencil(
            mapping,
            self._center,
            self._edges,
        )
        self._prepared = False

    def prepare(self, inputs: tf.Tensor, damping: tf.Tensor) -> None:
        center, edges = extract_symmetric_bands(
            lambda components: self._component_apply(inputs, components),
            self.B,
            self.n_components,
            self.Ny,
            self.Nx,
            self.precision,
            periodic_y=self.periodic_y,
            periodic_x=self.periodic_x,
        )
        self._center.assign(center)
        self._edges.assign(edges)
        self._prepared = True
        self._verify_if_requested(inputs, damping)

    def hvp(
        self,
        inputs: tf.Tensor,
        v_flat: tf.Tensor,
        damping: tf.Tensor,
    ) -> tf.Tensor:
        del inputs
        if not self._prepared:
            raise RuntimeError(
                "MOLHOBandedADOperator.prepare() must be called before hvp()."
            )
        result = self._join_components(
            self._banded.apply(self._split_components(v_flat))
        )
        return result + tf.cast(damping, self.precision) * v_flat

    def assemble_bands(self, inputs: tf.Tensor, damping: tf.Tensor):
        del inputs
        if not self._prepared:
            raise RuntimeError(
                "MOLHOBandedADOperator.prepare() must be called before "
                "assemble_bands()."
            )
        return {
            COMPONENT_CENTER_KEY: self._banded.center_matrix(damping),
            MOLHO_STENCIL_KEY: self._banded,
        }

    def synchronization_token(self) -> tf.Tensor:
        return self._edges

    def _split_components(self, flat: tf.Tensor) -> tf.Tensor:
        u, v = self.map.unflatten_theta(flat)
        return tf.concat([u, v], axis=1)

    def _join_components(self, components: tf.Tensor) -> tf.Tensor:
        return self.map.flatten_theta(
            [components[:, : self.Nz], components[:, self.Nz :]]
        )

    def _component_apply(
        self,
        inputs: tf.Tensor,
        components: tf.Tensor,
    ) -> tf.Tensor:
        h_flat = self._probe_hvp(inputs, self._join_components(components))
        return self._split_components(h_flat)


class SSABandedADOperator(_BandedADOperatorBase):
    """Nonperiodic SSA Hessian frozen as four scalar 9-point stencils."""

    name = "ssa_banded_ad"
    preconditioner_layout = "compact_ssa"

    def __init__(
        self,
        cost_fn,
        mapping,
        precision: str = "float32",
        verify_stencil: bool = False,
        probe_mode: str = "fd",
    ):
        if not supports_compact_ssa(mapping):
            raise ValueError(
                "SSABandedADOperator requires a nonperiodic identity mapping "
                "with shape (B, 1, Ny, Nx)."
            )
        super().__init__(
            cost_fn,
            mapping,
            precision,
            verify_stencil,
            probe_mode,
        )

        self.batch_size, _, self.ny, self.nx = tuple(mapping.shape)
        self._color, self._neighbour_colors = build_ssa_selectors(
            self.ny, self.nx
        )
        self._bands = allocate_ssa_bands(
            self.batch_size,
            self.ny,
            self.nx,
            self.precision,
            "ssa_band",
        )
        self._banded = SSABandedStencil(self._bands)
        self._prepared = False

    def prepare(self, inputs: tf.Tensor, damping: tf.Tensor) -> None:
        bands = extract_ssa_bands(
            lambda u, v: self._field_apply(inputs, u, v),
            self.batch_size,
            self.ny,
            self.nx,
            self.precision,
            self._color,
            self._neighbour_colors,
        )
        for key in SSA_BAND_KEYS:
            self._bands[key].assign(bands[key])
        self._prepared = True
        self._verify_if_requested(inputs, damping)

    def hvp(
        self, inputs: tf.Tensor, v_flat: tf.Tensor, damping: tf.Tensor
    ) -> tf.Tensor:
        del inputs
        if not self._prepared:
            raise RuntimeError(
                "SSABandedADOperator.prepare() must be called before hvp()."
            )
        u, v = self._split(v_flat)
        h_u, h_v = self._banded.apply(u, v)
        return self._join(h_u, h_v) + tf.cast(damping, self.precision) * v_flat

    def assemble_bands(self, inputs: tf.Tensor, damping: tf.Tensor):
        del inputs
        if not self._prepared:
            raise RuntimeError(
                "SSABandedADOperator.prepare() must be called before "
                "assemble_bands()."
            )
        damping = tf.cast(damping, self.precision)
        result = {}
        for key in SSA_BAND_KEYS:
            bands = tf.convert_to_tensor(self._bands[key])
            if key in (("u", "u"), ("v", "v")):
                bands = tf.concat([bands[0:1] + damping, bands[1:]], axis=0)
            result[key] = bands
        return result

    def synchronization_token(self) -> tf.Tensor:
        return self._bands[("u", "u")]

    def _split(self, flat: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        u, v = self.map.unflatten_theta(flat)
        return u[:, 0], v[:, 0]

    def _join(self, u: tf.Tensor, v: tf.Tensor) -> tf.Tensor:
        return self.map.flatten_theta([u[:, tf.newaxis], v[:, tf.newaxis]])

    def _field_apply(
        self, inputs: tf.Tensor, u: tf.Tensor, v: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        return self._split(self._probe_hvp(inputs, self._join(u, v)))
