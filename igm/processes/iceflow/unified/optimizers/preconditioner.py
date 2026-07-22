#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Preconditioners supported by the Newton-CG ice-flow solver."""

from typing import Callable, Optional

import tensorflow as tf

from .banded import (
    COMPONENT_BANDS_KEY,
    as_dtype,
    build_component_selectors,
    extract_component_bands,
    periodic_axes,
)


class Preconditioner:
    name = "identity"
    needs_operator = False

    def set_operator(self, operator) -> None:
        del operator

    def update(self, inputs: tf.Tensor, damping: tf.Tensor) -> None:
        del inputs, damping

    def apply(self, residual_flat: tf.Tensor) -> tf.Tensor:
        return residual_flat


class IdentityPreconditioner(Preconditioner):
    pass


class ComponentBlockJacobiPreconditioner(Preconditioner):
    """SPD point-block Jacobi over every U/V vertical component."""

    name = "block_jacobi"
    needs_operator = True

    def __init__(
        self,
        mapping,
        field_shape,
        precision="float32",
        eigenvalue_floor=1e-12,
    ):
        self.map = mapping
        self.dtype = as_dtype(precision)
        self.B, self.Nz, self.Ny, self.Nx = tuple(field_shape)
        self.n_components = 2 * self.Nz
        self.eigenvalue_floor = tf.constant(eigenvalue_floor, self.dtype)

        self.periodic_y, self.periodic_x = periodic_axes(mapping)
        self._colors, self._neighbour_colors, self._n_colors = (
            build_component_selectors(
                self.Ny,
                self.Nx,
                periodic_y=self.periodic_y,
                periodic_x=self.periodic_x,
            )
        )
        self._hvp: Optional[Callable[..., tf.Tensor]] = None
        self._assemble = None

        self.inverse_center = tf.Variable(
            tf.zeros(
                (
                    self.B,
                    self.n_components,
                    self.n_components,
                    self.Ny,
                    self.Nx,
                ),
                self.dtype,
            ),
            trainable=False,
            name="component_block_jacobi_inverse",
        )

    def set_operator(self, operator) -> None:
        self._hvp = operator.hvp if hasattr(operator, "hvp") else operator
        self._assemble = getattr(operator, "assemble_bands", None)

    def update(self, inputs: tf.Tensor, damping: tf.Tensor) -> None:
        if self._hvp is None:
            raise RuntimeError("block_jacobi operator was not registered.")

        bands = self._assemble(inputs, damping) if self._assemble is not None else None
        if isinstance(bands, dict) and COMPONENT_BANDS_KEY in bands:
            component_bands = bands[COMPONENT_BANDS_KEY]
            center = component_bands[0]
        else:
            zero_damping = tf.constant(0.0, self.dtype)
            component_bands = extract_component_bands(
                lambda components: self._component_apply(
                    inputs,
                    components,
                    zero_damping,
                ),
                self.B,
                self.n_components,
                self.Ny,
                self.Nx,
                self.dtype,
                self._colors,
                self._neighbour_colors,
                self._n_colors,
            )
            identity = tf.eye(self.n_components, dtype=self.dtype)
            identity = identity[tf.newaxis, :, :, tf.newaxis, tf.newaxis]
            center = component_bands[0] + tf.cast(damping, self.dtype) * identity
        matrix = tf.transpose(center, [0, 3, 4, 1, 2])
        matrix = 0.5 * (matrix + tf.linalg.matrix_transpose(matrix))

        eigenvalues, eigenvectors = tf.linalg.eigh(matrix)
        scale = tf.reduce_max(tf.abs(eigenvalues), axis=-1, keepdims=True)
        tiny_value = 1e-300 if self.dtype == tf.float64 else 1e-30
        floor = tf.maximum(
            self.eigenvalue_floor * scale,
            tf.cast(tiny_value, self.dtype),
        )
        eigenvalues = tf.maximum(eigenvalues, floor)
        inverse_matrix = tf.linalg.matmul(
            eigenvectors / eigenvalues[..., tf.newaxis, :],
            eigenvectors,
            transpose_b=True,
        )
        self.inverse_center.assign(
            tf.transpose(inverse_matrix, [0, 3, 4, 1, 2])
        )

    def _split(self, flat: tf.Tensor) -> tf.Tensor:
        u, v = self.map.unflatten_theta(flat)
        return tf.concat([u, v], axis=1)

    def _join(self, components: tf.Tensor) -> tf.Tensor:
        return self.map.flatten_theta(
            [components[:, : self.Nz], components[:, self.Nz :]]
        )

    def _component_apply(
        self,
        inputs: tf.Tensor,
        components: tf.Tensor,
        damping: tf.Tensor,
    ) -> tf.Tensor:
        result_flat = self._hvp(inputs, self._join(components), damping)
        return self._split(result_flat)

    @tf.function(reduce_retracing=True)
    def apply(self, residual_flat: tf.Tensor) -> tf.Tensor:
        components = tf.cast(self._split(residual_flat), self.dtype)
        solution = tf.einsum(
            "boiyx,biyx->boyx",
            self.inverse_center,
            components,
        )
        return tf.cast(self._join(solution), residual_flat.dtype)


def build_preconditioner(
    kind,
    mapping,
    precision="float32",
):
    kind = (kind or "none").lower()
    if kind in ("none", "identity"):
        return IdentityPreconditioner()

    if kind != "block_jacobi":
        raise ValueError(
            f"Unknown preconditioner kind: <{kind}>. Use 'none' or "
            "'block_jacobi'."
        )

    field_shape = getattr(mapping, "shape", None)
    if (
        getattr(mapping, "name", "") != "identity"
        or field_shape is None
        or len(field_shape) != 4
    ):
        raise ValueError(
            "block_jacobi requires an identity velocity mapping with shape "
            f"(B, Nz, Ny, Nx); got mapping '{getattr(mapping, 'name', '?')}' "
            f"with shape {field_shape}."
        )

    return ComponentBlockJacobiPreconditioner(
        mapping,
        tuple(field_shape),
        precision,
    )
