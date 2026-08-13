#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Preconditioners supported by the Newton-CG ice-flow solver."""

from typing import Callable, Optional, Tuple

import tensorflow as tf

from .banded import (
    COMPONENT_BANDS_KEY,
    COMPONENT_CENTER_KEY,
    as_dtype,
    build_component_selectors,
    extract_component_bands,
    periodic_axes,
)
from .barotropic_multigrid import (
    BarotropicMultigrid,
    barotropic_mode,
    project_molho_stencil,
)
from .molho_banded import MOLHO_STENCIL_KEY
from .ssa_banded import SSA_BAND_KEYS, supports_compact_ssa


def invert_2x2(
    a: tf.Tensor,
    b: tf.Tensor,
    c: tf.Tensor,
    d: tf.Tensor,
    relative_floor: tf.Tensor,
) -> tf.Tensor:
    """Invert pointwise 2x2 blocks with a determinant floor."""
    determinant = a * d - b * c
    tiny = 1e-300 if a.dtype == tf.float64 else 1e-30
    floor = tf.maximum(
        tf.cast(relative_floor, a.dtype) * tf.reduce_max(tf.abs(determinant)),
        tf.cast(tiny, a.dtype),
    )
    sign = tf.where(
        determinant < 0.0,
        -tf.ones_like(determinant),
        tf.ones_like(determinant),
    )
    determinant = sign * tf.maximum(tf.abs(determinant), floor)
    adjugate = tf.stack(
        [
            tf.stack([d, -b], axis=1),
            tf.stack([-c, a], axis=1),
        ],
        axis=1,
    )
    return adjugate * tf.math.reciprocal(
        determinant[:, tf.newaxis, tf.newaxis]
    )


def invert_spd_4x4(
    center: tf.Tensor,
    relative_floor: tf.Tensor,
) -> tf.Tensor:
    """Invert pointwise symmetric 4x4 blocks with modified LDL factors."""
    symmetric = 0.5 * (center + tf.transpose(center, [0, 2, 1, 3, 4]))

    def entry(row: int, col: int) -> tf.Tensor:
        return symmetric[:, row, col]

    diagonal_scale = tf.reduce_max(
        tf.abs(
            tf.stack([entry(index, index) for index in range(4)], axis=0)
        ),
        axis=0,
    )
    tiny = 1e-300 if center.dtype == tf.float64 else 1e-30
    floor = tf.maximum(
        tf.cast(relative_floor, center.dtype) * diagonal_scale,
        tf.cast(tiny, center.dtype),
    )

    d0 = tf.maximum(entry(0, 0), floor)
    l10 = entry(1, 0) / d0
    l20 = entry(2, 0) / d0
    l30 = entry(3, 0) / d0

    d1 = tf.maximum(entry(1, 1) - l10 * l10 * d0, floor)
    l21 = (entry(2, 1) - l20 * l10 * d0) / d1
    l31 = (entry(3, 1) - l30 * l10 * d0) / d1

    d2 = tf.maximum(
        entry(2, 2) - l20 * l20 * d0 - l21 * l21 * d1,
        floor,
    )
    l32 = (entry(3, 2) - l30 * l20 * d0 - l31 * l21 * d1) / d2
    d3 = tf.maximum(
        entry(3, 3)
        - l30 * l30 * d0
        - l31 * l31 * d1
        - l32 * l32 * d2,
        floor,
    )

    columns = []
    for column in range(4):
        rhs = [
            tf.ones_like(d0) if row == column else tf.zeros_like(d0)
            for row in range(4)
        ]
        y0 = rhs[0]
        y1 = rhs[1] - l10 * y0
        y2 = rhs[2] - l20 * y0 - l21 * y1
        y3 = rhs[3] - l30 * y0 - l31 * y1 - l32 * y2

        z0, z1, z2, z3 = y0 / d0, y1 / d1, y2 / d2, y3 / d3
        x3 = z3
        x2 = z2 - l32 * x3
        x1 = z1 - l21 * x2 - l31 * x3
        x0 = z0 - l10 * x1 - l20 * x2 - l30 * x3
        columns.append(tf.stack([x0, x1, x2, x3], axis=1))

    inverse = tf.stack(columns, axis=2)
    return 0.5 * (inverse + tf.transpose(inverse, [0, 2, 1, 3, 4]))


class Preconditioner:
    name = "identity"
    needs_operator = False

    def set_operator(self, operator) -> None:
        del operator

    def update(self, inputs: tf.Tensor, damping: tf.Tensor) -> None:
        del inputs, damping

    def apply(self, residual_flat: tf.Tensor) -> tf.Tensor:
        return residual_flat

    def synchronization_token(self) -> Optional[tf.Tensor]:
        return None


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
        self._selectors = None
        self._hvp: Optional[Callable[..., tf.Tensor]] = None
        self._assemble = None

        self.inverse_center = None

    def set_operator(self, operator) -> None:
        self._hvp = operator.hvp if hasattr(operator, "hvp") else operator
        self._assemble = getattr(operator, "assemble_bands", None)

    def update(self, inputs: tf.Tensor, damping: tf.Tensor) -> None:
        if self._hvp is None:
            raise RuntimeError("block_jacobi operator was not registered.")

        bands = self._assemble(inputs, damping) if self._assemble is not None else None
        if isinstance(bands, dict) and COMPONENT_CENTER_KEY in bands:
            center = bands[COMPONENT_CENTER_KEY]
        elif isinstance(bands, dict) and COMPONENT_BANDS_KEY in bands:
            component_bands = bands[COMPONENT_BANDS_KEY]
            center = component_bands[0]
        else:
            if self._selectors is None:
                self._selectors = build_component_selectors(
                    self.Ny,
                    self.Nx,
                    periodic_y=self.periodic_y,
                    periodic_x=self.periodic_x,
                )
            colors, neighbour_colors, n_colors = self._selectors
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
                colors,
                neighbour_colors,
                n_colors,
            )
            identity = tf.eye(self.n_components, dtype=self.dtype)
            identity = identity[tf.newaxis, :, :, tf.newaxis, tf.newaxis]
            center = component_bands[0] + tf.cast(damping, self.dtype) * identity
        if self.n_components == 2:
            inverse_center = invert_2x2(
                center[:, 0, 0],
                center[:, 0, 1],
                center[:, 1, 0],
                center[:, 1, 1],
                self.eigenvalue_floor,
            )
        else:
            inverse_center = self._invert_component_center(center)
        if self.inverse_center is None:
            self.inverse_center = tf.Variable(
                tf.zeros_like(inverse_center),
                trainable=False,
                name="component_block_jacobi_inverse",
            )
        self.inverse_center.assign(inverse_center)

    def _eigenvalue_floor(self, scale: tf.Tensor) -> tf.Tensor:
        tiny_value = 1e-300 if self.dtype == tf.float64 else 1e-30
        return tf.maximum(
            self.eigenvalue_floor * scale,
            tf.cast(tiny_value, self.dtype),
        )

    def _invert_component_center(self, center: tf.Tensor) -> tf.Tensor:
        if self.n_components == 4:
            return invert_spd_4x4(center, self.eigenvalue_floor)

        matrix = tf.transpose(center, [0, 3, 4, 1, 2])
        matrix = 0.5 * (matrix + tf.linalg.matrix_transpose(matrix))
        diagonal_scale = tf.reduce_max(
            tf.abs(tf.linalg.diag_part(matrix)), axis=-1
        )
        floor = self._eigenvalue_floor(diagonal_scale)
        identity = tf.eye(self.n_components, dtype=self.dtype)
        inverse = tf.linalg.inv(
            matrix + floor[..., tf.newaxis, tf.newaxis] * identity
        )
        return tf.transpose(inverse, [0, 3, 4, 1, 2])

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

    def synchronization_token(self) -> Optional[tf.Tensor]:
        return self.inverse_center


class BarotropicMultigridPreconditioner(ComponentBlockJacobiPreconditioner):
    """Full MOLHO point smoothing with a barotropic multigrid correction."""

    name = "barotropic_multigrid"

    def __init__(
        self,
        mapping,
        field_shape,
        precision="float32",
        eigenvalue_floor=1e-12,
        *,
        smoother_weight=2.0 / 3.0,
        smoother_steps=1,
        coarse_size=8,
    ):
        super().__init__(
            mapping,
            field_shape,
            precision,
            eigenvalue_floor,
        )
        if self.Nz != 2:
            raise ValueError("barotropic_multigrid requires MOLHO with Nz=2.")
        if not 0.0 < float(smoother_weight) <= 1.0:
            raise ValueError("multigrid.smoother_weight must be in (0, 1].")
        if int(smoother_steps) < 1:
            raise ValueError("multigrid.smoother_steps must be at least one.")
        if int(coarse_size) < 2:
            raise ValueError("multigrid.coarse_size must be at least two.")

        point_weight = min(float(smoother_weight), 0.5) if (
            self.periodic_y or self.periodic_x
        ) else float(smoother_weight)
        self.smoother_weight = tf.constant(point_weight, self.dtype)
        self.smoother_steps = int(smoother_steps)
        self.mode = barotropic_mode(mapping, self.dtype)
        self.active_y = self.Ny - 1 if self.periodic_y else self.Ny
        self.active_x = self.Nx - 1 if self.periodic_x else self.Nx
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
            name="molho_block_inverse",
        )
        self._damping = tf.Variable(
            tf.cast(0.0, self.dtype),
            trainable=False,
            name="molho_multigrid_damping",
        )
        self._stencil = None
        self.multigrid = BarotropicMultigrid(
            self.B,
            self.active_y,
            self.active_x,
            self.dtype,
            periodic_y=self.periodic_y,
            periodic_x=self.periodic_x,
            smoother_weight=point_weight,
            smoother_steps=self.smoother_steps,
            coarse_size=int(coarse_size),
            diagonal_floor=eigenvalue_floor,
        )

    def update(self, inputs: tf.Tensor, damping: tf.Tensor) -> None:
        if self._assemble is None:
            raise RuntimeError(
                "barotropic_multigrid requires a compact MOLHO operator."
            )
        assembled = self._assemble(inputs, damping)
        if not isinstance(assembled, dict) or MOLHO_STENCIL_KEY not in assembled:
            raise RuntimeError(
                "barotropic_multigrid requires compact symmetric MOLHO bands."
            )

        self._stencil = assembled[MOLHO_STENCIL_KEY]
        self._damping.assign(tf.cast(damping, self.dtype))
        center = assembled[COMPONENT_CENTER_KEY]
        self.inverse_center.assign(invert_spd_4x4(center, self.eigenvalue_floor))
        coarse_center, coarse_edges = project_molho_stencil(
            self._stencil,
            self.mode,
            self._damping,
            self.active_y,
            self.active_x,
        )
        self.multigrid.update(coarse_center, coarse_edges)

    def _operator_apply(self, components: tf.Tensor) -> tf.Tensor:
        return self._stencil.apply(components) + self._damping * components

    def _point_solve(self, residual: tf.Tensor) -> tf.Tensor:
        return tf.einsum(
            "boiyx,biyx->boyx",
            self.inverse_center,
            residual,
        )

    def _project(self, components: tf.Tensor) -> tf.Tensor:
        u = tf.einsum("z,bzyx->byx", self.mode, components[:, :2])
        v = tf.einsum("z,bzyx->byx", self.mode, components[:, 2:])
        return tf.stack([u, v], axis=1)[
            ..., : self.active_y, : self.active_x
        ]

    def _prolong_mode(self, barotropic: tf.Tensor) -> tf.Tensor:
        u = self.mode[tf.newaxis, :, tf.newaxis, tf.newaxis] * barotropic[:, 0:1]
        v = self.mode[tf.newaxis, :, tf.newaxis, tf.newaxis] * barotropic[:, 1:2]
        components = tf.concat([u, v], axis=1)
        return tf.pad(
            components,
            [
                [0, 0],
                [0, 0],
                [0, self.Ny - self.active_y],
                [0, self.Nx - self.active_x],
            ],
        )

    @tf.function(reduce_retracing=True)
    def apply(self, residual_flat: tf.Tensor) -> tf.Tensor:
        residual = tf.cast(self._split(residual_flat), self.dtype)
        value = self.smoother_weight * self._point_solve(residual)
        for _ in range(1, self.smoother_steps):
            point_residual = residual - self._operator_apply(value)
            value += self.smoother_weight * self._point_solve(point_residual)

        coarse_residual = self._project(
            residual - self._operator_apply(value)
        )
        value += self._prolong_mode(self.multigrid.apply(coarse_residual))

        for _ in range(self.smoother_steps):
            point_residual = residual - self._operator_apply(value)
            value += self.smoother_weight * self._point_solve(point_residual)
        return tf.cast(self._join(value), residual_flat.dtype)


class SSABlockJacobiPreconditioner(Preconditioner):
    """Point-block Jacobi using the center of the compact SSA stencil."""

    name = "block_jacobi"
    needs_operator = True

    def __init__(self, mapping, precision: str = "float32"):
        if not supports_compact_ssa(mapping):
            raise ValueError(
                "The compact SSA preconditioner requires a nonperiodic "
                "identity mapping with shape (B, 1, Ny, Nx)."
            )
        self.map = mapping
        self.dtype = as_dtype(precision)
        self.batch_size, _, self.ny, self.nx = tuple(mapping.shape)
        self.determinant_floor = tf.constant(1e-12, self.dtype)
        self.inverse_center = tf.Variable(
            tf.zeros((self.batch_size, 2, 2, self.ny, self.nx), self.dtype),
            trainable=False,
            name="ssa_block_inverse",
        )
        self._assemble = None

    def set_operator(self, operator) -> None:
        self._assemble = getattr(operator, "assemble_bands", None)

    @tf.function(reduce_retracing=True)
    def update(self, inputs: tf.Tensor, damping: tf.Tensor) -> None:
        if self._assemble is None:
            raise RuntimeError(
                "The compact SSA preconditioner requires compact operator bands."
            )
        bands = self._assemble(inputs, damping)
        if not isinstance(bands, dict) or not all(
            key in bands for key in SSA_BAND_KEYS
        ):
            raise RuntimeError(
                "The compact SSA operator did not provide all four band fields."
            )
        inverse = invert_2x2(
            bands[("u", "u")][0],
            bands[("u", "v")][0],
            bands[("v", "u")][0],
            bands[("v", "v")][0],
            self.determinant_floor,
        )
        self.inverse_center.assign(inverse)

    def _split(self, flat: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        u, v = self.map.unflatten_theta(flat)
        return u[:, 0], v[:, 0]

    def _join(self, u: tf.Tensor, v: tf.Tensor) -> tf.Tensor:
        return self.map.flatten_theta([u[:, tf.newaxis], v[:, tf.newaxis]])

    @tf.function(reduce_retracing=True)
    def apply(self, residual_flat: tf.Tensor) -> tf.Tensor:
        residual_u, residual_v = self._split(residual_flat)
        solution_u = (
            self.inverse_center[:, 0, 0] * residual_u
            + self.inverse_center[:, 0, 1] * residual_v
        )
        solution_v = (
            self.inverse_center[:, 1, 0] * residual_u
            + self.inverse_center[:, 1, 1] * residual_v
        )
        return self._join(solution_u, solution_v)

    def synchronization_token(self) -> tf.Tensor:
        return self.inverse_center


def build_preconditioner(
    kind,
    mapping,
    precision="float32",
    layout="component",
    options=None,
):
    kind = (kind or "none").lower()
    if kind in ("none", "identity"):
        return IdentityPreconditioner()

    if kind not in ("block_jacobi", "barotropic_multigrid"):
        raise ValueError(
            f"Unknown preconditioner kind: <{kind}>. Use 'none', "
            "'block_jacobi', or 'barotropic_multigrid'."
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

    if kind == "barotropic_multigrid":
        if layout != "compact_molho":
            raise ValueError(
                "barotropic_multigrid requires hvp_mode='banded', an identity "
                "mapping, basis_vertical='molho', and Nz=2."
            )
        return BarotropicMultigridPreconditioner(
            mapping,
            tuple(field_shape),
            precision,
            **dict(options or {}),
        )

    if layout == "compact_ssa":
        return SSABlockJacobiPreconditioner(mapping, precision)
    if layout not in ("component", "compact_molho"):
        raise ValueError(f"Unknown preconditioner layout: <{layout}>.")

    return ComponentBlockJacobiPreconditioner(
        mapping,
        tuple(field_shape),
        precision,
    )
