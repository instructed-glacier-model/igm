"""Barotropic multigrid correction for compact MOLHO Hessians."""

from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np
import tensorflow as tf

from .banded import OFFSETS
from .molho_banded import (
    SymmetricBandedStencil,
    allocate_symmetric_bands,
    center_pairs,
    extract_symmetric_bands_batched,
)


def invert_spd_2x2(center: tf.Tensor, relative_floor: tf.Tensor) -> tf.Tensor:
    """Invert pointwise SPD blocks with a pivot-floored LDL factorization."""
    a = center[:, 0, 0]
    b = 0.5 * (center[:, 0, 1] + center[:, 1, 0])
    d = center[:, 1, 1]
    scale = tf.maximum(tf.abs(a), tf.abs(d))
    tiny = 1e-300 if center.dtype == tf.float64 else 1e-30
    floor = tf.maximum(
        tf.cast(relative_floor, center.dtype) * scale,
        tf.cast(tiny, center.dtype),
    )
    d0 = tf.maximum(a, floor)
    l10 = b / d0
    d1 = tf.maximum(d - l10 * l10 * d0, floor)
    inverse_11 = 1.0 / d1
    inverse_01 = -l10 * inverse_11
    inverse_00 = 1.0 / d0 + l10 * l10 * inverse_11
    return tf.stack(
        [
            tf.stack([inverse_00, inverse_01], axis=1),
            tf.stack([inverse_01, inverse_11], axis=1),
        ],
        axis=1,
    )


class AxisTransfer:
    """One-dimensional linear prolongation and its transpose restriction."""

    def __init__(self, fine_size: int, periodic: bool, coarsen: bool):
        self.fine_size = int(fine_size)
        self.periodic = bool(periodic)

        if not coarsen or fine_size <= 2:
            sample = np.arange(fine_size, dtype=np.int32)
        else:
            sample = np.arange(0, fine_size, 2, dtype=np.int32)
            if not periodic and sample[-1] != fine_size - 1:
                sample = np.append(sample, fine_size - 1)
        self.coarse_size = int(len(sample))

        left = np.empty(fine_size, dtype=np.int32)
        right = np.empty(fine_size, dtype=np.int32)
        right_weight = np.zeros(fine_size, dtype=np.float64)
        for fine_index in range(fine_size):
            exact = np.where(sample == fine_index)[0]
            if exact.size:
                left[fine_index] = right[fine_index] = exact[0]
                continue

            upper = int(np.searchsorted(sample, fine_index))
            if upper < len(sample):
                lower = upper - 1
                distance = float(sample[upper] - sample[lower])
                left[fine_index], right[fine_index] = lower, upper
                right_weight[fine_index] = (fine_index - sample[lower]) / distance
            else:
                left[fine_index] = len(sample) - 1
                right[fine_index] = 0
                distance = float(fine_size - sample[-1] + sample[0])
                right_weight[fine_index] = (fine_index - sample[-1]) / distance

        self.left = tf.constant(left, tf.int32)
        self.right = tf.constant(right, tf.int32)
        self.left_weight = tf.constant(1.0 - right_weight)
        self.right_weight = tf.constant(right_weight)

    def prolong(self, value: tf.Tensor, axis: int) -> tf.Tensor:
        axis %= value.shape.rank
        dtype = value.dtype
        shape = [1] * value.shape.rank
        shape[axis] = self.fine_size
        left_weight = tf.reshape(tf.cast(self.left_weight, dtype), shape)
        right_weight = tf.reshape(tf.cast(self.right_weight, dtype), shape)
        return (
            left_weight * tf.gather(value, self.left, axis=axis)
            + right_weight * tf.gather(value, self.right, axis=axis)
        )

    def restrict(self, value: tf.Tensor, axis: int) -> tf.Tensor:
        rank = value.shape.rank
        axis %= rank
        permutation = [axis] + [index for index in range(rank) if index != axis]
        inverse_permutation = np.argsort(permutation).tolist()
        value = tf.transpose(value, permutation)
        rank = value.shape.rank
        weight_shape = [self.fine_size] + [1] * (rank - 1)
        left_weight = tf.reshape(
            tf.cast(self.left_weight, value.dtype), weight_shape
        )
        right_weight = tf.reshape(
            tf.cast(self.right_weight, value.dtype), weight_shape
        )
        restricted = tf.math.unsorted_segment_sum(
            left_weight * value,
            self.left,
            self.coarse_size,
        )
        restricted += tf.math.unsorted_segment_sum(
            right_weight * value,
            self.right,
            self.coarse_size,
        )
        return tf.transpose(restricted, inverse_permutation)


class GridTransfer:
    """Tensor-product bilinear transfer between two grid levels."""

    def __init__(
        self,
        ny: int,
        nx: int,
        *,
        periodic_y: bool,
        periodic_x: bool,
        coarse_size: int,
    ):
        self.y = AxisTransfer(ny, periodic_y, ny > coarse_size)
        self.x = AxisTransfer(nx, periodic_x, nx > coarse_size)
        self.coarse_shape = (self.y.coarse_size, self.x.coarse_size)

    def prolong(self, coarse: tf.Tensor) -> tf.Tensor:
        return self.x.prolong(self.y.prolong(coarse, -2), -1)

    def restrict(self, fine: tf.Tensor) -> tf.Tensor:
        return self.y.restrict(self.x.restrict(fine, -1), -2)


@dataclass
class MultigridLevel:
    stencil: SymmetricBandedStencil
    apply: Callable[[tf.Tensor], tf.Tensor]
    center: tf.Variable
    edges: tf.Variable
    inverse_center: tf.Variable


class BarotropicMultigrid:
    """Symmetric geometric V-cycle for the two horizontal barotropic modes."""

    def __init__(
        self,
        batch_size: int,
        ny: int,
        nx: int,
        dtype: tf.DType,
        *,
        periodic_y: bool,
        periodic_x: bool,
        smoother_weight: float,
        smoother_steps: int,
        coarse_size: int,
        diagonal_floor: float = 1e-12,
    ):
        self.batch_size = int(batch_size)
        self.dtype = dtype
        self.periodic_y = bool(periodic_y)
        self.periodic_x = bool(periodic_x)
        self.smoother_weight = tf.constant(smoother_weight, dtype)
        self.smoother_steps = int(smoother_steps)
        self.coarse_size = int(coarse_size)
        self.diagonal_floor = tf.constant(diagonal_floor, dtype)

        self.levels: List[MultigridLevel] = []
        self.transfers: List[GridTransfer] = []
        level_ny, level_nx = int(ny), int(nx)
        level_index = 0
        while True:
            center, edges = allocate_symmetric_bands(
                self.batch_size,
                2,
                level_ny,
                level_nx,
                dtype,
                f"barotropic_level_{level_index}",
            )
            stencil = SymmetricBandedStencil(
                center,
                edges,
                periodic_y=self.periodic_y,
                periodic_x=self.periodic_x,
                duplicated_endpoints=False,
            )
            apply = tf.function(
                stencil.apply,
                autograph=False,
                reduce_retracing=True,
            )
            inverse_center = tf.Variable(
                tf.zeros(
                    (self.batch_size, 2, 2, level_ny, level_nx), dtype
                ),
                trainable=False,
                name=f"barotropic_inverse_{level_index}",
            )
            self.levels.append(
                MultigridLevel(stencil, apply, center, edges, inverse_center)
            )
            if max(level_ny, level_nx) <= self.coarse_size:
                break

            transfer = GridTransfer(
                level_ny,
                level_nx,
                periodic_y=self.periodic_y,
                periodic_x=self.periodic_x,
                coarse_size=self.coarse_size,
            )
            self.transfers.append(transfer)
            level_ny, level_nx = transfer.coarse_shape
            level_index += 1

        bottom_ny = int(self.levels[-1].center.shape[-2])
        bottom_nx = int(self.levels[-1].center.shape[-1])
        self._bottom_n = 2 * bottom_ny * bottom_nx
        self._dense_gather, self._dense_segments = self._build_dense_map(
            bottom_ny,
            bottom_nx,
        )
        self.bottom_inverse = tf.Variable(
            tf.eye(self._bottom_n, batch_shape=[self.batch_size], dtype=dtype),
            trainable=False,
            name="barotropic_bottom_inverse",
        )

    def _build_dense_map(
        self, ny: int, nx: int
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Map bottom-level stencil entries into a dense matrix."""
        offset_index = {offset: index for index, offset in enumerate(OFFSETS)}
        gather = []
        segments = []
        n = 2 * ny * nx
        for output_component in range(2):
            for y in range(ny):
                for x in range(nx):
                    row = (output_component * ny + y) * nx + x
                    for dy, dx in OFFSETS:
                        input_y, input_x = y + dy, x + dx
                        if self.periodic_y:
                            input_y %= ny
                        if self.periodic_x:
                            input_x %= nx
                        if not (0 <= input_y < ny and 0 <= input_x < nx):
                            continue
                        for input_component in range(2):
                            col = (input_component * ny + input_y) * nx + input_x
                            gather.append(
                                [
                                    offset_index[(dy, dx)],
                                    y,
                                    x,
                                    output_component,
                                    input_component,
                                ]
                            )
                            segments.append(row * n + col)
        return tf.constant(gather, tf.int32), tf.constant(segments, tf.int32)

    @tf.function(autograph=False, reduce_retracing=True)
    def _update_bottom_solver(self) -> None:
        bands = self.levels[-1].stencil.dense_bands()
        spatial_first = tf.transpose(bands, [0, 4, 5, 2, 3, 1])
        values = tf.gather_nd(spatial_first, self._dense_gather)
        dense = tf.math.unsorted_segment_sum(
            values,
            self._dense_segments,
            self._bottom_n * self._bottom_n,
        )
        dense = tf.reshape(
            tf.transpose(dense),
            [self.batch_size, self._bottom_n, self._bottom_n],
        )
        dense = 0.5 * (dense + tf.linalg.matrix_transpose(dense))
        scale = tf.reduce_max(tf.abs(tf.linalg.diag_part(dense)), axis=-1)
        tiny = 1e-300 if self.dtype == tf.float64 else 1e-30
        jitter = tf.maximum(
            self.diagonal_floor * scale,
            tf.cast(tiny, self.dtype),
        )
        identity = tf.eye(self._bottom_n, dtype=self.dtype)
        factor = tf.linalg.cholesky(
            dense + jitter[:, tf.newaxis, tf.newaxis] * identity
        )
        # Cache the inverse because every outer CG step reuses this tiny solve.
        self.bottom_inverse.assign(
            tf.linalg.cholesky_solve(
                factor,
                tf.eye(
                    self._bottom_n,
                    batch_shape=[self.batch_size],
                    dtype=self.dtype,
                ),
            )
        )

    @tf.function(autograph=False, reduce_retracing=True)
    def update(self, fine_center: tf.Tensor, fine_edges: tf.Tensor) -> None:
        self.levels[0].center.assign(fine_center)
        self.levels[0].edges.assign(fine_edges)

        for level_index, transfer in enumerate(self.transfers):
            fine = self.levels[level_index].stencil
            coarse_level = self.levels[level_index + 1]
            coarse_ny = int(coarse_level.center.shape[-2])
            coarse_nx = int(coarse_level.center.shape[-1])
            center, edges = extract_symmetric_bands_batched(
                lambda value: transfer.restrict(
                    fine.apply_many(transfer.prolong(value))
                ),
                self.batch_size,
                2,
                coarse_ny,
                coarse_nx,
                self.dtype,
                periodic_y=self.periodic_y,
                periodic_x=self.periodic_x,
                duplicated_endpoints=False,
            )
            coarse_level.center.assign(center)
            coarse_level.edges.assign(edges)

        for level in self.levels:
            level.inverse_center.assign(
                invert_spd_2x2(
                    level.stencil.center_matrix(damping=None),
                    self.diagonal_floor,
                )
            )
        self._update_bottom_solver()

    @staticmethod
    def _point_solve(level: MultigridLevel, residual: tf.Tensor) -> tf.Tensor:
        return tf.einsum(
            "boiyx,biyx->boyx",
            level.inverse_center,
            residual,
        )

    def _smooth(
        self,
        level_index: int,
        rhs: tf.Tensor,
        value: tf.Tensor,
    ) -> tf.Tensor:
        level = self.levels[level_index]
        for _ in range(self.smoother_steps):
            residual = rhs - level.apply(value)
            value += self.smoother_weight * self._point_solve(level, residual)
        return value

    def _v_cycle(self, level_index: int, rhs: tf.Tensor) -> tf.Tensor:
        if level_index == len(self.levels) - 1:
            flat = tf.reshape(rhs, [self.batch_size, self._bottom_n])
            solution = tf.einsum("bij,bj->bi", self.bottom_inverse, flat)
            return tf.reshape(solution, tf.shape(rhs))

        level = self.levels[level_index]
        value = self.smoother_weight * self._point_solve(level, rhs)
        for _ in range(1, self.smoother_steps):
            value = self._smooth(level_index, rhs, value)
        residual = rhs - level.apply(value)
        coarse_rhs = self.transfers[level_index].restrict(residual)
        coarse_error = self._v_cycle(level_index + 1, coarse_rhs)
        value += self.transfers[level_index].prolong(coarse_error)
        return self._smooth(level_index, rhs, value)

    @tf.function(autograph=False, reduce_retracing=True)
    def apply(self, residual: tf.Tensor) -> tf.Tensor:
        return self._v_cycle(0, residual)


def barotropic_mode(mapping, dtype: tf.DType) -> tf.Tensor:
    """Return the normalized vertical mode retained on coarse grids."""
    frozen_bed = any(type(bc).__name__ == "FrozenBed" for bc in mapping.apply_bcs)
    values = [0.0, 1.0] if frozen_bed else [2.0**-0.5, 2.0**-0.5]
    return tf.constant(values, dtype)


def project_molho_stencil(
    stencil: SymmetricBandedStencil,
    mode: tf.Tensor,
    damping: tf.Tensor,
    active_y: int,
    active_x: int,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Project a four-component MOLHO stencil onto horizontal velocity."""
    zero = tf.zeros_like(mode)
    prolongation = tf.stack(
        [
            tf.stack([mode[0], zero[0]]),
            tf.stack([mode[1], zero[1]]),
            tf.stack([zero[0], mode[0]]),
            tf.stack([zero[1], mode[1]]),
        ],
        axis=0,
    )
    full_center = stencil.center_matrix(damping)
    center = tf.einsum(
        "pa,bpqyx,qc->bacyx",
        prolongation,
        full_center,
        prolongation,
    )
    edges = tf.einsum(
        "pa,kbpqyx,qc->kbacyx",
        prolongation,
        stencil.edges,
        prolongation,
    )
    packed_center = tf.stack(
        [center[:, row, col] for row, col in center_pairs(2)], axis=0
    )
    return (
        packed_center[..., :active_y, :active_x],
        edges[..., :active_y, :active_x],
    )
