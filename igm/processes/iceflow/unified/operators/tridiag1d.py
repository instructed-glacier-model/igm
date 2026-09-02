#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Direct block-tridiagonal Hessian operator for y-invariant SSA problems.

Applies when the grid is ``Ny=2`` with a periodic north-south boundary
condition: ``PeriodicNS`` always overwrites row 1 with row 0
(``unified/bcs/periodic_ns.py``), so row 1 of ``theta`` never influences the
cost and the Hessian's row-1 direction is structurally zero. The surviving
unknowns are a single row of ``u(x), v(x)`` coupled only to their immediate
x-neighbours (q1/SSA stencil), i.e. an exact 2x2-block tridiagonal system.
"""

from typing import Callable, Dict, Tuple

import tensorflow as tf

from .banded import _axis_colors, periodic_axes, shift_local
from .energy_operator import _BandedADOperatorBase

def supports_tridiag1d(mapping) -> bool:
    """Return whether ``mapping`` is a y-invariant, x-only SSA problem."""
    field_shape = getattr(mapping, "shape", None)
    if (
        getattr(mapping, "name", "") != "identity"
        or field_shape is None
        or len(field_shape) != 4
    ):
        return False
    _, Nz, Ny, _ = tuple(field_shape)
    if Nz != 1 or Ny != 2:
        return False
    periodic_y, periodic_x = periodic_axes(mapping)
    return bool(periodic_y and not periodic_x)


def extract_tridiag1d_bands(
    apply_fn: Callable[[int, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]],
    batch_size: int,
    nx: int,
    dtype: tf.DType,
    colors_x: tf.Tensor,
    neighbour_colors: Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
    n_colors: int,
) -> tf.Tensor:
    """Recover a dense 2x2-block 3-point (west/center/east) x-stencil."""
    responses_uu, responses_vu = [], []
    responses_uv, responses_vv = [], []

    for spatial_color in range(n_colors):
        mask = tf.cast(tf.equal(colors_x, spatial_color), dtype)
        probe = tf.broadcast_to(mask, (batch_size, nx))

        h_u, h_v = apply_fn(0, probe)
        responses_uu.append(h_u)
        responses_vu.append(h_v)

        h_u, h_v = apply_fn(1, probe)
        responses_uv.append(h_u)
        responses_vv.append(h_v)

    return assemble_tridiag1d_bands(
        tf.stack(responses_uu, axis=0),
        tf.stack(responses_vu, axis=0),
        tf.stack(responses_uv, axis=0),
        tf.stack(responses_vv, axis=0),
        neighbour_colors,
        n_colors,
        dtype,
    )


def assemble_tridiag1d_bands(
    stack_uu: tf.Tensor,
    stack_vu: tf.Tensor,
    stack_uv: tf.Tensor,
    stack_vv: tf.Tensor,
    neighbour_colors: Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
    n_colors: int,
    dtype: tf.DType,
) -> tf.Tensor:
    """Select west/center/east blocks from colour-probe responses."""

    bands = []
    for offset_colors in neighbour_colors:
        selector = tf.one_hot(tf.cast(offset_colors, tf.int32), n_colors, dtype=dtype)
        b_uu = tf.einsum("xc,cbx->bx", selector, stack_uu)
        b_vu = tf.einsum("xc,cbx->bx", selector, stack_vu)
        b_uv = tf.einsum("xc,cbx->bx", selector, stack_uv)
        b_vv = tf.einsum("xc,cbx->bx", selector, stack_vv)
        row_u = tf.stack([b_uu, b_uv], axis=1)
        row_v = tf.stack([b_vu, b_vv], axis=1)
        bands.append(tf.stack([row_u, row_v], axis=1))

    return tf.stack(bands, axis=0)


class Tridiag1DADOperator(_BandedADOperatorBase):
    """Hessian frozen as an exact 2x2-block tridiagonal x-stencil."""

    name = "tridiag1d_ad"
    preconditioner_layout = "tridiag1d"

    def __init__(
        self,
        cost_fn,
        mapping,
        precision: str = "float32",
        verify_stencil: bool = False,
        probe_mode: str = "autodiff",
    ):
        if not supports_tridiag1d(mapping):
            raise ValueError(
                "Tridiag1DADOperator requires an identity SSA mapping with "
                "shape (B, 1, 2, Nx), periodic in y and non-periodic in x "
                f"(bcs must include periodic_ns but not periodic_we); got "
                f"mapping '{getattr(mapping, 'name', '?')}' with shape "
                f"{getattr(mapping, 'shape', None)}."
            )
        super().__init__(cost_fn, mapping, precision, verify_stencil, probe_mode)

        self.B, self.Nz, self.Ny, self.Nx = tuple(mapping.shape)
        self._colors_x, self._n_colors_x = _axis_colors(
            self.Nx, periodic=False, duplicated_endpoint=False
        )
        self._neighbour_colors = tuple(
            shift_local(
                self._colors_x[tf.newaxis, :], 0, dx, periodic_x=False, fill_value=-1
            )[0]
            for dx in (-1, 0, 1)
        )
        self._bands = tf.Variable(
            tf.zeros((3, self.B, 2, 2, self.Nx), self.precision),
            trainable=False,
            name="tridiag1d_bands",
        )
        self._prepared = False

    def prepare(self, inputs: tf.Tensor, damping: tf.Tensor) -> None:
        self._bands.assign(self._extract_bands(inputs))
        self._prepared = True
        self._verify_if_requested(inputs, damping)

    @tf.function(reduce_retracing=True)
    def _extract_bands(self, inputs: tf.Tensor) -> tf.Tensor:
        """Run all colour probes and assemble the stencil as one compiled graph.

        Fusing the probe loop trades a larger one-time trace (each nested
        ``cost_grad_at``/AD call gets re-embedded rather than reused as a
        cached standalone call) for a much cheaper steady-state call after
        that -- worth it once a run does more than a handful of Newton steps.
        """
        return extract_tridiag1d_bands(
            lambda comp, probe: self._component_apply(inputs, comp, probe),
            self.B,
            self.Nx,
            self.precision,
            self._colors_x,
            self._neighbour_colors,
            self._n_colors_x,
        )

    def hvp(
        self,
        inputs: tf.Tensor,
        v_flat: tf.Tensor,
        damping: tf.Tensor,
    ) -> tf.Tensor:
        del inputs
        if not self._prepared:
            raise RuntimeError("Tridiag1DADOperator.prepare() must be called before hvp().")
        u, v = self.split_row0(v_flat)
        h_u, h_v = self._apply_stencil(u, v)
        h_flat = self.join_row0(h_u, h_v)
        return h_flat + tf.cast(damping, self.precision) * v_flat

    def assemble_bands(self, inputs: tf.Tensor, damping: tf.Tensor) -> Dict[str, tf.Tensor]:
        del inputs
        if not self._prepared:
            raise RuntimeError(
                "Tridiag1DADOperator.prepare() must be called before assemble_bands()."
            )
        damping = tf.cast(damping, self.precision)
        identity = tf.eye(2, dtype=self.precision)[tf.newaxis, :, :, tf.newaxis]
        west, center, east = self._bands[0], self._bands[1], self._bands[2]
        return {"west": west, "center": center + damping * identity, "east": east}

    def synchronization_token(self) -> tf.Tensor:
        return self._bands

    def split_row0(self, flat: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        u, v = self.map.unflatten_theta(flat)
        return u[:, 0, 0, :], v[:, 0, 0, :]

    def join_row0(self, u_row0: tf.Tensor, v_row0: tf.Tensor) -> tf.Tensor:
        U = self._embed_row0(u_row0)
        V = self._embed_row0(v_row0)
        return self.map.flatten_theta([U, V])

    def _embed_row0(self, row0: tf.Tensor) -> tf.Tensor:
        expanded = row0[:, tf.newaxis, tf.newaxis, :]
        zero_row1 = tf.zeros_like(expanded)
        return tf.concat([expanded, zero_row1], axis=2)

    def _apply_stencil(self, u: tf.Tensor, v: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        u_west = tf.pad(u[:, :-1], [[0, 0], [1, 0]])
        u_east = tf.pad(u[:, 1:], [[0, 0], [0, 1]])
        v_west = tf.pad(v[:, :-1], [[0, 0], [1, 0]])
        v_east = tf.pad(v[:, 1:], [[0, 0], [0, 1]])

        def block_apply(block, uu, vv):
            return (
                block[:, 0, 0] * uu + block[:, 0, 1] * vv,
                block[:, 1, 0] * uu + block[:, 1, 1] * vv,
            )

        hu_w, hv_w = block_apply(self._bands[0], u_west, v_west)
        hu_c, hv_c = block_apply(self._bands[1], u, v)
        hu_e, hv_e = block_apply(self._bands[2], u_east, v_east)
        return hu_w + hu_c + hu_e, hv_w + hv_c + hv_e

    def _component_apply(
        self,
        inputs: tf.Tensor,
        comp: int,
        probe_row0: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        zero_row0 = tf.zeros_like(probe_row0)
        u_row0 = probe_row0 if comp == 0 else zero_row0
        v_row0 = probe_row0 if comp == 1 else zero_row0
        v_flat = self.join_row0(u_row0, v_row0)
        h_flat = self._probe_hvp(inputs, v_flat)
        return self.split_row0(h_flat)
