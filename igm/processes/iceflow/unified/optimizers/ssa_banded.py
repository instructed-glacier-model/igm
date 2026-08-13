"""Compact nonperiodic SSA stencil storage and graph-color extraction."""

from typing import Callable, Dict, Mapping, Sequence, Tuple

import tensorflow as tf

from .banded import OFFSETS, periodic_axes


SSABandKey = Tuple[str, str]
SSA_BAND_KEYS: Tuple[SSABandKey, ...] = (
    ("u", "u"),
    ("u", "v"),
    ("v", "u"),
    ("v", "v"),
)


def supports_compact_ssa(mapping) -> bool:
    """Return whether the mapping supports the compact SSA representation."""
    field_shape = getattr(mapping, "shape", None)
    return bool(
        getattr(mapping, "name", "") == "identity"
        and field_shape is not None
        and len(field_shape) == 4
        and field_shape[1] == 1
        and not any(periodic_axes(mapping))
    )


def _shift(field: tf.Tensor, dy: int, dx: int) -> tf.Tensor:
    if dy > 0:
        field = tf.pad(field[:, dy:], [[0, 0], [0, dy], [0, 0]])
    elif dy < 0:
        field = tf.pad(field[:, :dy], [[0, 0], [-dy, 0], [0, 0]])
    if dx > 0:
        field = tf.pad(field[:, :, dx:], [[0, 0], [0, 0], [0, dx]])
    elif dx < 0:
        field = tf.pad(field[:, :, :dx], [[0, 0], [0, 0], [-dx, 0]])
    return field


def build_ssa_selectors(ny: int, nx: int):
    color = tf.cast(
        tf.range(nx)[tf.newaxis] % 3 + 3 * (tf.range(ny)[:, tf.newaxis] % 3),
        tf.int8,
    )
    shifted = tf.cast(color + 1, tf.int32)[tf.newaxis]
    neighbour_colors = tuple(
        tf.cast(_shift(shifted, dy, dx)[0] - 1, tf.int8)
        for dy, dx in OFFSETS
    )
    return color, neighbour_colors


def allocate_ssa_bands(
    batch_size: int,
    ny: int,
    nx: int,
    dtype: tf.DType,
    name: str,
) -> Dict[SSABandKey, tf.Variable]:
    shape = (len(OFFSETS), batch_size, ny, nx)
    return {
        key: tf.Variable(
            tf.zeros(shape, dtype),
            trainable=False,
            name=f"{name}_{key[0]}{key[1]}",
        )
        for key in SSA_BAND_KEYS
    }


def extract_ssa_bands(
    apply_fn: Callable[[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]],
    batch_size: int,
    ny: int,
    nx: int,
    dtype: tf.DType,
    color: tf.Tensor,
    neighbour_colors: Sequence[tf.Tensor],
) -> Dict[SSABandKey, tf.Tensor]:
    """Recover four coupled 9-point stencils from 18 colored probes."""
    zero = tf.zeros((batch_size, ny, nx), dtype)
    responses = {key: [] for key in SSA_BAND_KEYS}

    for spatial_color in range(9):
        probe = tf.broadcast_to(
            tf.cast(tf.equal(color, spatial_color), dtype)[tf.newaxis],
            (batch_size, ny, nx),
        )
        h_u, h_v = apply_fn(probe, zero)
        responses[("u", "u")].append(h_u)
        responses[("v", "u")].append(h_v)
        h_u, h_v = apply_fn(zero, probe)
        responses[("u", "v")].append(h_u)
        responses[("v", "v")].append(h_v)

    response_stacks = {
        key: tf.stack(response, axis=0) for key, response in responses.items()
    }
    bands = {key: [] for key in SSA_BAND_KEYS}
    for offset_colors in neighbour_colors:
        selector = tf.one_hot(tf.cast(offset_colors, tf.int32), 9, dtype=dtype)
        for key in SSA_BAND_KEYS:
            bands[key].append(
                tf.einsum("yxc,cbyx->byx", selector, response_stacks[key])
            )
    return {key: tf.stack(value, axis=0) for key, value in bands.items()}


class SSABandedStencil:
    """Coupled 9-point SSA stencil stored as four scalar band fields."""

    def __init__(self, bands: Mapping[SSABandKey, tf.Tensor]):
        self.bands = bands
        self.ny = bands[("u", "u")].shape[2]
        self.nx = bands[("u", "u")].shape[3]

    def apply(
        self, u: tf.Tensor, v: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        u_padded = tf.pad(u, [[0, 0], [1, 1], [1, 1]])
        v_padded = tf.pad(v, [[0, 0], [1, 1], [1, 1]])
        h_u = tf.zeros_like(u)
        h_v = tf.zeros_like(v)
        for index, (dy, dx) in enumerate(OFFSETS):
            shifted_u = u_padded[
                :, 1 + dy : 1 + dy + self.ny, 1 + dx : 1 + dx + self.nx
            ]
            shifted_v = v_padded[
                :, 1 + dy : 1 + dy + self.ny, 1 + dx : 1 + dx + self.nx
            ]
            h_u = (
                h_u
                + self.bands[("u", "u")][index] * shifted_u
                + self.bands[("u", "v")][index] * shifted_v
            )
            h_v = (
                h_v
                + self.bands[("v", "u")][index] * shifted_u
                + self.bands[("v", "v")][index] * shifted_v
            )
        return h_u, h_v
