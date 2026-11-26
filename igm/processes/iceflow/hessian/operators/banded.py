#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Periodic-aware component-block stencils used by Newton-CG."""

from typing import Callable, List, Optional, Sequence, Tuple

import tensorflow as tf

OFFSETS: Tuple[Tuple[int, int], ...] = (
    (0, 0),
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
)
COMPONENT_BANDS_KEY = "component_blocks"
COMPONENT_CENTER_KEY = "component_center"


def axis_period(n: int, periodic: bool, duplicated_endpoint: bool) -> Optional[int]:
    """Effective period of a stencil axis, or None when it does not wrap."""
    if not periodic:
        return None
    return max(n - 1, 1) if duplicated_endpoint else max(n, 1)


def wrap_offset(
    offset: Tuple[int, int],
    period_y: Optional[int],
    period_x: Optional[int],
) -> Tuple[int, int]:
    """Reduce a stencil offset to the neighbour it reaches on the wrapped grid."""
    dy, dx = offset
    if period_y is not None:
        dy %= period_y
    if period_x is not None:
        dx %= period_x
    return dy, dx


def component_band_multipliers(
    ny: int,
    nx: int,
    *,
    periodic_y: bool = False,
    periodic_x: bool = False,
    duplicated_endpoints: bool = True,
) -> Tuple[float, ...]:
    """Per-offset weights that count aliased periodic couplings exactly once.

    On a periodic axis with fewer than three active cells, distinct 9-point
    offsets reach the same neighbour (with one active row, a cell is its own
    north and south neighbour), so the colour-based extraction assigns each
    aliased offset the full folded coupling and a plain sum over the bands
    would count it several times. Keeping only the first offset of every
    aliased group restores the exact operator; with three or more active
    cells per periodic axis every weight is one.
    """
    period_y = axis_period(ny, periodic_y, duplicated_endpoints)
    period_x = axis_period(nx, periodic_x, duplicated_endpoints)
    seen = set()
    multipliers = []
    for offset in OFFSETS:
        signature = wrap_offset(offset, period_y, period_x)
        multipliers.append(0.0 if signature in seen else 1.0)
        seen.add(signature)
    return tuple(multipliers)


def as_dtype(precision) -> tf.DType:
    if isinstance(precision, str):
        precision = {
            "single": "float32",
            "double": "float64",
            "half": "float16",
        }.get(precision, precision)
    return tf.dtypes.as_dtype(precision)


def periodic_axes(mapping) -> Tuple[bool, bool]:
    bc_names = {type(bc).__name__ for bc in mapping.apply_bcs}
    periodic_y = bool(bc_names.intersection({"PeriodicNS", "PeriodicNSGlobal"}))
    periodic_x = bool(bc_names.intersection({"PeriodicWE", "PeriodicWEGlobal"}))
    return periodic_y, periodic_x


def _cycle_square_colors(n: int) -> Tuple[List[int], int]:
    """Color a periodic radius-one axis so distance-one neighbours differ."""
    if n <= 0:
        return [], 0
    if n <= 2:
        return list(range(n)), n

    for n_colors in range(3, 6):
        states = {(0, 1): [0, 1]}
        for _ in range(2, n):
            next_states = {}
            for (previous_2, previous_1), path in states.items():
                for color in range(n_colors):
                    if color in (previous_2, previous_1):
                        continue
                    next_states.setdefault((previous_1, color), path + [color])
            states = next_states

        for path in states.values():
            if path[-1] != path[0] and path[-1] != path[1] and path[-2] != path[0]:
                return path, n_colors

    raise RuntimeError(f"Could not color periodic stencil of length {n}.")


def _axis_colors(
    n: int,
    periodic: bool,
    duplicated_endpoint: bool,
) -> Tuple[tf.Tensor, int]:
    if periodic:
        active_n = max(n - 1, 1) if duplicated_endpoint else n
        colors, n_colors = _cycle_square_colors(active_n)
        if duplicated_endpoint and n > 1:
            colors.append(-1)
        return tf.constant(colors, tf.int32), n_colors

    n_colors = min(3, n)
    return tf.range(n, dtype=tf.int32) % max(n_colors, 1), n_colors


def shift_local(
    field: tf.Tensor,
    dy: int,
    dx: int,
    *,
    periodic_y: bool = False,
    periodic_x: bool = False,
    fill_value=0,
) -> tf.Tensor:
    """Shift the last two axes, respecting duplicated periodic endpoints."""

    def shift_axis(value, offset, axis, periodic):
        if offset == 0:
            return value

        n = value.shape[axis]
        if n is None:
            raise ValueError("Banded operators require statically known grid sizes.")

        output_indices = tf.range(n, dtype=tf.int32)
        if periodic:
            active_n = max(n - 1, 1)
            valid = output_indices < active_n
            source_indices = tf.math.floormod(output_indices + offset, active_n)
        else:
            source_indices = output_indices + offset
            valid = tf.logical_and(source_indices >= 0, source_indices < n)
            source_indices = tf.clip_by_value(source_indices, 0, max(n - 1, 0))

        shifted = tf.gather(value, source_indices, axis=axis)
        mask_shape = [1] * value.shape.rank
        mask_shape[axis] = n
        valid = tf.reshape(valid, mask_shape)
        return tf.where(valid, shifted, tf.cast(fill_value, value.dtype))

    field = shift_axis(field, dy, -2, periodic_y)
    return shift_axis(field, dx, -1, periodic_x)


def build_component_selectors(
    ny: int,
    nx: int,
    *,
    periodic_y: bool = False,
    periodic_x: bool = False,
    duplicated_endpoints: bool = True,
):
    """Build spatial colors and neighbour-color lookup tables."""
    colors_y, n_colors_y = _axis_colors(ny, periodic_y, duplicated_endpoints)
    colors_x, n_colors_x = _axis_colors(nx, periodic_x, duplicated_endpoints)
    valid = tf.logical_and(
        colors_y[:, tf.newaxis] >= 0,
        colors_x[tf.newaxis, :] >= 0,
    )
    color = colors_x[tf.newaxis, :] + n_colors_x * colors_y[:, tf.newaxis]
    color = tf.where(valid, color, tf.constant(-1, tf.int32))

    def shift_color(value, dy, dx):
        if duplicated_endpoints:
            return shift_local(
                value,
                dy,
                dx,
                periodic_y=periodic_y,
                periodic_x=periodic_x,
                fill_value=-1,
            )
        if periodic_y and dy:
            value = tf.roll(value, shift=-dy, axis=-2)
            dy = 0
        if periodic_x and dx:
            value = tf.roll(value, shift=-dx, axis=-1)
            dx = 0
        return shift_local(value, dy, dx, fill_value=-1)

    neighbour_colors = [
        tf.cast(
            shift_color(
                color[tf.newaxis],
                dy,
                dx,
            )[0],
            tf.int8,
        )
        for dy, dx in OFFSETS
    ]
    return tf.cast(color, tf.int8), neighbour_colors, n_colors_y * n_colors_x


def extract_component_bands(
    apply_fn: Callable[[tf.Tensor], tf.Tensor],
    batch_size: int,
    n_components: int,
    ny: int,
    nx: int,
    dtype: tf.DType,
    color: tf.Tensor,
    neighbour_colors: List[tf.Tensor],
    n_colors: int,
    multipliers: Optional[Sequence[float]] = None,
) -> tf.Tensor:
    """Extract a dense-component 9-point stencil by graph coloring.

    ``multipliers`` (see ``component_band_multipliers``) weights each offset
    band so couplings aliased by a degenerate periodic axis count once.
    """
    if multipliers is None:
        multipliers = (1.0,) * len(OFFSETS)
    bands_by_input = []

    for input_component in range(n_components):
        responses = []
        component_mask = tf.one_hot(input_component, n_components, dtype=dtype)
        component_mask = component_mask[tf.newaxis, :, tf.newaxis, tf.newaxis]
        for spatial_color in range(n_colors):
            spatial_mask = tf.cast(tf.equal(color, spatial_color), dtype)
            probe = component_mask * spatial_mask[tf.newaxis, tf.newaxis]
            probe = tf.broadcast_to(
                probe,
                (batch_size, n_components, ny, nx),
            )
            responses.append(apply_fn(probe))

        response_stack = tf.stack(responses, axis=0)
        input_bands = []
        for offset_colors, multiplier in zip(neighbour_colors, multipliers):
            selector = tf.one_hot(
                tf.cast(offset_colors, tf.int32),
                n_colors,
                dtype=dtype,
            )
            band = tf.einsum("yxc,cboyx->boyx", selector, response_stack)
            if multiplier != 1.0:
                band *= tf.cast(multiplier, dtype)
            input_bands.append(band)
        bands_by_input.append(tf.stack(input_bands, axis=0))

    # (offset, batch, output-component, input-component, y, x)
    return tf.stack(bands_by_input, axis=3)


class ComponentBandedOperator:
    """9-point operator with dense coupling between velocity components."""

    def __init__(
        self,
        bands: tf.Tensor,
        *,
        periodic_y: bool = False,
        periodic_x: bool = False,
    ):
        self.bands = bands
        self.periodic_y = periodic_y
        self.periodic_x = periodic_x

    def apply(self, components: tf.Tensor) -> tf.Tensor:
        result = tf.zeros_like(components)
        for offset_index, (dy, dx) in enumerate(OFFSETS):
            shifted = shift_local(
                components,
                dy,
                dx,
                periodic_y=self.periodic_y,
                periodic_x=self.periodic_x,
            )
            result += tf.einsum(
                "boiyx,biyx->boyx",
                self.bands[offset_index],
                shifted,
            )
        return result
