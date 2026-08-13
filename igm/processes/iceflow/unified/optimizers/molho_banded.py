"""Compact symmetric stencils for the two-layer MOLHO discretization."""

from typing import Callable, Dict, Tuple

import tensorflow as tf

from .banded import OFFSETS, build_component_selectors, periodic_axes, shift_local


MOLHO_STENCIL_KEY = "molho_symmetric_stencil"
EDGE_OFFSETS: Tuple[Tuple[int, int], ...] = (
    (1, 0),
    (0, 1),
    (1, 1),
    (1, -1),
)


def supports_compact_molho(mapping, basis_vertical: str = "molho") -> bool:
    """Return whether the mapping has the supported two-layer MOLHO layout."""
    shape = getattr(mapping, "shape", None)
    return bool(
        str(basis_vertical).lower() == "molho"
        and getattr(mapping, "name", "") == "identity"
        and shape is not None
        and len(shape) == 4
        and shape[1] == 2
    )


def _shift(
    field: tf.Tensor,
    dy: int,
    dx: int,
    *,
    periodic_y: bool,
    periodic_x: bool,
    duplicated_endpoints: bool,
) -> tf.Tensor:
    if duplicated_endpoints:
        return shift_local(
            field,
            dy,
            dx,
            periodic_y=periodic_y,
            periodic_x=periodic_x,
        )

    def shift_nonperiodic(value: tf.Tensor, offset: int, axis: int) -> tf.Tensor:
        if offset == 0:
            return value
        if abs(offset) != 1:
            raise ValueError("Compact stencils only support one-cell shifts.")
        if axis == -2:
            zero = tf.zeros_like(value[..., :1, :])
            parts = (
                [value[..., 1:, :], zero]
                if offset > 0
                else [zero, value[..., :-1, :]]
            )
        else:
            zero = tf.zeros_like(value[..., :1])
            parts = [value[..., 1:], zero] if offset > 0 else [zero, value[..., :-1]]
        return tf.concat(parts, axis=axis)

    if dy:
        if periodic_y:
            field = tf.roll(field, shift=-dy, axis=-2)
        else:
            field = shift_nonperiodic(field, dy, -2)
    if dx:
        if periodic_x:
            field = tf.roll(field, shift=-dx, axis=-1)
        else:
            field = shift_nonperiodic(field, dx, -1)
    return field


def center_pairs(n_components: int) -> Tuple[Tuple[int, int], ...]:
    return tuple(
        (row, col)
        for row in range(n_components)
        for col in range(row, n_components)
    )


class SymmetricBandedStencil:
    """Self-adjoint component stencil with one stored orientation per edge."""

    def __init__(
        self,
        center: tf.Tensor,
        edges: tf.Tensor,
        *,
        periodic_y: bool = False,
        periodic_x: bool = False,
        duplicated_endpoints: bool = True,
    ):
        self.center = center
        self.edges = edges
        self.n_components = int(edges.shape[2])
        self.pairs = center_pairs(self.n_components)
        self.periodic_y = bool(periodic_y)
        self.periodic_x = bool(periodic_x)
        self.duplicated_endpoints = bool(duplicated_endpoints)

    def center_matrix(self, damping=0.0) -> tf.Tensor:
        values = {pair: self.center[index] for index, pair in enumerate(self.pairs)}
        rows = []
        for row in range(self.n_components):
            rows.append(
                tf.stack(
                    [
                        values[min(row, col), max(row, col)]
                        for col in range(self.n_components)
                    ],
                    axis=1,
                )
            )
        matrix = tf.stack(rows, axis=1)
        if damping is not None:
            identity = tf.eye(self.n_components, dtype=matrix.dtype)
            identity = identity[tf.newaxis, :, :, tf.newaxis, tf.newaxis]
            matrix = matrix + tf.cast(damping, matrix.dtype) * identity
        return matrix

    def apply(self, components: tf.Tensor) -> tf.Tensor:
        center = self.center_matrix(damping=None)
        result = tf.einsum("boiyx,biyx->boyx", center, components)

        for index, (dy, dx) in enumerate(EDGE_OFFSETS):
            edge = self.edges[index]
            positive = _shift(
                components,
                dy,
                dx,
                periodic_y=self.periodic_y,
                periodic_x=self.periodic_x,
                duplicated_endpoints=self.duplicated_endpoints,
            )
            result += tf.einsum("boiyx,biyx->boyx", edge, positive)

            reciprocal = tf.transpose(edge, [0, 2, 1, 3, 4])
            reciprocal = _shift(
                reciprocal,
                -dy,
                -dx,
                periodic_y=self.periodic_y,
                periodic_x=self.periodic_x,
                duplicated_endpoints=self.duplicated_endpoints,
            )
            negative = _shift(
                components,
                -dy,
                -dx,
                periodic_y=self.periodic_y,
                periodic_x=self.periodic_x,
                duplicated_endpoints=self.duplicated_endpoints,
            )
            result += tf.einsum("boiyx,biyx->boyx", reciprocal, negative)
        return result

    def apply_many(self, components: tf.Tensor) -> tf.Tensor:
        """Apply to tensors shaped ``(..., batch, component, y, x)``."""
        center = self.center_matrix(damping=None)
        result = tf.einsum("boiyx,...biyx->...boyx", center, components)
        for index, (dy, dx) in enumerate(EDGE_OFFSETS):
            edge = self.edges[index]
            positive = _shift(
                components,
                dy,
                dx,
                periodic_y=self.periodic_y,
                periodic_x=self.periodic_x,
                duplicated_endpoints=self.duplicated_endpoints,
            )
            result += tf.einsum("boiyx,...biyx->...boyx", edge, positive)

            reciprocal = tf.transpose(edge, [0, 2, 1, 3, 4])
            reciprocal = _shift(
                reciprocal,
                -dy,
                -dx,
                periodic_y=self.periodic_y,
                periodic_x=self.periodic_x,
                duplicated_endpoints=self.duplicated_endpoints,
            )
            negative = _shift(
                components,
                -dy,
                -dx,
                periodic_y=self.periodic_y,
                periodic_x=self.periodic_x,
                duplicated_endpoints=self.duplicated_endpoints,
            )
            result += tf.einsum(
                "boiyx,...biyx->...boyx", reciprocal, negative
            )
        return result

    def dense_bands(self) -> tf.Tensor:
        bands = {(0, 0): self.center_matrix(damping=None)}
        for index, (dy, dx) in enumerate(EDGE_OFFSETS):
            edge = self.edges[index]
            bands[(dy, dx)] = edge
            reciprocal = tf.transpose(edge, [0, 2, 1, 3, 4])
            bands[(-dy, -dx)] = _shift(
                reciprocal,
                -dy,
                -dx,
                periodic_y=self.periodic_y,
                periodic_x=self.periodic_x,
                duplicated_endpoints=self.duplicated_endpoints,
            )
        return tf.stack([bands[offset] for offset in OFFSETS], axis=0)


def allocate_symmetric_bands(
    batch_size: int,
    n_components: int,
    ny: int,
    nx: int,
    dtype: tf.DType,
    name: str,
) -> Tuple[tf.Variable, tf.Variable]:
    center = tf.Variable(
        tf.zeros(
            (len(center_pairs(n_components)), batch_size, ny, nx), dtype
        ),
        trainable=False,
        name=f"{name}_center",
    )
    edges = tf.Variable(
        tf.zeros(
            (
                len(EDGE_OFFSETS),
                batch_size,
                n_components,
                n_components,
                ny,
                nx,
            ),
            dtype,
        ),
        trainable=False,
        name=f"{name}_edges",
    )
    return center, edges


def _select_response(
    response_stack: tf.Tensor,
    offset_colors: tf.Tensor,
    n_colors: int,
) -> tf.Tensor:
    selector = tf.one_hot(
        tf.cast(offset_colors, tf.int32),
        n_colors,
        dtype=response_stack.dtype,
    )
    return tf.einsum("yxc,cboyx->boyx", selector, response_stack)


def extract_symmetric_bands(
    apply_fn: Callable[[tf.Tensor], tf.Tensor],
    batch_size: int,
    n_components: int,
    ny: int,
    nx: int,
    dtype: tf.DType,
    *,
    periodic_y: bool = False,
    periodic_x: bool = False,
    duplicated_endpoints: bool = True,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Extract and average reciprocal entries without retaining full bands."""
    color, neighbour_colors, n_colors = build_component_selectors(
        ny,
        nx,
        periodic_y=periodic_y,
        periodic_x=periodic_x,
        duplicated_endpoints=duplicated_endpoints,
    )
    offset_colors: Dict[Tuple[int, int], tf.Tensor] = dict(
        zip(OFFSETS, neighbour_colors)
    )
    pairs = center_pairs(n_components)
    center_parts = [tf.zeros((batch_size, ny, nx), dtype) for _ in pairs]
    edge_parts = [
        tf.zeros((batch_size, n_components, n_components, ny, nx), dtype)
        for _ in EDGE_OFFSETS
    ]

    for input_component in range(n_components):
        component_mask = tf.one_hot(input_component, n_components, dtype=dtype)
        component_mask = component_mask[tf.newaxis, :, tf.newaxis, tf.newaxis]
        responses = []
        for spatial_color in range(n_colors):
            spatial_mask = tf.cast(tf.equal(color, spatial_color), dtype)
            probe = component_mask * spatial_mask[tf.newaxis, tf.newaxis]
            probe = tf.broadcast_to(
                probe,
                (batch_size, n_components, ny, nx),
            )
            responses.append(apply_fn(probe))
        response_stack = tf.stack(responses, axis=0)

        center_response = _select_response(
            response_stack,
            offset_colors[(0, 0)],
            n_colors,
        )
        for pair_index, (row, col) in enumerate(pairs):
            if row == col == input_component:
                center_parts[pair_index] += center_response[:, row]
            else:
                if col == input_component:
                    center_parts[pair_index] += 0.5 * center_response[:, row]
                if row == input_component:
                    center_parts[pair_index] += 0.5 * center_response[:, col]

        input_mask = tf.one_hot(input_component, n_components, dtype=dtype)
        for edge_index, (dy, dx) in enumerate(EDGE_OFFSETS):
            direct = _select_response(
                response_stack,
                offset_colors[(dy, dx)],
                n_colors,
            )
            direct = tf.einsum("boyx,i->boiyx", direct, input_mask)

            reciprocal = _select_response(
                response_stack,
                offset_colors[(-dy, -dx)],
                n_colors,
            )
            reciprocal = _shift(
                reciprocal,
                dy,
                dx,
                periodic_y=periodic_y,
                periodic_x=periodic_x,
                duplicated_endpoints=duplicated_endpoints,
            )
            reciprocal = tf.einsum(
                "o,biyx->boiyx", input_mask, reciprocal
            )
            edge_parts[edge_index] += 0.5 * (direct + reciprocal)

    return tf.stack(center_parts, axis=0), tf.stack(edge_parts, axis=0)


def compact_symmetric_bands(
    dense_bands: tf.Tensor,
    *,
    periodic_y: bool = False,
    periodic_x: bool = False,
    duplicated_endpoints: bool = True,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Pack dense 9-point bands using Hessian reciprocity."""
    center_matrix = 0.5 * (
        dense_bands[0] + tf.transpose(dense_bands[0], [0, 2, 1, 3, 4])
    )
    packed_center = tf.stack(
        [
            center_matrix[:, row, col]
            for row, col in center_pairs(int(center_matrix.shape[1]))
        ],
        axis=0,
    )
    offset_index = {offset: index for index, offset in enumerate(OFFSETS)}
    edges = []
    for dy, dx in EDGE_OFFSETS:
        direct = dense_bands[offset_index[(dy, dx)]]
        reciprocal = dense_bands[offset_index[(-dy, -dx)]]
        reciprocal = tf.transpose(reciprocal, [0, 2, 1, 3, 4])
        reciprocal = _shift(
            reciprocal,
            dy,
            dx,
            periodic_y=periodic_y,
            periodic_x=periodic_x,
            duplicated_endpoints=duplicated_endpoints,
        )
        edges.append(0.5 * (direct + reciprocal))
    return packed_center, tf.stack(edges, axis=0)


def extract_symmetric_bands_batched(
    apply_many_fn: Callable[[tf.Tensor], tf.Tensor],
    batch_size: int,
    n_components: int,
    ny: int,
    nx: int,
    dtype: tf.DType,
    *,
    periodic_y: bool = False,
    periodic_x: bool = False,
    duplicated_endpoints: bool = True,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Extract all colored component probes through one batched apply."""
    color, neighbour_colors, n_colors = build_component_selectors(
        ny,
        nx,
        periodic_y=periodic_y,
        periodic_x=periodic_x,
        duplicated_endpoints=duplicated_endpoints,
    )
    components = tf.eye(n_components, dtype=dtype)
    spatial = tf.cast(
        tf.equal(
            color[tf.newaxis],
            tf.cast(
                tf.range(n_colors)[:, tf.newaxis, tf.newaxis], color.dtype
            ),
        ),
        dtype,
    )
    probes = tf.einsum("ic,kyx->ikcyx", components, spatial)
    probes = tf.broadcast_to(
        probes[:, :, tf.newaxis],
        (n_components, n_colors, batch_size, n_components, ny, nx),
    )
    responses = apply_many_fn(probes)

    dense_bands = []
    for offset_colors in neighbour_colors:
        selector = tf.one_hot(
            tf.cast(offset_colors, tf.int32),
            n_colors,
            dtype=dtype,
        )
        dense_bands.append(
            tf.einsum("yxc,icboyx->boiyx", selector, responses)
        )
    return compact_symmetric_bands(
        tf.stack(dense_bands, axis=0),
        periodic_y=periodic_y,
        periodic_x=periodic_x,
        duplicated_endpoints=duplicated_endpoints,
    )


def allocate_molho_bands(
    mapping,
    dtype: tf.DType,
    name: str = "molho_band",
) -> Tuple[tf.Variable, tf.Variable]:
    if not supports_compact_molho(mapping):
        raise ValueError("Compact MOLHO bands require identity mapping with Nz=2.")
    batch_size, _, ny, nx = tuple(mapping.shape)
    return allocate_symmetric_bands(batch_size, 4, ny, nx, dtype, name)


def build_molho_stencil(
    mapping,
    center: tf.Tensor,
    edges: tf.Tensor,
) -> SymmetricBandedStencil:
    periodic_y, periodic_x = periodic_axes(mapping)
    return SymmetricBandedStencil(
        center,
        edges,
        periodic_y=periodic_y,
        periodic_x=periodic_x,
        duplicated_endpoints=True,
    )
