#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Calve off mechanically unanchored ice (rigid-body modes).

Floating ice has no basal drag, so its force balance closes only through
membrane stresses transmitted to grounded ice by full Q1 cells. A floating
body that lacks that path (an iceberg, or a shelf piece hinged on one
unsupported node) can translate or rotate freely: its velocity is not
determined and a Newton-CG solve returns arbitrary speeds on it, which
throttle the CFL step. Such ice is removed once at initialization and, with
an evolving calving front, after every thickness update. Grounded ice is
never touched: however isolated, it keeps its basal drag.
"""

import tensorflow as tf

from .fronts.utils import neighbour_bool_any
from .masks import compute_grounded_mask


def _reach(seed: tf.Tensor, domain: tf.Tensor) -> tf.Tensor:
    """Cells of the bool ``domain`` reachable from ``seed`` (flood fill).

    Connectivity is through edge neighbours because a node-only diagonal
    contact carries no membrane stress. In a winding component the graph
    distance can approach the number of cells, which bounds the loop.
    """
    domain = tf.cast(domain, tf.bool)

    def dilate(region):
        x = tf.cast(region, tf.float32)[tf.newaxis, :, :, tf.newaxis]
        grown = tf.maximum(
            tf.nn.max_pool2d(x, ksize=(3, 1), strides=1, padding="SAME"),
            tf.nn.max_pool2d(x, ksize=(1, 3), strides=1, padding="SAME"),
        )
        return tf.logical_and(grown[0, :, :, 0] > 0.5, domain)

    def step(region, grew):
        larger = dilate(region)
        return larger, tf.reduce_any(tf.logical_and(larger, tf.logical_not(region)))

    region, _ = tf.while_loop(
        lambda region, grew: grew,
        step,
        (tf.logical_and(tf.cast(seed, tf.bool), domain), tf.constant(True)),
        maximum_iterations=tf.size(domain),
    )
    return region


def _cells(nodes: tf.Tensor) -> tf.Tensor:
    """Q1 cells whose four corner nodes are all true."""
    return nodes[:-1, :-1] & nodes[:-1, 1:] & nodes[1:, :-1] & nodes[1:, 1:]


def _corners_any(cells: tf.Tensor) -> tf.Tensor:
    """Nodes that are a corner of at least one true cell."""
    c = tf.cast(cells, tf.int32)
    return (
        tf.pad(c, [[0, 1], [0, 1]])
        + tf.pad(c, [[0, 1], [1, 0]])
        + tf.pad(c, [[1, 0], [0, 1]])
        + tf.pad(c, [[1, 0], [1, 0]])
        > 0
    )


def _cells_with_grounded_corner(grounded: tf.Tensor) -> tf.Tensor:
    """Q1 cells with at least one grounded corner node."""
    return grounded[:-1, :-1] | grounded[:-1, 1:] | grounded[1:, :-1] | grounded[1:, 1:]


@tf.function(reduce_retracing=True)
def anchored_ice_mask(ice: tf.Tensor, grounded: tf.Tensor) -> tf.Tensor:
    """Bool (Ny, Nx) mask of the ice that is mechanically anchored.

    Floating ice is retained when it belongs to a full-Q1-cell component
    connected through cell edges to a cell with a grounded corner. Grounded
    ice is retained even when it is an isolated node.
    """
    ice = tf.cast(ice, tf.bool)
    grounded = tf.logical_and(tf.cast(grounded, tf.bool), ice)

    cells = _cells(ice)
    grounded_cells = tf.logical_and(cells, _cells_with_grounded_corner(grounded))
    anchored_cells = _reach(grounded_cells, cells)
    supported = _corners_any(anchored_cells)
    return tf.logical_and(ice, tf.logical_or(grounded, supported))


def remove_rigid_body_modes(state, rho_ratio: float) -> None:
    """Zero the unanchored ice of ``state`` (``thk``, and ``thk_true``/``Href`` if present).

    Only the dropped ice columns are touched, so the padding a sub-grid front
    adds around the remaining ice is left to the front scheme.
    """
    columns = state.thk_true if hasattr(state, "thk_true") else state.thk
    thk = tf.convert_to_tensor(columns)
    ice = thk > 0.0
    grounded = compute_grounded_mask(thk, state.topg, state.water_level, rho_ratio)
    keep_nodes = anchored_ice_mask(ice, grounded)
    dropped = tf.logical_and(ice, tf.logical_not(keep_nodes))
    keep = 1.0 - tf.cast(dropped, state.thk.dtype)

    if hasattr(state, "thk_true"):
        state.thk_true.assign(state.thk_true * keep)
        # The front schemes expose one layer of iceflow-only padding around
        # true columns. Keep padding and Href only beside retained ice.
        keep_extended = tf.logical_or(keep_nodes, neighbour_bool_any(keep_nodes))
        state.thk = state.thk * tf.cast(keep_extended, state.thk.dtype)
        if hasattr(state, "Href"):
            state.Href.assign(state.Href * tf.cast(keep_extended, state.Href.dtype))
    else:
        state.thk = state.thk * keep
