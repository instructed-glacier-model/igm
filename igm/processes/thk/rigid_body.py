#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
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

With a calving front, a node that has just filled can sit next to the
anchored ice before its neighbours fill and form full Q1 cells with it; such
edge-adjacent nodes are kept, and the reservoir ``Href`` of the partial cells
is kept only next to the remaining ice.
"""

import tensorflow as tf

from igm.common import State
from igm.utils.math.connectivity import reach
from igm.utils.math.neighbours import any_neighbour

from .masks import compute_grounded_mask


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
    # Edge connectivity: a node-only diagonal contact carries no membrane stress.
    anchored_cells = reach(grounded_cells, cells)
    supported = _corners_any(anchored_cells)
    return tf.logical_and(ice, tf.logical_or(grounded, supported))


def remove_rigid_body_modes(
    state: State, rho_ratio: float, front: bool = False
) -> tf.Tensor:
    """Zero the unanchored ice of ``state``; return the removed thickness (m).

    ``front`` marks a run with a calving front: nodes edge-adjacent to the
    anchored ice are then kept, and ``state.Href`` survives only next to the
    remaining ice. The returned field includes the removed ``Href``.
    """
    thk = tf.convert_to_tensor(state.thk)
    ice = thk > 0.0
    grounded = compute_grounded_mask(thk, state.topg, state.water_level, rho_ratio)
    keep = anchored_ice_mask(ice, grounded)
    if front:
        keep = keep | (ice & any_neighbour(keep))
    dropped = ice & ~keep
    removed = tf.where(dropped, thk, tf.zeros_like(thk))
    state.thk = tf.where(dropped, tf.zeros_like(thk), thk)
    if front:
        href = tf.convert_to_tensor(state.Href)
        orphan = ~any_neighbour(keep)
        removed += tf.where(orphan, href, tf.zeros_like(href))
        state.Href = tf.where(orphan, tf.zeros_like(href), href)
    return removed
