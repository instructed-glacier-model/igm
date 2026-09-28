#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Ice shelves: their labels and the distances that define the PICO boxes."""

from typing import Optional, Tuple

import tensorflow as tf

from igm.utils.math.connectivity import graph_distance, label_components

from ..utils import DIAGONAL, EDGE, any_neighbour
from .geometry import Geometry


@tf.function(autograph=False, jit_compile=True)
def _ice_rises(
    grounded_ice: tf.Tensor, cell_area: tf.Tensor, max_area: float
) -> tf.Tensor:
    labels = label_components(grounded_ice)
    count = tf.math.unsorted_segment_sum(
        tf.cast(grounded_ice, cell_area.dtype), labels, tf.size(labels) + 1
    )
    return grounded_ice & (tf.gather(count, labels) * cell_area < max_area)


def ice_rises(geom: Geometry, dx: tf.Tensor, max_area: float) -> tf.Tensor:
    """Grounded-ice areas smaller than ``max_area`` (m2), e.g. pinning points."""
    dx = tf.cast(dx, geom.draft.dtype)
    return _ice_rises(geom.grounded & geom.ice, dx * dx, float(max_area))


def shelf_labels(geom: Geometry, rises: Optional[tf.Tensor] = None) -> tf.Tensor:
    """Ice-shelf labels: edge-connected components of the shelf (0 outside).

    With ``rises``, shelves joined by an ice rise share one label, as in the
    PICO implementation of PISM. Labels are distinct but not consecutive, in
    ``[1, ny*nx]``; use ``ny*nx + 1`` segments to reduce over them.
    """
    if rises is None:
        return label_components(geom.shelf)
    labels = label_components(geom.shelf | rises)
    return tf.where(geom.shelf, labels, 0)


@tf.function(autograph=False, jit_compile=True)
def _shelf_distances(
    shelf: tf.Tensor,
    grounded_ice: tf.Tensor,
    open_ocean: tf.Tensor,
    open_edge: tf.Tensor,
    passable: tf.Tensor,
    labels: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    gl_seed = shelf & any_neighbour(grounded_ice, EDGE + DIAGONAL)
    front_seed = shelf & any_neighbour(open_ocean, EDGE)
    has_front = tf.math.unsorted_segment_max(
        tf.cast(front_seed, tf.int32), labels, tf.size(labels) + 1
    )
    front_seed = tf.where(
        tf.gather(has_front, labels) > 0, front_seed, shelf & open_edge
    )
    domain = shelf | passable
    return graph_distance(gl_seed, domain), graph_distance(front_seed, domain)


def shelf_distances(
    geom: Geometry, labels: tf.Tensor, rises: Optional[tf.Tensor] = None
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Graph distances (cells) to the grounding line and to the calving front.

    These are the distances d_GL and d_CF that define the PICO boxes (Reese
    et al., 2018). They are computed as in the PICO implementation of PISM,
    in edge-neighbour steps, from the shelf nodes with a grounded-ice node
    among their 8 neighbours (grounding line) and from those with open ocean
    among their 4 neighbours (calving front), both at distance 1. A shelf
    without such a front, e.g. one that ends on an open domain side, starts
    it at its nodes on that side. Paths cross the holes of the shelves and,
    with ``rises``, the ice rises, which are not grounding line. Nodes that
    cannot be reached from a seed are at 0.
    """
    grounded_ice = geom.grounded & geom.ice
    passable = geom.holes
    if rises is not None:
        grounded_ice = grounded_ice & ~rises
        passable = passable | rises
    return _shelf_distances(
        geom.shelf, grounded_ice, geom.open_ocean, geom.open_edge, passable, labels
    )
