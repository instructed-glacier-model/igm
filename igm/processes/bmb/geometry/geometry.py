#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Ice-ocean geometry: flotation, exposure to the ocean, and floating fraction.

Flotation follows ``thk.masks`` exactly, so the grounding line seen by the
basal mass balance is the one of the thickness and ice-flow processes. Water
(floating ice and ice-free water) is exposed to the ocean when it connects,
through cell edges, to a domain side open to the ocean (a ``zero`` thickness
boundary) or, when no such side touches water, to the ice-free water.
Floating ice elsewhere, such as over a subglacial lake, receives no ocean
melt. Periodic sides are not wrapped around.

The open ocean is the exposed ice-free water connected to the outer ocean
through ice-free water only: through an open side, or else the largest
body of ice-free water. Ice-free water enclosed by an ice shelf (a hole) is
not open ocean, so it is neither a calving front nor an input of PICO.
"""

from typing import NamedTuple, Sequence, Tuple

import numpy as np
import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State
from igm.processes.ocean.ocean import ocean_depth
from igm.processes.thk.boundary import get_boundary_conditions
from igm.processes.thk.masks import flotation_function
from igm.processes.thk.surfaces import get_density_ratio
from igm.utils.math.connectivity import label_components, reach

from ..utils import shifted


class Geometry(NamedTuple):
    """Fields and masks of one update, on the nodes of the IGM grid."""

    thk: tf.Tensor  # thickness of the true ice columns (m)
    draft: (
        tf.Tensor
    )  # ice base, or sea floor without ice, rel. to water level (m, <= 0)
    phi: tf.Tensor  # flotation function, positive where grounded
    ice: tf.Tensor  # ice-covered nodes
    grounded: tf.Tensor  # grounded ice and ice-free land
    shelf: tf.Tensor  # floating ice exposed to the ocean
    open_ocean: tf.Tensor  # ice-free water connected to the outer ocean
    holes: tf.Tensor  # exposed ice-free water enclosed by ice shelves
    open_edge: tf.Tensor  # nodes on the domain sides open to the ocean


def open_edges(cfg: DictConfig, shape: Sequence[int]) -> tf.Tensor:
    """Nodes on the domain sides through which the ocean extends (``zero``)."""
    sides = get_boundary_conditions(cfg)
    edge = np.zeros(shape, bool)
    edge[0, :] |= sides.top == "zero"
    edge[-1, :] |= sides.bottom == "zero"
    edge[:, 0] |= sides.left == "zero"
    edge[:, -1] |= sides.right == "zero"
    return tf.constant(edge)


def largest_component(mask: tf.Tensor) -> tf.Tensor:
    """The largest edge-connected component of the bool ``mask``."""
    labels = label_components(mask)
    counts = tf.math.unsorted_segment_sum(
        tf.cast(mask, tf.int32), labels, tf.size(labels) + 1
    )
    largest = tf.cast(tf.argmax(counts[1:]), tf.int32) + 1
    return mask & tf.equal(labels, largest)


@tf.function(autograph=False, jit_compile=True)
def _geometry(
    thk: tf.Tensor,
    topg: tf.Tensor,
    water_level: tf.Tensor,
    lsurf: tf.Tensor,
    open_edge: tf.Tensor,
    rho_ratio: float,
    connected_only: bool,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    phi = flotation_function(thk, topg, water_level, rho_ratio)
    ice = thk > 0.0
    grounded = phi > 0.0
    water = tf.logical_not(grounded)
    if connected_only:
        edge_seed = water & open_edge
        seed = tf.where(tf.reduce_any(edge_seed), edge_seed, water & ~ice)
        exposed = reach(seed, water)
    else:
        exposed = water
    free = exposed & ~ice
    free_seed = free & open_edge
    open_ocean = tf.cond(
        tf.reduce_any(free_seed),
        lambda: reach(free_seed, free),
        lambda: largest_component(free),
    )
    draft = ocean_depth(thk, lsurf, topg, water_level)
    return draft, phi, ice, grounded, exposed & ice, open_ocean, free & ~open_ocean


def compute_geometry(cfg: DictConfig, state: State) -> Geometry:
    thk = state.thk
    if hasattr(state, "_bmb_open_edge"):
        open_edge = state._bmb_open_edge
    else:
        open_edge = open_edges(cfg, thk.shape)
    fields = _geometry(
        tf.convert_to_tensor(thk),
        tf.convert_to_tensor(state.topg),
        tf.convert_to_tensor(state.water_level),
        tf.convert_to_tensor(state.lsurf),
        open_edge,
        1.0 / get_density_ratio(cfg),
        bool(cfg.processes.bmb.ocean_connected_only),
    )
    return Geometry(tf.convert_to_tensor(thk), *fields, open_edge)


@tf.function(autograph=False, jit_compile=True)
def floating_fraction(phi: tf.Tensor, samples: int) -> tf.Tensor:
    """Floating fraction of each node's dual cell from the flotation function.

    ``samples`` points per direction and per quarter cell, at the centres of
    a regular subdivision; edge nodes see a mirrored dual cell.
    """
    offsets = [float(t) for t in (np.arange(samples) + 0.5) / (2 * samples)]
    # Mirror each axis about the edge node (itself on a one-node axis).
    rows, cols = ("REFLECT" if n > 1 else "SYMMETRIC" for n in phi.shape)
    p = tf.pad(tf.pad(phi, [[1, 1], [0, 0]], mode=rows), [[0, 0], [1, 1]], mode=cols)
    floating = tf.zeros_like(phi)
    for sy in (-1, 1):
        for sx in (-1, 1):
            centre, across, along, diagonal = (
                shifted(p, 0, 0),
                shifted(p, sy, 0),
                shifted(p, 0, sx),
                shifted(p, sy, sx),
            )
            for a in offsets:
                for b in offsets:
                    sample = (
                        (1.0 - a) * (1.0 - b) * centre
                        + (1.0 - a) * b * along
                        + a * (1.0 - b) * across
                        + a * b * diagonal
                    )
                    floating += tf.cast(sample <= 0.0, phi.dtype)
    return floating / (4.0 * samples**2)
