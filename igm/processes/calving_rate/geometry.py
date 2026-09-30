#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Where the calving laws are evaluated, and where their rate applies.

A law is evaluated on the ice nodes that carry velocity in the ice flow
(``masks.iceflow_node_mask``). The rate applies in the *front band*: the
front cells of both sides plus ``band`` rings of dilation around them (so
band + 1 cells each side), where the front borders the open ocean; laws
that are carried by :func:`..calving_rate.spread` reach ``band`` rings into
the ocean, so the outermost ocean ring keeps a zero rate. The open ocean
is the ice-free water connected
through cell edges to a domain side open to the ocean (a ``zero`` thickness
boundary), or else the largest body of ice-free water; ice-free water
enclosed in an ice shelf (a hole, a rift) is not a calving front.

``front_speed`` is the ice speed at the front, with which the front moves:
the velocity of the supported nodes extended into the band, linearly into
the first ring of cells (the front's level-set velocity, and on a flowline
the speed that continuity gives the ice filling a partial cell), then by
the mean over the neighbours. It extrapolates the velocity components,
where the flux fill threshold of ``thk.fronts`` extrapolates the speed: the
two differ slightly for oblique flow.

The open-ocean helpers are shared with the ``bmb`` process
(``processes.bmb.geometry``), which owns them.
"""

from typing import NamedTuple

import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State
from igm.processes.bmb.geometry.geometry import largest_component, open_edges
from igm.processes.thk.fronts.common import extend_velocity
from igm.processes.thk.masks import flotation_function, iceflow_node_mask
from igm.processes.thk.surfaces import get_density_ratio
from igm.utils.math.connectivity import reach
from igm.utils.math.getmag import getmag
from igm.utils.math.neighbours import any_neighbour, dilate


class FrontGeometry(NamedTuple):
    """Fields and masks of one update, on the nodes of the IGM grid."""

    thk: tf.Tensor  # ice thickness (m)
    ice: tf.Tensor  # ice-covered nodes
    supported: tf.Tensor  # ice nodes carrying velocity in the ice flow
    floating: tf.Tensor  # ice at or below flotation
    marine: tf.Tensor  # bed below the water level
    water_depth: tf.Tensor  # max(water_level - topg, 0) (m)
    speed: tf.Tensor  # |(ubar, vbar)| on the supported nodes (m/yr)
    front_speed: tf.Tensor  # ice speed extended to the band (m/yr, see above)
    band: tf.Tensor  # cells where the rate applies
    dx: tf.Tensor  # grid spacing (m)


def open_ocean(
    free: tf.Tensor, open_edge: tf.Tensor, connected_only: bool
) -> tf.Tensor:
    """Ice-free water connected to the outer ocean (see the module docstring)."""
    if not connected_only:
        return free
    seed = free & open_edge
    return tf.cond(
        tf.reduce_any(seed), lambda: reach(seed, free), lambda: largest_component(free)
    )


def front_geometry(cfg: DictConfig, state: State) -> FrontGeometry:
    """The geometry of the current state, with a front band ``band`` cells wide."""
    p = cfg.processes.calving_rate
    thk = tf.convert_to_tensor(state.thk)
    dtype = thk.dtype
    if hasattr(state, "_calving_open_edge"):
        open_edge = state._calving_open_edge
    else:
        open_edge = open_edges(cfg, thk.shape)
    has_velocity = hasattr(state, "ubar") and hasattr(state, "vbar")
    ubar = tf.cast(state.ubar, dtype) if has_velocity else tf.zeros_like(thk)
    vbar = tf.cast(state.vbar, dtype) if has_velocity else tf.zeros_like(thk)
    return _front_geometry(
        thk,
        tf.cast(state.topg, dtype),
        tf.cast(state.water_level, dtype),
        tf.cast(state.usurf, dtype),
        ubar,
        vbar,
        tf.cast(state.dx, dtype),
        tf.convert_to_tensor(open_edge),
        1.0 / get_density_ratio(cfg),
        bool(p.ocean_connected_only),
        int(p.band),
        has_velocity,
    )


@tf.function(autograph=False, jit_compile=True)
def _front_geometry(
    thk: tf.Tensor,
    topg: tf.Tensor,
    water_level: tf.Tensor,
    usurf: tf.Tensor,
    ubar: tf.Tensor,
    vbar: tf.Tensor,
    dx: tf.Tensor,
    open_edge: tf.Tensor,
    rho_ratio: float,
    connected_only: bool,
    band_width: int,
    has_velocity: bool,
) -> FrontGeometry:
    """Build front geometry from tensors in one compiled device graph."""
    ice = thk > 0.0
    marine = topg < water_level
    supported = iceflow_node_mask(thk, usurf, water_level, rho_ratio)
    floating = ice & (flotation_function(thk, topg, water_level, rho_ratio) <= 0.0)
    ocean = open_ocean(~ice & marine, open_edge, connected_only)
    seed = (ocean & any_neighbour(ice)) | (ice & any_neighbour(ocean))
    band = dilate(seed, band_width) & marine
    if has_velocity:
        speed = tf.where(supported, getmag(ubar, vbar), tf.zeros_like(thk))
        ubar, vbar = extend_velocity(
            ubar, vbar, supported, band, band_width, linear=True
        )
        front_speed = tf.where(supported | band, getmag(ubar, vbar), tf.zeros_like(thk))
    else:
        speed = front_speed = tf.zeros_like(thk)
    return FrontGeometry(
        thk=thk,
        ice=ice,
        supported=supported,
        floating=floating,
        marine=marine,
        water_depth=tf.maximum(water_level - topg, 0.0),
        speed=speed,
        front_speed=front_speed,
        band=band,
        dx=dx,
    )
