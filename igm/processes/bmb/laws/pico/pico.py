#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""PICO, the Potsdam Ice-shelf Cavity mOdel (Reese et al., 2018).

Each ice shelf is divided into n_D boxes, from the grounding line to the
calving front. With d the largest distance to the grounding line on the
shelf and d_ref the largest over all shelves (or ``reference_distance``),

    n_D = 1 + round(sqrt(d / d_ref) (n_boxes - 1)).

A node at graph distances d_GL and d_CF from the grounding line and the
calving front has the relative position r = d_GL / (d_GL + d_CF) and lies in
box k when 1 - sqrt(1 - (k-1)/n_D) <= r < 1 - sqrt(1 - k/n_D), which gives
boxes of about equal area. Water of temperature T_0 and salinity S_0 enters
box 1 at the grounding line and is carried towards the front by the
overturning

    q = C rho* (beta (S_0 - S_1) - alpha (T_0 - T_1)).

In box k the melt m = gamma_T / (nu lambda) (T_k - T_f(S_k, z)) cools and
freshens the water (nu = rho_i / rho_w, lambda = L / c_o), which determines
T_k and S_k from the means of box k - 1 (from a quadratic equation in box 1).
T_f = a S + b - c p is the potential-temperature freezing point of Reese et
al. (2018), with the pressure p = rho_w g |z| at the ice draft z.

As in Reese et al. (2018), the box areas are those of the boxes (node
counts), and T_0 and S_0 are means of the ocean fields over the
continental-shelf ocean (ice-free water with a bed above
``continental_shelf_depth``) of each basin of ``state.basins`` (a single
basin without it). The remaining choices follow the PICO implementation of
PISM: a shelf spanning several basins averages their values weighted by its
area in each, T_0 is clamped 1e-3 K above the local freezing point, and a
shelf without grounding line or calving front gets the melt of Beckmann and
Goosse (2003). Optionally, as PISM does by default, grounded areas smaller
than ``maximum_ice_rise_area`` are ice rises: not grounding line, and
crossed by the distances like the holes of the shelves. A shelf without
continental-shelf ocean in any of its basins uses instead the shelf means
of the ocean fields at its base, as Kori-ULB does, and an empty box passes
the water of the previous box on unchanged. Refreezing in the outer boxes
is kept only with ``allow_refreezing``.
"""

from typing import NamedTuple, Optional, Tuple, Union

import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State
from igm.processes.ocean.seawater import freezing_point

from ...geometry import Geometry, ice_rises, shelf_distances, shelf_labels
from ...utils import SECONDS_PER_YEAR, densities, require_process, segment_means

BECKMANN_GOOSSE_FACTOR = 5.0e-3  # PISM default of ocean.pik_melt_factor
GRAVITY = 9.81  # m s-2


class Boxes(NamedTuple):
    """Box-model solution on the shelf nodes (0 elsewhere)."""

    box: tf.Tensor  # box number, 0 outside the box model
    temp: tf.Tensor  # temperature T_k (°C)
    salinity: tf.Tensor  # salinity S_k (g kg-1)
    box_temp: tf.Tensor  # mean temperature of the node's box, T_0 in box 0 (°C)
    box_salinity: tf.Tensor  # mean salinity of the node's box, S_0 in box 0
    melt: tf.Tensor  # melt rate (m ice eq. yr-1)


class Constants(NamedTuple):
    """Parameters of the box model, as static values of the compiled solver."""

    n_boxes: int
    gamma_T: float  # m s-1
    overturning: float  # C rho* (m3 s-1)
    alpha: float  # K-1
    beta: float  # (g/kg)-1
    nu_lambda: float  # rho_i / rho_w L / c_o (K)
    freezing: Tuple[float, float, float]  # a, b, c rho_w g, T_f in depth z
    reference_distance: float  # m, 0 for the largest in the domain
    continental_shelf_depth: float  # m


def initialize(cfg: DictConfig, state: State) -> None:
    require_process(cfg, "ocean", "pico")
    densities(cfg, "pico")


def melt_rate(cfg: DictConfig, state: State, geom: Geometry) -> tf.Tensor:
    boxes = solve_boxes(cfg, state, geom)
    publish(state, boxes)
    return boxes.melt


def publish(state: State, boxes: Boxes) -> None:
    """Box number, temperature and salinity diagnostics."""
    state.pico_box = tf.cast(boxes.box, boxes.temp.dtype)
    state.pico_temp = boxes.temp
    state.pico_salinity = boxes.salinity


def constants(cfg: DictConfig) -> Constants:
    p = cfg.processes.bmb.pico
    physics = cfg.processes.bmb.physics
    rho_i, rho_w = densities(cfg, "pico")
    return Constants(
        n_boxes=int(p.n_boxes),
        gamma_T=float(p.gamma_T),
        overturning=float(p.overturning * p.rho_star),
        alpha=float(p.alpha),
        beta=float(p.beta),
        nu_lambda=float(rho_i / rho_w * physics.L_ice / physics.c_ocean),
        freezing=(
            float(p.freezing_point.a),
            float(p.freezing_point.b),
            float(p.freezing_point.c) * rho_w * GRAVITY,
        ),
        reference_distance=float(p.reference_distance),
        continental_shelf_depth=float(p.continental_shelf_depth),
    )


def solve_boxes(cfg: DictConfig, state: State, geom: Geometry) -> Boxes:
    p = cfg.processes.bmb.pico
    rises = None
    if p.maximum_ice_rise_area > 0.0:
        rises = ice_rises(geom, state.dx, p.maximum_ice_rise_area * 1.0e6)
    labels = shelf_labels(geom, rises)
    d_gl, d_cf = shelf_distances(geom, labels, rises)
    dtype = geom.draft.dtype
    if hasattr(state, "basins"):
        basins = tf.cast(tf.round(state.basins), tf.int32)
    else:
        basins = tf.zeros_like(labels)
    return Boxes(
        *_solve_boxes(
            geom.draft,
            geom.shelf,
            geom.open_ocean,
            tf.cast(state.topg - state.water_level, dtype),
            tf.cast(state.ocean_temp, dtype),
            tf.cast(state.ocean_salinity, dtype),
            basins,
            labels,
            d_gl,
            d_cf,
            tf.cast(state.dx, dtype),
            constants(cfg),
        )
    )


def box_numbers(
    n_boxes: int,
    d_ref: Optional[Union[float, tf.Tensor]],
    labels: tf.Tensor,
    d_gl: tf.Tensor,
    d_cf: tf.Tensor,
) -> tf.Tensor:
    """PICO box of each shelf node, 0 without grounding line or front.

    ``d_ref`` is the reference distance in cells, or None for the largest
    distance to the grounding line in the domain. Uses the closed form
    k = 1 + floor(n_D r (2 - r)) of the box boundaries, with
    r (2 - r) = d_GL (d_GL + 2 d_CF) / (d_GL + d_CF)^2, capped at n_D and at
    d_GL as in PISM.
    """
    num_segments = tf.size(labels) + 1
    d_max = tf.math.unsorted_segment_max(d_gl, labels, num_segments)
    d_max = tf.cast(tf.gather(d_max, labels), tf.float64)
    if d_ref is None:
        d_ref = tf.cast(tf.reduce_max(d_gl), tf.float64)
    d_ref = tf.maximum(tf.cast(d_ref, tf.float64), 1.0)

    n_d = tf.floor(1.0 + tf.sqrt(tf.maximum(d_max, 0.0) / d_ref) * (n_boxes - 1) + 0.5)
    n_d = tf.clip_by_value(n_d, 1.0, float(n_boxes))
    d = tf.cast(d_gl, tf.float64)
    f = tf.cast(d_cf, tf.float64)
    k = 1.0 + tf.floor(n_d * d * (d + 2.0 * f) / tf.maximum(d + f, 1.0) ** 2)
    k = tf.minimum(tf.minimum(k, n_d), d)
    return tf.where((d_gl > 0) & (d_cf > 0), tf.cast(k, tf.int32), 0)


def ocean_input(
    temp: tf.Tensor,
    salinity: tf.Tensor,
    shelf: tf.Tensor,
    shelf_sea: tf.Tensor,
    basins: tf.Tensor,
    labels: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Temperature and salinity T_0, S_0 entering each shelf (per node).

    Means over the continental-shelf ocean ``shelf_sea`` of each basin,
    averaged over the basins of each shelf weighted by its nodes in each; a
    shelf without such ocean takes the means of the fields at its base.
    """
    num_segments = tf.size(labels) + 1
    basins = tf.clip_by_value(basins, 0, num_segments - 1)
    (temp_b, salinity_b), count_b = segment_means(
        [temp, salinity], basins, shelf_sea, num_segments
    )
    (temp_s, salinity_s), count_s = segment_means(
        [temp_b, salinity_b], labels, shelf & (count_b > 0), num_segments
    )
    (temp_base, salinity_base), _ = segment_means(
        [temp, salinity], labels, shelf, num_segments
    )
    return (
        tf.where(count_s > 0, temp_s, temp_base),
        tf.where(count_s > 0, salinity_s, salinity_base),
    )


@tf.function(autograph=False, jit_compile=True)
def _solve_boxes(
    z: tf.Tensor,
    shelf: tf.Tensor,
    open_ocean: tf.Tensor,
    bed: tf.Tensor,
    temp: tf.Tensor,
    salinity: tf.Tensor,
    basins: tf.Tensor,
    labels: tf.Tensor,
    d_gl: tf.Tensor,
    d_cf: tf.Tensor,
    dx: tf.Tensor,
    c: Constants,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    num_segments = tf.size(labels) + 1
    d_ref = c.reference_distance / dx if c.reference_distance > 0.0 else None
    box = tf.where(shelf, box_numbers(c.n_boxes, d_ref, labels, d_gl, d_cf), 0)

    # Area of each box of the node's shelf, all boxes in one reduction.
    in_boxes = tf.cast(box[..., tf.newaxis] == tf.range(1, c.n_boxes + 1), z.dtype)
    counts = tf.math.unsorted_segment_sum(in_boxes, labels, num_segments)
    area = tf.gather(counts, labels) * dx * dx

    def freezing(salinity):
        return freezing_point(salinity, z, c.freezing)

    # Placeholder values off the shelf keep every expression finite there.
    shelf_sea = open_ocean & (bed > c.continental_shelf_depth)
    temp0, salinity0 = ocean_input(temp, salinity, shelf, shelf_sea, basins, labels)
    salinity0 = tf.where(shelf, salinity0, 35.0)
    temp0 = tf.maximum(temp0, freezing(salinity0) + 1.0e-3)
    temp0 = tf.where(shelf, temp0, 0.0)

    # Box 1: quadratic equation for the cooling x = T_0 - T_1.
    t_star = freezing(salinity0) - temp0
    s = salinity0 / c.nu_lambda
    p = area[..., 0] * c.gamma_T / (c.overturning * (c.beta * s - c.alpha))
    root = p / 2.0 + tf.sqrt(tf.maximum(p**2 / 4.0 - p * t_star, 0.0))
    x = tf.math.divide_no_nan(-p * t_star, root)
    in_box = box == 1
    temp = tf.where(in_box, temp0 - x, temp0)
    salinity = tf.where(in_box, salinity0 - s * x, salinity0)
    overturning = c.overturning * (c.beta * s - c.alpha) * x  # m3 s-1
    (temp_in, salinity_in, q), _ = segment_means(
        [temp, salinity, overturning], labels, in_box, num_segments
    )
    box_temp = tf.where(in_box, temp_in, temp0)
    box_salinity = tf.where(in_box, salinity_in, salinity0)

    # Boxes k > 1, fed by the means of box k - 1 and the box-1 overturning.
    for k in range(2, c.n_boxes + 1):
        g1 = area[..., k - 1] * c.gamma_T
        t_star = freezing(salinity_in) - temp_in
        x = tf.math.divide_no_nan(
            -g1 * t_star, q + g1 * (1.0 - c.freezing[0] * salinity_in / c.nu_lambda)
        )
        in_box = box == k
        temp = tf.where(in_box, temp_in - x, temp)
        salinity = tf.where(in_box, salinity_in * (1.0 - x / c.nu_lambda), salinity)
        # An empty box k passes the water of box k - 1 on unchanged.
        (temp_mean, salinity_mean), count = segment_means(
            [temp, salinity], labels, in_box, num_segments
        )
        temp_in = tf.where(count > 0, temp_mean, temp_in)
        salinity_in = tf.where(count > 0, salinity_mean, salinity_in)
        box_temp = tf.where(in_box, temp_in, box_temp)
        box_salinity = tf.where(in_box, salinity_in, box_salinity)

    exchange = c.gamma_T / c.nu_lambda * SECONDS_PER_YEAR  # m ice yr-1 K-1
    melt = exchange * (temp - freezing(salinity))
    melt = tf.where(box > 0, melt, BECKMANN_GOOSSE_FACTOR * melt)

    return (
        box,
        tf.where(shelf, temp, 0.0),
        tf.where(shelf, salinity, 0.0),
        tf.where(shelf, box_temp, 0.0),
        tf.where(shelf, box_salinity, 0.0),
        tf.where(shelf, melt, 0.0),
    )
