#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Sub-grid calving front of Albrecht et al. (2011), as implemented in PISM.

Albrecht, T., Martin, M., Haseloff, M., Winkelmann, R., and Levermann, A.:
Parameterization for subgrid-scale motion of ice-shelf calving fronts, The
Cryosphere, 5, 35-44, 2011. The algorithm follows PISM (``GeometryEvolution``
part_grid and ``FrontRetreat``, https://github.com/pism/pism, GPL v3+).

One step, after the shared transport step (:mod:`.common`), which advances
the full cells and adds the ice-side inflow to the reservoir ``Href`` of the
partial cells (ice-free ocean cells next to ice):

1. **Advance.** A partial cell becomes full when ``Href >= H_r``, the
   threshold thickness (:func:`.common.threshold_thickness`); it then gets
   ``thk = H_r`` (from the flux into the cell, or PISM's mean of the neighbours
   with ``threshold: mean``) and the residual ``Href - H_r`` is split equally among its
   ice-free ocean neighbours, which may fill in turn, for up to
   ``max_iterations`` passes (PISM). What is left is kept in place. With
   ``residual: discard`` it is calved instead (Albrecht et al., variant 1).
2. **Retreat.** With the lateral ablation rate ``a = c + m_cf`` (m/yr) of
   the ``calving_rate`` process, a partial cell loses ``dt a H_r L / dx``,
   with ``L`` the front length in the cell (:func:`.common.front_faces`:
   1 at a straight front, as in PISM, and sqrt(2) across a 45-degree
   staircase, which removes PISM's orientation bias).
   What exceeds its ``Href`` is taken from the adjacent marine full cells,
   which become partial cells (``Href = thk - deficit``, ``thk = 0``): the
   front retreats by at most one cell per step, and a larger demand is
   reported in ``state.calving_unapplied_thk``.
3. **Rules and clean-up.** ``min_thickness`` calves thin floating front
   cells, ``fixed`` removes ice beyond the initial front, and ``Href`` is kept
   only on partial cells.

The front of the thickness field stays a sharp cliff: a cell joins the ice
at the full thickness ``H_r`` and leaves it as a whole.
"""

from typing import Optional, Tuple

import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State
from igm.utils.math.neighbours import any_neighbour, count_neighbours, neighbour_sum

from ..transport import explicit
from .common import (
    ablation_rate,
    apply_min_thickness,
    clean_up,
    fill_partial_cells,
    front_faces,
    initialize_front,
    publish,
    threshold_speed,
    threshold_thickness,
    transport,
)

UPDATE_MODE = "replace_transport"
COMPATIBLE_TRANSPORTS = ("explicit",)
SUPPORTED_BOUNDARY_MODES = explicit.SUPPORTED_BOUNDARY_MODES
AVAILABLE = True
UNAVAILABLE_REASON = ""

RESIDUAL_POLICIES = ("discard", "redistribute")


def initialize(cfg: DictConfig, state: State) -> None:
    p = cfg.processes.thk.front.get("sub_grid", None) or {}
    residual = str(p.get("residual", "redistribute")).strip().lower()
    if residual not in RESIDUAL_POLICIES:
        raise ValueError(
            "cfg.processes.thk.front.sub_grid.residual must be one of "
            f"{', '.join(RESIDUAL_POLICIES)}; got {residual!r}."
        )
    max_iterations = int(p.get("max_iterations", 10))
    if max_iterations < 0:
        raise ValueError(
            "cfg.processes.thk.front.sub_grid.max_iterations must be >= 0."
        )
    options = initialize_front(cfg, state)
    options["redistribute"] = residual == "redistribute"
    options["max_iterations"] = max_iterations


def update(cfg: DictConfig, state: State) -> None:
    components = state.thk_components
    options = components.component_state["front"]
    thk, href = transport(state)
    fixed = getattr(state, "front_initial_extent", None) if options["fixed"] else None
    publish(
        state,
        *front_step(
            thk,
            href,
            tf.convert_to_tensor(state.topg),
            tf.convert_to_tensor(state.water_level),
            ablation_rate(cfg, state),
            tf.cast(state.dt, thk.dtype),
            tf.cast(state.dx, thk.dtype),
            fixed,
            float(components.rho_ratio),
            int(options["max_iterations"]),
            bool(options["redistribute"]),
            float(options["min_thickness"]),
            threshold_speed(state, options),
        ),
    )


def _retreat(
    thk: tf.Tensor,
    href: tf.Tensor,
    topg: tf.Tensor,
    water_level: tf.Tensor,
    ablation: tf.Tensor,
    dt: tf.Tensor,
    dx: tf.Tensor,
    rho_ratio: float,
    speed: Optional[tf.Tensor],
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Remove ``dt a H_r L / dx`` from each front cell (PISM FrontRetreat).

    Returns the new ``(thk, Href)`` and the ablation that could not be
    applied (more than one cell of retreat in one step).
    """
    marine = topg < water_level
    ice = thk > 0.0
    front = (~ice) & marine & any_neighbour(ice) & (ablation > 0.0)
    H_r = threshold_thickness(thk, topg, water_level, rho_ratio, speed)
    fill = tf.where(H_r > 0.0, href / tf.maximum(H_r, 1e-30), tf.zeros_like(href))
    fraction = tf.where(ice, tf.ones_like(thk), tf.clip_by_value(fill, 0.0, 1.0))
    length = tf.cast(front_faces(ice, fraction), thk.dtype)
    demand = dt * ablation * H_r * length / dx
    demand = tf.where(front, demand, tf.zeros_like(demand))
    deficit = tf.maximum(demand - href, 0.0)
    href = tf.maximum(href - demand, 0.0)

    # The deficit goes to the adjacent marine full cells, in equal parts.
    target = ice & marine
    count = count_neighbours(target, thk.dtype)
    share = tf.where(
        count > 0.0, deficit / tf.maximum(count, 1.0), tf.zeros_like(deficit)
    )
    unapplied = tf.where(count > 0.0, tf.zeros_like(deficit), deficit)
    received = tf.where(target, neighbour_sum(share), tf.zeros_like(thk))
    converted = received > 0.0
    unapplied += tf.maximum(received - thk, 0.0)
    href = tf.where(converted, tf.maximum(thk - received, 0.0), href)
    thk = tf.where(converted, tf.zeros_like(thk), thk)
    return thk, href, unapplied


@tf.function(jit_compile=True)
def front_step(
    thk: tf.Tensor,
    href: tf.Tensor,
    topg: tf.Tensor,
    water_level: tf.Tensor,
    ablation: tf.Tensor,
    dt: tf.Tensor,
    dx: tf.Tensor,
    fixed_extent: Optional[tf.Tensor],
    rho_ratio: float,
    max_iterations: int,
    redistribute: bool,
    min_thickness: float,
    speed: Optional[tf.Tensor] = None,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Advance, retreat, rules and clean-up after the transport step.

    Returns ``(thk, Href, calved, unapplied, ice_area_fraction)``, where
    ``calved`` (m) is the ice removed from each cell, so that the total of
    ``thk + Href`` changes during this call by exactly ``-sum(calved)``
    (the advance only moves ice between cells).
    """
    topg = tf.cast(topg, thk.dtype)
    water_level = tf.cast(water_level, thk.dtype)
    marine = topg < water_level

    # 1. Advance: fill, then redistribute the residuals (device-side loop).
    # With "discard", the residual leaves the domain and is counted as calved.
    thk, href, discarded = fill_partial_cells(
        thk, href, topg, water_level, rho_ratio, speed, max_iterations, redistribute
    )
    before = thk + href

    # 2. Retreat by the lateral ablation rate.
    thk, href, unapplied = _retreat(
        thk, href, topg, water_level, ablation, dt, dx, rho_ratio, speed
    )

    # 3. Rules and clean-up.
    thk = apply_min_thickness(thk, topg, water_level, rho_ratio, min_thickness)
    if fixed_extent is not None:
        beyond = marine & ~fixed_extent
        thk = tf.where(beyond, tf.zeros_like(thk), thk)
        href = tf.where(beyond, tf.zeros_like(href), href)
    thk, href = clean_up(thk, href, marine)

    # Retreat, rules and clean-up only remove ice, cell by cell.
    calved = discarded + tf.maximum(before - (thk + href), 0.0)
    threshold = threshold_thickness(thk, topg, water_level, rho_ratio, speed)
    fill = tf.where(
        threshold > 0.0, href / tf.maximum(threshold, 1e-30), tf.zeros_like(href)
    )
    fraction = tf.where(
        thk > 0.0,
        tf.ones_like(thk),
        tf.where(href > 0.0, tf.minimum(fill, 1.0), tf.zeros_like(thk)),
    )
    return thk, href, calved, unapplied, fraction
