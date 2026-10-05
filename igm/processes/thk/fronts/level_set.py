#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Mass-consistent level-set calving front, after Bondzio et al. (2016).

Bondzio, J. H., Seroussi, H., Morlighem, M., Kleiner, T., Rückamp, M.,
Humbert, A., and Larour, E. Y.: Modelling calving front dynamics using a
level-set method: application to Jakobshavn Isbræ, West Greenland, The
Cryosphere, 10, 497-510, 2016.

The front is the zero contour of ``state.psi`` (m, a signed distance,
negative in the ice), which moves with the front velocity
``w = u - a n`` (Bondzio et al., Eqs. 6-7), with ``a = c + m_cf`` the lateral
ablation rate of the ``calving_rate`` process and ``n = grad psi / |grad psi|``:

    d psi / dt + w . grad psi = 0,

first-order upwind in device-side substeps that keep the front CFL number
below 1/2. ``w`` vanishes at a stationary front.

Unlike ISSM (and Kori-ULB), where ice is removed or created node by node,
the mass follows the level set exactly:

* **Fill fraction.** A cell on the front (its ``psi`` changes sign with an
  edge neighbour) is covered by the fraction ``phi = clip(1/2 - psi/w)``,
  with ``w = dx (|n_x| + |n_y|)`` its width across the front; other cells are
  fully in or out.
* **Transport** is the shared step (:mod:`.common`): full cells advance, and
  the ice-side inflow goes into the reservoir ``Href`` of the partial cells.
* **Calving** removes the ice over the area the front leaves: the column
  thickness of a partial cell is kept, ``Href <- Href phi / A_adv`` with
  ``A_adv`` the area the ice has flowed over during the step (the fraction
  after the advection alone, not clipped at the cell edge). The calved volume is
  therefore ``a H`` per unit front length and time.
* **Advance and retreat.** A partial cell whose fraction reaches 1 (or whose
  volume already fills it) becomes full at the threshold thickness (as in
  :mod:`.sub_grid`), and its residual goes to the neighbouring cells the
  front has entered; a full cell the ablation has entered becomes a partial
  cell holding ``thk phi / phi_adv`` (the covered area left by the calving
  alone). A trailing edge -- ice advected away from an ice-free cell -- is
  already thinned to the correct cell mean by the transport, so it stays a
  full cell and moves cell by cell, as in :mod:`.sub_grid`.
* **Consistency.** ``psi`` is tied to the ice after every step: inside every
  full cell, outside every cell without ice or reservoir (whatever emptied
  it: calving, the front rules, the rigid-body pass, a restart), and free
  only in the partial cells. With a front held by a rule (a thickness
  threshold, no calving rate) the level set would otherwise keep moving with
  the ice into the ocean while the rule removes the ice behind it.
* **Velocity.** The level set moves with the ice velocity extended into the
  ocean, linearly in the first ring: on a spreading shelf the ice at the
  front moves faster than at the last full node, and continuity (the fill of
  the partial cells) advances the ice at that speed.

``psi`` is re-distanced (Sussman et al., 1994) every ``reinit_freq`` steps
(default: every step, which keeps the upwind front speed exact), with the
partial cells frozen and the full and empty cells clamped again after, so
that no fill fraction, hence no mass, changes. Land margins keep ordinary
transport; ``psi`` follows them.
"""

from typing import Optional, Tuple

import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State
from igm.utils.math.neighbours import any_neighbour, count_neighbours, neighbour_sum

from ..masks import iceflow_node_mask
from ..transport import explicit
from .common import (
    ablation_rate,
    apply_min_thickness,
    clean_up,
    extend_velocity,
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

#: Front CFL number of the level-set substeps.
LEVEL_SET_CFL = 0.5

#: A cell counts as full once its fill fraction is within this of 1.
FULL = 1.0 - 1.0e-4

#: Full and far empty cells are held this relative distance beyond half their
#: width inside (outside) the front, so that the change of the width across
#: the front from one step to the next does not demote full cells or pull
#: empty ones into it. Empty cells NEXT TO the ice sit exactly at half their
#: width: an outward margin there would hold the front out of reach of the
#: cell for ``u dt < margin dx``, and its routed inflow would be wiped as
#: spurious calving every step (a slow front in a domain whose time step is
#: set by faster ice elsewhere).
MARGIN = 0.01


def initialize(cfg: DictConfig, state: State) -> None:
    p = cfg.processes.thk.front.get("level_set", None) or {}
    options = initialize_front(cfg, state)
    options["reinit_freq"] = int(p.get("reinit_freq", 1))
    options["reinit_iter"] = int(p.get("reinit_iter", 5))
    options["band"] = int(p.get("band", 3))
    options["steps_since_reinit"] = 0
    if options["reinit_iter"] < 0 or options["band"] < 1:
        raise ValueError(
            "cfg.processes.thk.front.level_set needs reinit_iter >= 0 and band >= 1."
        )

    thk = tf.convert_to_tensor(state.thk)
    dx = tf.cast(state.dx, thk.dtype)
    psi = getattr(state, "psi", None)
    if psi is None or tuple(psi.shape) != tuple(thk.shape):
        threshold = threshold_thickness(
            thk,
            tf.cast(state.topg, thk.dtype),
            tf.cast(state.water_level, thk.dtype),
            float(state.thk_components.rho_ratio),
            threshold_speed(state, options) if hasattr(state, "ubar") else None,
        )
        # Synchronise ties the fresh psi to the ice: on an oblique front the
        # raw ramp (built with width dx) puts full cells inside by less than
        # half their true width, and the first step would demote them.
        state.psi = synchronise(
            initial_psi(thk, state.Href, threshold, dx, 4 * options["band"] + 10),
            thk,
            tf.cast(state.Href, thk.dtype),
            dx,
            options["reinit_iter"],
        )
    else:
        state.psi = synchronise(
            tf.cast(tf.convert_to_tensor(psi), thk.dtype),
            thk,
            tf.cast(state.Href, thk.dtype),
            dx,
            4 * options["band"] + 10,
        )
    if options["fixed"]:
        state.front_initial_psi = tf.identity(state.psi)
    href = tf.cast(state.Href, thk.dtype)
    state.ice_area_fraction = tf.where(
        thk > 0.0,
        tf.ones_like(thk),
        tf.where(href > 0.0, fill_fraction(state.psi, dx), tf.zeros_like(thk)),
    )


def update(cfg: DictConfig, state: State) -> None:
    components = state.thk_components
    options = components.component_state["front"]
    rho_ratio = float(components.rho_ratio)
    thk_old = tf.convert_to_tensor(state.thk)
    thk, href = transport(state)
    dtype = thk.dtype
    dx = tf.cast(state.dx, dtype)
    # The level set moves with the ice velocity extended linearly into the
    # first ring (the speed of the ice at the front of a spreading shelf).
    ubar, vbar = _psi_velocity(
        thk_old,
        tf.cast(state.topg, dtype),
        tf.cast(state.water_level, dtype),
        tf.cast(state.usurf, dtype),
        tf.cast(state.ubar, dtype),
        tf.cast(state.vbar, dtype),
        rho_ratio,
        int(options["band"]),
    )
    floor = getattr(state, "front_initial_psi", None) if options["fixed"] else None
    psi_adv, psi_new = advance_psi(
        tf.cast(state.psi, dtype),
        ubar,
        vbar,
        ablation_rate(cfg, state),
        tf.cast(state.dt, dtype),
        dx,
        floor,
    )
    thk, href, psi, calved, fraction = mass_step(
        thk,
        href,
        psi_adv,
        psi_new,
        tf.cast(state.topg, dtype),
        tf.cast(state.water_level, dtype),
        dx,
        rho_ratio,
        float(options["min_thickness"]),
        threshold_speed(state, options),
    )

    options["steps_since_reinit"] += 1
    if (
        options["reinit_freq"] > 0
        and options["steps_since_reinit"] >= options["reinit_freq"]
    ):
        psi = synchronise(psi, thk, href, dx, options["reinit_iter"])
        options["steps_since_reinit"] = 0

    state.psi = psi
    publish(state, thk, href, calved, tf.zeros_like(thk), fraction)


# ---------------------------------------------------------------------------
# Level-set kernels
# ---------------------------------------------------------------------------


@tf.function(jit_compile=True)
def _psi_velocity(
    thk: tf.Tensor,
    topg: tf.Tensor,
    water_level: tf.Tensor,
    usurf: tf.Tensor,
    ubar: tf.Tensor,
    vbar: tf.Tensor,
    rho_ratio: float,
    band: int,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Ice velocity extended ``band`` rings into the ocean, linearly in the first."""
    known = iceflow_node_mask(thk, usurf, water_level, rho_ratio)
    targets = (topg < water_level) | (thk > 0.0)
    return extend_velocity(ubar, vbar, known, targets, band, linear=True)


def _one_sided(psi: tf.Tensor, dx: tf.Tensor) -> Tuple[tf.Tensor, ...]:
    """Backward and forward differences in x and y (zero gradient at edges)."""
    p = tf.pad(psi, [[1, 1], [1, 1]], mode="SYMMETRIC")
    c = p[1:-1, 1:-1]
    return (
        (c - p[1:-1, :-2]) / dx,
        (p[1:-1, 2:] - c) / dx,
        (c - p[:-2, 1:-1]) / dx,
        (p[2:, 1:-1] - c) / dx,
    )


def _grad_upwind(
    Dxm: tf.Tensor, Dxp: tf.Tensor, Dym: tf.Tensor, Dyp: tf.Tensor, outward: bool
) -> tf.Tensor:
    """Godunov |grad psi| for a front moving outward (psi decreasing) or inward.

    Per axis the larger of the two admissible one-sided slopes (squared), not
    their sum, which would overestimate the gradient at kinks.
    """
    if outward:
        gx = tf.maximum(
            tf.square(tf.maximum(Dxm, 0.0)), tf.square(tf.minimum(Dxp, 0.0))
        )
        gy = tf.maximum(
            tf.square(tf.maximum(Dym, 0.0)), tf.square(tf.minimum(Dyp, 0.0))
        )
    else:
        gx = tf.maximum(
            tf.square(tf.minimum(Dxm, 0.0)), tf.square(tf.maximum(Dxp, 0.0))
        )
        gy = tf.maximum(
            tf.square(tf.minimum(Dym, 0.0)), tf.square(tf.maximum(Dyp, 0.0))
        )
    return tf.sqrt(gx + gy)


def front_cells(psi: tf.Tensor) -> tf.Tensor:
    """Cells whose sign of ``psi`` differs from one of their edge neighbours."""
    inside = psi < 0.0
    return (inside & any_neighbour(~inside)) | (~inside & any_neighbour(inside))


def ramp_width(psi: tf.Tensor, dx: tf.Tensor) -> tf.Tensor:
    """Width of a cell across the front, ``dx (|n_x| + |n_y|)`` (dx to sqrt(2) dx).

    The covered fraction of a square cell whose centre is at the signed
    distance ``psi`` from a straight front goes from 1 to 0 over this width.
    """
    p = tf.pad(psi, [[1, 1], [1, 1]], mode="SYMMETRIC")
    gx = p[1:-1, 2:] - p[1:-1, :-2]
    gy = p[2:, 1:-1] - p[:-2, 1:-1]
    norm = tf.sqrt(gx * gx + gy * gy)
    width = dx * (tf.abs(gx) + tf.abs(gy)) / tf.maximum(norm, 1e-30)
    return tf.where(norm > 0.0, width, dx + 0.0 * psi)


def fill_fraction(
    psi: tf.Tensor, dx: tf.Tensor, width: Optional[tf.Tensor] = None
) -> tf.Tensor:
    """Ice-covered fraction of each cell: linear across the front, else 0 or 1."""
    width = ramp_width(psi, dx) if width is None else width
    ramp = tf.clip_by_value(0.5 - psi / width, 0.0, 1.0)
    return tf.where(front_cells(psi), ramp, tf.cast(psi < 0.0, psi.dtype))


def _normal(psi: tf.Tensor, dx: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Unit normal ``grad psi / |grad psi|`` (central differences), outward."""
    p = tf.pad(psi, [[1, 1], [1, 1]], mode="SYMMETRIC")
    gx = (p[1:-1, 2:] - p[1:-1, :-2]) / (2.0 * dx)
    gy = (p[2:, 1:-1] - p[:-2, 1:-1]) / (2.0 * dx)
    norm = tf.maximum(tf.sqrt(gx * gx + gy * gy), 1e-12)
    return gx / norm, gy / norm


def _upwind(psi: tf.Tensor, wx: tf.Tensor, wy: tf.Tensor, dx: tf.Tensor) -> tf.Tensor:
    """First-order upwind ``w . grad psi``."""
    Dxm, Dxp, Dym, Dyp = _one_sided(psi, dx)
    return (
        tf.maximum(wx, 0.0) * Dxm
        + tf.minimum(wx, 0.0) * Dxp
        + tf.maximum(wy, 0.0) * Dym
        + tf.minimum(wy, 0.0) * Dyp
    )


@tf.function(jit_compile=True)
def advance_psi(
    psi: tf.Tensor,
    ubar: tf.Tensor,
    vbar: tf.Tensor,
    ablation: tf.Tensor,
    dt: tf.Tensor,
    dx: tf.Tensor,
    floor: Optional[tf.Tensor] = None,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Move the front with ``w = u - a n`` (Bondzio et al., 2016, Eq. 6).

    Returns ``psi`` moved by the ice velocity alone (the area the ice flows
    into, for the mass) and by the front velocity ``w``; ``w`` vanishes at a
    stationary front, so no splitting error builds up there. With a
    ``floor`` (the initial ``psi``, for a fixed front) the front cannot pass
    its initial position.
    """
    speed = tf.reduce_max(tf.abs(ubar) + tf.abs(vbar) + 2.0 * tf.abs(ablation))
    substeps = tf.maximum(
        1, tf.cast(tf.math.ceil(speed * dt / (LEVEL_SET_CFL * dx)), tf.int32)
    )
    h = dt / tf.cast(substeps, psi.dtype)

    def advect(i: tf.Tensor, psi: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return i + 1, psi - h * _upwind(psi, ubar, vbar, dx)

    def move(i: tf.Tensor, psi: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        nx, ny = _normal(psi, dx)
        return i + 1, psi - h * _upwind(
            psi, ubar - ablation * nx, vbar - ablation * ny, dx
        )

    _, psi_adv = tf.while_loop(lambda i, _: i < substeps, advect, (0, psi))
    _, psi_new = tf.while_loop(lambda i, _: i < substeps, move, (0, psi))
    if floor is not None:
        psi_new = tf.maximum(psi_new, tf.cast(floor, psi.dtype))
    # The ablation only moves the front back: never beyond the advected one.
    psi_new = tf.maximum(psi_new, psi_adv)
    return psi_adv, psi_new


@tf.function(jit_compile=True)
def reinitialise(
    psi: tf.Tensor,
    dx: tf.Tensor,
    iterations: int,
    frozen: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    """Sussman re-distancing; ``frozen`` cells (default: the front cells) keep ``psi``."""
    frozen = front_cells(psi) if frozen is None else frozen
    sign = psi / tf.sqrt(psi * psi + dx * dx)
    step = 0.5 * dx

    def body(i: tf.Tensor, psi: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        Dxm, Dxp, Dym, Dyp = _one_sided(psi, dx)
        grad = tf.where(
            sign > 0.0,
            _grad_upwind(Dxm, Dxp, Dym, Dyp, outward=True),
            _grad_upwind(Dxm, Dxp, Dym, Dyp, outward=False),
        )
        return i + 1, tf.where(frozen, psi, psi - step * sign * (grad - 1.0))

    _, psi = tf.while_loop(lambda i, _: i < iterations, body, (0, psi))
    return psi


def constrain(
    psi: tf.Tensor,
    thk: tf.Tensor,
    href: tf.Tensor,
    dx: tf.Tensor,
    width: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    """``psi`` inside every full cell and outside every cell without ice or reservoir.

    An empty cell next to the ice sits exactly at half its width, so that the
    smallest advection brings the front into it (see ``MARGIN``); empty cells
    farther out keep the margin, so that the change of the width from one
    step to the next cannot pull them spuriously into the front.
    """
    width = ramp_width(psi, dx) if width is None else width
    ice = thk > 0.0
    psi = tf.where(ice, tf.minimum(psi, -(0.5 + 0.5 * MARGIN) * width), psi)
    empty = ~ice & (href <= 0.0)
    edge = tf.where(any_neighbour(ice), 0.5 * width, (0.5 + 0.5 * MARGIN) * width)
    return tf.where(empty, tf.maximum(psi, edge), psi)


@tf.function(jit_compile=True)
def synchronise(
    psi: tf.Tensor, thk: tf.Tensor, href: tf.Tensor, dx: tf.Tensor, iterations: int
) -> tf.Tensor:
    """Re-distance ``psi`` around the partial cells and tie it to the ice."""
    partial = (thk <= 0.0) & (href > 0.0)
    psi = reinitialise(constrain(psi, thk, href, dx), dx, iterations, partial)
    return constrain(psi, thk, href, dx)


def initial_psi(
    thk: tf.Tensor,
    href: tf.Tensor,
    threshold: tf.Tensor,
    dx: tf.Tensor,
    iterations: int,
) -> tf.Tensor:
    """``psi`` whose fill fractions match the ice and the partial cells."""
    fill = tf.where(
        threshold > 0.0, href / tf.maximum(threshold, 1e-30), tf.zeros_like(href)
    )
    fraction = tf.where(thk > 0.0, tf.ones_like(thk), tf.clip_by_value(fill, 0.0, 1.0))
    return reinitialise(dx * (0.5 - fraction), dx, iterations)


# ---------------------------------------------------------------------------
# Mass
# ---------------------------------------------------------------------------


@tf.function(jit_compile=True)
def mass_step(
    thk: tf.Tensor,
    href: tf.Tensor,
    psi_adv: tf.Tensor,
    psi: tf.Tensor,
    topg: tf.Tensor,
    water_level: tf.Tensor,
    dx: tf.Tensor,
    rho_ratio: float,
    min_thickness: float,
    speed: Optional[tf.Tensor] = None,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Make the ice follow the level set; return ``(thk, Href, psi, calved, fraction)``.

    ``calved`` (m) is the ice removed from each cell; the total of
    ``thk + Href`` changes by exactly ``-sum(calved)``.
    """
    marine = topg < water_level
    width = ramp_width(psi_adv, dx)
    phi = fill_fraction(psi, dx, width)
    # Areas the ice has flowed over and still covers, not clipped at the
    # cell edge: a partial cell does not pass ice on, so its column is
    # Href over the advected area, and calving keeps the covered part.
    area_adv = tf.maximum(0.5 - psi_adv / width, 0.0)
    area = tf.maximum(0.5 - psi / width, 0.0)

    # Calving: ice leaves with the area the front leaves (column kept). Only
    # the ablation part demotes a full cell: the ratio phi/phi_adv is the
    # covered area left after the calving alone, so a trailing edge (ice
    # advected away from an ice-free cell), already thinned to the correct
    # cell mean by the transport, is not thinned a second time.
    before = thk + href
    ice = thk > 0.0
    keep = tf.where(
        area_adv > 0.0, tf.minimum(area / tf.maximum(area_adv, 1e-30), 1.0), 0.0 * phi
    )
    href = tf.where(marine & ~ice, href * keep, href)
    phi_adv = fill_fraction(psi_adv, dx, width)
    ratio = tf.minimum(phi / tf.maximum(phi_adv, 1e-30), 1.0)
    entered = marine & ice & (ratio < FULL)
    href = tf.where(entered, thk * ratio, href)
    thk = tf.where(entered, tf.zeros_like(thk), thk)
    calved = tf.maximum(before - (thk + href), 0.0)

    # Advance: a partial cell full by area (or already by volume) becomes ice
    # at the threshold thickness; the rest goes to the neighbouring cells the
    # front has entered, or else to the ice-free ocean around it.
    threshold = threshold_thickness(thk, topg, water_level, rho_ratio, speed)
    by_volume = (threshold > 0.0) & (href >= threshold)
    full = marine & (thk <= 0.0) & (href > 0.0) & ((phi >= FULL) | by_volume)
    target = tf.where(threshold > 0.0, tf.minimum(href, threshold), href)
    thk = tf.where(full, target, thk)
    residual = tf.where(full, href - target, tf.zeros_like(href))
    href = tf.where(full, tf.zeros_like(href), href)
    ocean = marine & (thk <= 0.0)
    entered_count = count_neighbours(ocean & (phi > 0.0), thk.dtype)
    receiver = tf.where(entered_count > 0.0, ocean & (phi > 0.0), ocean)
    count = count_neighbours(receiver, thk.dtype)
    share = tf.where(
        count > 0.0, residual / tf.maximum(count, 1.0), tf.zeros_like(residual)
    )
    # Only the receivers collect, so each donor pays out exactly its
    # residual: its share to each of the receiver neighbours it was
    # divided over (an ocean-wide gather would pay out more).
    gained = tf.where(receiver, neighbour_sum(share), tf.zeros_like(href))
    href = href + gained
    thk = thk + tf.where(count > 0.0, tf.zeros_like(thk), residual)

    # Rules and clean-up.
    before = thk + href
    thk = apply_min_thickness(thk, topg, water_level, rho_ratio, min_thickness)
    thk, href = clean_up(thk, href, marine)
    removed = before - (thk + href)
    calved += tf.maximum(removed, 0.0)

    # Keep psi consistent with the ice: inside every full cell, outside every
    # cell without ice or reservoir (emptied by calving or the rules, or on
    # land, where the margin moves with the transport). A cell that received
    # a residual ahead of the level set is covered up to its fill fraction.
    ice = thk > 0.0
    psi = constrain(psi, thk, href, dx, width)
    threshold = threshold_thickness(thk, topg, water_level, rho_ratio, speed)
    fill = tf.where(
        threshold > 0.0, href / tf.maximum(threshold, 1e-30), tf.zeros_like(href)
    )
    ahead = (gained > 0.0) & ~ice & (href > 0.0)
    psi = tf.where(
        ahead, tf.minimum(psi, width * (0.5 - tf.clip_by_value(fill, 0.0, 1.0))), psi
    )

    # The area ratios above share psi_adv's width (a consistent pair); the
    # published fraction is recomputed from the final, constrained psi.
    fraction = tf.where(
        ice,
        tf.ones_like(thk),
        tf.where(href > 0.0, fill_fraction(psi, dx), tf.zeros_like(thk)),
    )
    return thk, href, psi, calved, fraction
