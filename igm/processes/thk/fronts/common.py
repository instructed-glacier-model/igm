#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Machinery shared by the calving-front methods.

Representation (both methods)
-----------------------------
``state.thk`` is the one, physical thickness (volume per unit area) and the
field the ice flow reads. A node carries ice only once its finite-volume cell
is full, so the stress-balance front is sharp on the boundary of full Q1
cells (see :func:`igm.processes.thk.masks.iceflow_node_mask`).

``state.Href`` (m, area-specific volume, PISM's ``ice_area_specific_volume``)
is the reservoir of the *partial cells*: ice-free ocean cells next to ice.
A partial cell is invisible to the ice flow and the surfaces; it receives
the mass balance over its covered fraction (PISM gives it none), so that its
ice column changes like the full ice next to it. The total volume is
``sum(thk + Href) dx**2``. The methods differ only in how a partial cell's
fill fraction is defined: from volume (``sub_grid``, Albrecht et al., 2011)
or from a level-set function (``level_set``). Both promote a cell to full
ice when it is full and demote it when the front re-enters it.

Transport step
--------------
The flux into a partial cell must be the ice-side flux ``u H`` (Albrecht et
al., 2011, Eq. 2). The ice flow sets the velocity to zero off its active
nodes, so the velocity is first extended into the ocean cells and into
orphan ice nodes (ice nodes without an active Q1 cell) by the mean over the
neighbours that carry velocity, and fluxes out of cells next to the front
are first-order upwind. The divergence is the explicit transport's slope
limiter, with its boundaries (the options cached by
:func:`..transport.explicit.initialize`); the whole step is one compiled
kernel.

Published diagnostics: ``state.calved_thk`` (m of ice removed from each cell
during the last step by calving, frontal melt, the front rules and the
clean-up), ``state.calving_unapplied_thk`` (m of ablation that could not be
applied because the front would have retreated more than one cell), and
``state.ice_area_fraction`` (1 full, fill fraction in partial cells, 0
elsewhere).
"""

from typing import Any, Dict, Optional, Tuple

import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State
from igm.utils.math.neighbours import (
    any_neighbour,
    count_neighbours,
    dilate,
    neighbour_mean,
    neighbour_sum,
    neighbours,
)

from igm.utils.grad.compute_divflux_slope_limiter import (
    compute_divflux_slope_limiter,
    compute_divflux_slope_limiter_boundaries,
)

from ..masks import flotation_function, iceflow_node_mask
from ..sources import mass_balance
from ..transport import explicit

#: Diagnostics published by every front method.
DIAGNOSTICS = ("calved_thk", "calving_unapplied_thk", "ice_area_fraction")

#: Thickness of a partial cell once full: from the flux into it (continuity),
#: or the neighbours' mean (PISM).
THRESHOLDS = ("flux", "mean")


# ---------------------------------------------------------------------------
# Options and fields
# ---------------------------------------------------------------------------


def front_options(cfg: DictConfig) -> Dict[str, Any]:
    """Validate the static ``cfg.processes.thk.front`` options once."""
    p = cfg.processes.thk.front
    min_thickness = float(p.get("min_thickness", 0.0))
    if not min_thickness >= 0.0:
        raise ValueError("cfg.processes.thk.front.min_thickness must be >= 0.")
    threshold = str(p.get("threshold", "flux")).strip().lower()
    if threshold not in THRESHOLDS:
        raise ValueError(
            "cfg.processes.thk.front.threshold must be one of "
            f"{', '.join(THRESHOLDS)}; got {threshold!r}."
        )
    return {
        "first_order": bool(p.get("first_order", True)),
        "min_thickness": min_thickness,
        "fixed": bool(p.get("fixed", False)),
        "flux_threshold": threshold == "flux",
    }


def initialize_front(cfg: DictConfig, state: State) -> Dict[str, Any]:
    """Set up the transport, the partial-cell reservoir and the diagnostics.

    An ``Href`` read from an input file (a restart) is kept; it is otherwise
    zero. Returns the static front options, also stored in
    ``state.thk_components.component_state["front"]``.
    """
    explicit.initialize(cfg, state)
    options = front_options(cfg)
    state.thk_components.component_state["front"] = options

    thk = tf.convert_to_tensor(state.thk)
    href = getattr(state, "Href", None)
    if href is None or tuple(href.shape) != tuple(thk.shape):
        state.Href = tf.zeros_like(thk)
    else:
        state.Href = tf.cast(tf.convert_to_tensor(href), thk.dtype)
    state.calved_thk = tf.zeros_like(thk)
    state.calving_unapplied_thk = tf.zeros_like(thk)
    # A restart may bring partial cells: their fill fraction from Href.
    threshold = threshold_thickness(
        thk,
        tf.cast(state.topg, thk.dtype),
        tf.cast(state.water_level, thk.dtype),
        float(state.thk_components.rho_ratio),
        threshold_speed(state, options) if hasattr(state, "ubar") else None,
    )
    fill = tf.where(
        threshold > 0.0, state.Href / tf.maximum(threshold, 1e-30), tf.zeros_like(thk)
    )
    state.ice_area_fraction = tf.where(
        thk > 0.0, tf.ones_like(thk), tf.clip_by_value(fill, 0.0, 1.0)
    )
    if options["fixed"]:
        state.front_initial_extent = (thk > 0.0) | (state.Href > 0.0)
    return options


def ablation_rate(cfg: DictConfig, state: State) -> tf.Tensor:
    """Lateral ablation rate ``c + m_cf`` (m/yr) at the front, >= 0.

    It is the sum of ``state.calving_rate`` and ``state.frontal_melt_rate``
    published by the ``calving_rate`` process, and zero without it.
    """
    thk = state.thk
    if "calving_rate" not in cfg.processes or not hasattr(state, "calving_rate"):
        return tf.zeros_like(thk)
    rate = tf.cast(state.calving_rate, thk.dtype)
    if hasattr(state, "frontal_melt_rate"):
        rate = rate + tf.cast(state.frontal_melt_rate, thk.dtype)
    return tf.maximum(rate, 0.0)


def publish(
    state: State,
    thk: tf.Tensor,
    href: tf.Tensor,
    calved: tf.Tensor,
    unapplied: tf.Tensor,
    fraction: tf.Tensor,
) -> None:
    """Store the result of a front step on ``state``."""
    state.thk = thk
    state.Href = href
    state.calved_thk = calved
    state.calving_unapplied_thk = unapplied
    state.ice_area_fraction = fraction


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def surface(
    thk: tf.Tensor, topg: tf.Tensor, water_level: tf.Tensor, rho_ratio: float
) -> tf.Tensor:
    """Upper surface from flotation (``rho_ratio`` = water / ice density)."""
    lsurf = tf.maximum(topg, water_level - thk / rho_ratio)
    return lsurf + thk


def threshold_thickness(
    thk: tf.Tensor,
    topg: tf.Tensor,
    water_level: tf.Tensor,
    rho_ratio: float,
    speed: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    """Thickness ``H_r`` of a partial cell once full (PISM part_grid_threshold_thickness).

    From the ice-covered edge neighbours: their mean thickness ``H`` and
    surface ``h``; where the bed is above their mean ice base ``h - H``, the
    cell is filled up to the mean surface instead. Zero without ice-covered
    neighbours.

    With the ice ``speed``, ``H`` is instead the thickness continuity gives
    to the ice entering the cell: the mean over the ice neighbours of
    ``H_n |u_n| / |u_p|``, with ``|u_p|`` the speed extrapolated linearly from
    the neighbour and the cell behind it (between ``|u_n|`` and ``2 |u_n|``).
    The neighbours' plain mean (PISM) fills a front that thins downstream with
    a wall one cell thick at every advance, a first-order error that travels
    with the front (Albrecht et al., 2011, Sect. 3, correct it with the slope
    of the analytic profile); the flux form is exact for the steady spreading
    profile (u H constant) and equals the mean where the speed is uniform.
    """
    ice = thk > 0.0
    h_avg = neighbour_mean(surface(thk, topg, water_level, rho_ratio), ice)
    H_mean = neighbour_mean(thk, ice)
    if speed is None:
        H_avg = H_mean
    else:
        speed = tf.cast(speed, thk.dtype)
        sp = tf.pad(speed, [[2, 2], [2, 2]])
        hp = tf.pad(thk, [[2, 2], [2, 2]])
        far_s = [sp[:-4, 2:-2], sp[4:, 2:-2], sp[2:-2, :-4], sp[2:-2, 4:]]
        far_H = [hp[:-4, 2:-2], hp[4:, 2:-2], hp[2:-2, :-4], hp[2:-2, 4:]]
        total, count = tf.zeros_like(thk), tf.zeros_like(thk)
        for H_n, s_n, s_nn, H_nn in zip(
            neighbours(thk, 0), neighbours(speed, 0), far_s, far_H
        ):
            behind = (H_nn > 0.0) & (s_nn > 0.0)
            s_p = tf.where(
                behind, tf.clip_by_value(2.0 * s_n - s_nn, s_n, 2.0 * s_n), s_n
            )
            ratio = tf.where(s_p > 0.0, s_n / tf.maximum(s_p, 1e-30), tf.ones_like(s_n))
            weight = tf.cast(H_n > 0.0, thk.dtype)
            total += weight * H_n * ratio
            count += weight
        H_avg = tf.where(
            count > 0.0, total / tf.maximum(count, 1.0), tf.zeros_like(thk)
        )
    H_thr = tf.where(topg + H_mean > h_avg, h_avg - topg, H_avg)
    return tf.where(any_neighbour(ice), tf.maximum(H_thr, 0.0), tf.zeros_like(thk))


def threshold_speed(state: State, options: Dict[str, Any]) -> Optional[tf.Tensor]:
    """The ice speed for the flux threshold, or ``None`` for PISM's mean."""
    if not options["flux_threshold"]:
        return None
    return tf.sqrt(
        tf.square(tf.cast(state.ubar, state.thk.dtype))
        + tf.square(tf.cast(state.vbar, state.thk.dtype))
    )


def floating(
    thk: tf.Tensor, topg: tf.Tensor, water_level: tf.Tensor, rho_ratio: float
) -> tf.Tensor:
    """Ice-covered cells at or below flotation (same tolerance as everywhere)."""
    return (thk > 0.0) & (flotation_function(thk, topg, water_level, rho_ratio) <= 0.0)


def front_faces(ice: tf.Tensor, fraction: tf.Tensor) -> tf.Tensor:
    """Front length in an ice-free cell next to ice, in units of dx.

    The front meets the ice across the cell's faces with its ice-covered
    edge neighbours; with ``n`` the front normal, its length is
    ``sum |n . e_face|`` over those faces, which is also the factor of the
    ice-side flux ``u H`` into the cell for ice flowing normal to the front.
    A front retreating at the rate ``a`` then loses ``a H dt`` per unit length
    whatever its orientation (PISM counts one face per cell, which is exact
    only for a front along a grid axis). ``n`` is the Sobel gradient of the
    ice fraction ``fraction`` (1 in full cells, the fill fraction in partial
    cells), which averages the grid staircase over three cells.
    """
    f = tf.pad(fraction, [[1, 1], [1, 1]], mode="SYMMETRIC")
    gx = (f[:-2, 2:] + 2.0 * f[1:-1, 2:] + f[2:, 2:]) - (
        f[:-2, :-2] + 2.0 * f[1:-1, :-2] + f[2:, :-2]
    )
    gy = (f[2:, :-2] + 2.0 * f[2:, 1:-1] + f[2:, 2:]) - (
        f[:-2, :-2] + 2.0 * f[:-2, 1:-1] + f[:-2, 2:]
    )
    norm = tf.sqrt(gx * gx + gy * gy)
    nx = tf.abs(gx) / tf.maximum(norm, 1e-30)
    ny = tf.abs(gy) / tf.maximum(norm, 1e-30)
    south, north, west, east = (
        tf.cast(n, fraction.dtype) for n in neighbours(ice, False)
    )
    oriented = (west + east) * nx + (south + north) * ny
    # Fall back to the face count when the oriented length vanishes although
    # ice faces exist (a flat fraction, or a one-cell strait whose normal is
    # orthogonal to every ice face): such a cell would otherwise never drain.
    return tf.where(oriented > 1e-3, oriented, west + east + south + north)


def _extrapolated_mean(f: tf.Tensor, known: tf.Tensor) -> tf.Tensor:
    """Mean over the known edge neighbours of ``f`` extrapolated linearly.

    Each known neighbour ``n`` contributes ``2 f_n - f_nn``, with ``nn`` the
    cell behind it, when ``nn`` is known and the value lies between ``f_n``
    and ``2 f_n``; otherwise ``f_n``.
    """
    weight = tf.cast(known, f.dtype)
    fp = tf.pad(f, [[2, 2], [2, 2]])
    wp = tf.pad(weight, [[2, 2], [2, 2]])
    far = [fp[:-4, 2:-2], fp[4:, 2:-2], fp[2:-2, :-4], fp[2:-2, 4:]]
    far_w = [wp[:-4, 2:-2], wp[4:, 2:-2], wp[2:-2, :-4], wp[2:-2, 4:]]
    total, count = tf.zeros_like(f), tf.zeros_like(f)
    for n, nw, ff, fw in zip(neighbours(f, 0), neighbours(weight, 0), far, far_w):
        linear = 2.0 * n - ff
        low, high = tf.minimum(n, 2.0 * n), tf.maximum(n, 2.0 * n)
        value = tf.where(fw > 0.0, tf.clip_by_value(linear, low, high), n)
        total += nw * value
        count += nw
    return tf.where(count > 0.0, total / tf.maximum(count, 1.0), tf.zeros_like(f))


def extend_velocity(
    ubar: tf.Tensor,
    vbar: tf.Tensor,
    known: tf.Tensor,
    targets: tf.Tensor,
    steps: int,
    linear: bool = False,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Extend ``(ubar, vbar)`` from ``known`` cells into ``targets``.

    Each step fills the targets next to a known cell with the mean over
    their known edge neighbours (PISM's margin extrapolation), then counts
    them as known; other cells are unchanged. With the first-order flux out
    of a front cell, the flux into the partial cell is then the front cell's
    own ``u H``, which behind an advancing front equals the flux through the
    shelf. With ``linear``, the first ring is extrapolated linearly instead
    (between once and twice the neighbour's velocity): the speed of the ice
    at the front of a spreading shelf, which moves the level set.
    """
    for step in range(int(steps)):
        new = targets & ~known & any_neighbour(known)
        if linear and step == 0:
            ubar = tf.where(new, _extrapolated_mean(ubar, known), ubar)
            vbar = tf.where(new, _extrapolated_mean(vbar, known), vbar)
        else:
            ubar = tf.where(new, neighbour_mean(ubar, known), ubar)
            vbar = tf.where(new, neighbour_mean(vbar, known), vbar)
        known = known | new
    return ubar, vbar


# ---------------------------------------------------------------------------
# Transport step
# ---------------------------------------------------------------------------


def transport(state: State, extension_steps: int = 2) -> Tuple[tf.Tensor, tf.Tensor]:
    """Advance full cells and fill the partial-cell reservoir by one step.

    Returns the new full-cell thickness and reservoir ``(thk, Href)`` and sets
    ``state.divflux``. The velocity is extended ``extension_steps`` rings
    into the ocean; two are needed: orphan ice nodes (without an active Q1
    cell) take the velocity of their neighbours, and the ice-free cells next
    to them take theirs in turn, so that no face out of the ice carries half
    its velocity. One compiled kernel does the whole step.
    """
    components = state.thk_components
    options = components.component_state["front"]
    boundary = components.transport_options
    dtype = state.thk.dtype
    new_thk, new_href, state.divflux = _transport_kernel(
        tf.convert_to_tensor(state.thk),
        tf.convert_to_tensor(state.Href),
        tf.cast(state.topg, dtype),
        tf.cast(state.water_level, dtype),
        tf.cast(state.usurf, dtype),
        tf.cast(state.ubar, dtype),
        tf.cast(state.vbar, dtype),
        tf.cast(mass_balance(state), dtype),
        tf.cast(state.ice_area_fraction, dtype),
        tf.cast(state.dt, dtype),
        tf.cast(state.dx, dtype),
        boundary["left_ghost"],
        boundary["right_ghost"],
        boundary["top_ghost"],
        boundary["bottom_ghost"],
        float(components.rho_ratio),
        max(1, int(extension_steps)),
        bool(options["first_order"]),
        str(boundary["slope_type"]),
        bool(boundary["has_boundary_condition"]),
        (
            bool(boundary["left_symmetric"]),
            bool(boundary["right_symmetric"]),
            bool(boundary["top_symmetric"]),
            bool(boundary["bottom_symmetric"]),
        ),
    )
    return new_thk, new_href


@tf.function(jit_compile=True)
def _transport_kernel(
    thk: tf.Tensor,
    href: tf.Tensor,
    topg: tf.Tensor,
    water_level: tf.Tensor,
    usurf: tf.Tensor,
    ubar: tf.Tensor,
    vbar: tf.Tensor,
    source: tf.Tensor,
    fraction: tf.Tensor,
    dt: tf.Tensor,
    dx: tf.Tensor,
    left_ghost: Optional[tf.Tensor],
    right_ghost: Optional[tf.Tensor],
    top_ghost: Optional[tf.Tensor],
    bottom_ghost: Optional[tf.Tensor],
    rho_ratio: float,
    extension_steps: int,
    first_order: bool,
    slope_type: str,
    has_boundary_condition: bool,
    symmetric: Tuple[bool, bool, bool, bool],
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Velocity extension, divergence (explicit scheme) and routing, fused."""
    marine = topg < water_level
    ice = thk > 0.0
    ocean = ~ice & marine
    known = iceflow_node_mask(thk, usurf, water_level, rho_ratio)
    ubar, vbar = extend_velocity(ubar, vbar, known, ocean | ice, extension_steps)
    mask = dilate(ocean, 1) if first_order else None
    if has_boundary_condition:
        left, right, top, bottom = symmetric
        divflux = compute_divflux_slope_limiter_boundaries(
            ubar,
            vbar,
            thk,
            dx,
            dx,
            dt,
            slope_type=slope_type,
            left=left,
            right=right,
            top=top,
            bottom=bottom,
            first_order_mask=mask,
            left_ghost=left_ghost,
            right_ghost=right_ghost,
            top_ghost=top_ghost,
            bottom_ghost=bottom_ghost,
        )
    else:
        divflux = compute_divflux_slope_limiter(
            ubar, vbar, thk, dx, dx, dt, slope_type=slope_type, first_order_mask=mask
        )
    new_thk, new_href = _route(thk, href, divflux, source, fraction, dt, marine)
    return new_thk, new_href, divflux


@tf.function(jit_compile=True)
def _route(
    thk: tf.Tensor,
    href: tf.Tensor,
    divflux: tf.Tensor,
    source: tf.Tensor,
    fraction: tf.Tensor,
    dt: tf.Tensor,
    marine: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Full and land cells take ``dt (source - div)``; partial cells store the inflow.

    A partial cell also takes the source over its covered ``fraction``. Ice-free
    ocean cells never gain thickness otherwise: advected ice goes into the
    reservoir of a partial cell, and no ice forms from a surface balance over
    the open ocean.

    Two known, deliberate approximations: the non-negativity clips lose the
    part of a sink (melt) that exceeds the ice present, uncounted in
    ``calved_thk`` (the budget is exact without melt); and inflow through a
    Dirichlet ghost into an ice-free marine edge cell is dropped, since the
    ghost is not a neighbour (the front would have to retreat onto the
    Dirichlet side first).
    """
    ice = thk > 0.0
    ocean = ~ice & marine
    partial = ocean & any_neighbour(ice)
    advanced = tf.maximum(thk + dt * (source - divflux), 0.0)
    new_thk = tf.where(ocean, tf.zeros_like(thk), advanced)
    gain = dt * (
        tf.maximum(-divflux, 0.0)
        + source * tf.where(href > 0.0, fraction, 0.0 * fraction)
    )
    new_href = tf.where(partial, tf.maximum(href + gain, 0.0), href)
    return new_thk, new_href


# ---------------------------------------------------------------------------
# Advance: partial cells that fill
# ---------------------------------------------------------------------------


def _promote(
    thk: tf.Tensor,
    href: tf.Tensor,
    topg: tf.Tensor,
    water_level: tf.Tensor,
    rho_ratio: float,
    speed: Optional[tf.Tensor],
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Turn every full partial cell into ice at ``H_r``; return the residual."""
    threshold = threshold_thickness(thk, topg, water_level, rho_ratio, speed)
    # A zero threshold (the neighbours' surface below the bed) takes the
    # whole reservoir (PISM). Only cells next to ice fill: a stranded
    # reservoir is calved by the clean-up.
    threshold = tf.where(threshold > 0.0, threshold, href)
    ready = (thk <= 0.0) & (href > 0.0) & (href >= threshold) & any_neighbour(thk > 0.0)
    thk = tf.where(ready, threshold, thk)
    residual = tf.where(ready, href - threshold, tf.zeros_like(href))
    href = tf.where(ready, tf.zeros_like(href), href)
    return thk, href, residual


def _redistribute(
    thk: tf.Tensor, href: tf.Tensor, residual: tf.Tensor, marine: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Split each residual equally among the ice-free ocean edge neighbours.

    A cell without such a neighbour keeps its residual as thickness.
    """
    ocean = (thk <= 0.0) & marine
    count = count_neighbours(ocean, thk.dtype)
    share = tf.where(
        count > 0.0, residual / tf.maximum(count, 1.0), tf.zeros_like(residual)
    )
    href = href + tf.where(ocean, neighbour_sum(share), tf.zeros_like(href))
    thk = thk + tf.where(count > 0.0, tf.zeros_like(thk), residual)
    return thk, href


def fill_partial_cells(
    thk: tf.Tensor,
    href: tf.Tensor,
    topg: tf.Tensor,
    water_level: tf.Tensor,
    rho_ratio: float,
    speed: Optional[tf.Tensor],
    max_iterations: int,
    redistribute: bool,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Advance the front: partial cells with ``Href >= H_r`` become ice.

    A filled cell gets ``thk = H_r`` (:func:`threshold_thickness`); its residual
    ``Href - H_r`` is split equally among its ice-free ocean neighbours, which
    may fill in turn, for up to ``max_iterations`` passes of a device-side
    loop (PISM part_grid); what is left stays in place. Without
    ``redistribute`` the residual is returned as discarded (Albrecht et al.,
    2011, variant 1). Returns ``(thk, Href, discarded)``.
    """
    marine = topg < water_level
    thk, href, residual = _promote(thk, href, topg, water_level, rho_ratio, speed)
    if not redistribute:
        return thk, href, residual

    def more(
        i: tf.Tensor, thk: tf.Tensor, href: tf.Tensor, residual: tf.Tensor
    ) -> tf.Tensor:
        return (i < max_iterations) & tf.reduce_any(residual > 0.0)

    def one_pass(
        i: tf.Tensor, thk: tf.Tensor, href: tf.Tensor, residual: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        thk, href = _redistribute(thk, href, residual, marine)
        thk, href, residual = _promote(thk, href, topg, water_level, rho_ratio, speed)
        return i + 1, thk, href, residual

    _, thk, href, residual = tf.while_loop(
        more, one_pass, (tf.constant(0), thk, href, residual)
    )
    return thk + residual, href, tf.zeros_like(residual)


# ---------------------------------------------------------------------------
# Front rules and clean-up
# ---------------------------------------------------------------------------


def apply_min_thickness(
    thk: tf.Tensor,
    topg: tf.Tensor,
    water_level: tf.Tensor,
    rho_ratio: float,
    min_thickness: float,
) -> tf.Tensor:
    """Calve floating front ice thinner than ``min_thickness``.

    The thickness rule of Albrecht et al. (2011) and PISM's
    ``thickness_calving``: a floating full cell next to ice-free ocean is
    removed when its thickness is below the threshold. The rule is applied
    until the front is at least ``min_thickness`` thick, so that a cell that
    fills at the front in the same step cannot shield a thinner one behind
    it (PISM applies it once per step, which its slower filling allows).
    Any ice-free marine cell counts as ocean, holes and rifts included, as in
    PISM (``next_to_ice_free_ocean``): the ``ocean_connected_only`` option of
    the ``calving_rate`` process does not reach this rule.
    Returns the new thickness; a threshold of 0 is a no-op.
    """
    if min_thickness <= 0.0:
        return thk
    marine = topg < water_level

    def calve(thk: tf.Tensor, _: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        ocean = (thk <= 0.0) & marine
        front = floating(thk, topg, water_level, rho_ratio) & any_neighbour(ocean)
        remove = front & (thk < min_thickness)
        return tf.where(remove, tf.zeros_like(thk), thk), tf.reduce_any(remove)

    thk, _ = tf.while_loop(
        lambda thk, more: more,
        calve,
        (thk, tf.constant(True)),
        maximum_iterations=tf.size(thk),
    )
    return thk


def clean_up(
    thk: tf.Tensor, href: tf.Tensor, marine: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Keep the reservoir consistent (PISM ``ensure_consistency`` and icebergs).

    Where a cell holds both ice and ``Href``, or is no longer below the water
    level, its ``Href`` becomes thickness; ``Href`` on a cell no longer next
    to ice is removed (calved).
    """
    ice = thk > 0.0
    merge = (href > 0.0) & (ice | ~marine)
    thk = tf.where(merge, thk + href, thk)
    href = tf.where(merge, tf.zeros_like(href), href)
    orphan = ~any_neighbour(thk > 0.0)
    href = tf.where(orphan, tf.zeros_like(href), href)
    return thk, href
