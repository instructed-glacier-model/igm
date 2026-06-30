#!/usr/bin/env python3

"""

Implements the Parameterization for subgrid-scale motion of ice-shelf calving fronts
of Albrecht et al. (2011), The Cryosphere 5, 35-44;
It is a vectorized TensorFlow
rewrite of PISM's reference implementation (https://github.com/pism/pism,
Copyright (C) PISM Authors, GNU GPL v3+) with these IGM modifications:
flotation cap on the threshold thickness, href_cap_factor safety clamp,
no lateral residual redistribution, extended-thickness / halo padding
for the iceflow solver, dual slope-limiter blend at the front band, and
an Href-budget calving sink for retreat (level_set-style).

Full cells carry thk; partial cells (ocean cells with an icy 4-neighbour)
carry Href (m, area-specific volume). state.thk_true is the true
step-function column (H in full cells, 0 in partial); state.thk is
overwritten at end of step with an iceflow-facing extended version
where partial cells are padded to interior-mean thickness. Advance
(Albrecht 2011 / PISM): inflow accumulates in Href until it reaches
H_threshold (interior-neighbour mean surface, capped at flotation),
then the cell promotes to thk = H_threshold and leftover Href stays
on the cell. Retreat (level_set-style): when
state.calving_rate is set, _apply_calving drains a c * H_ref * dt / dx
budget from Href first, then optionally from the adjacent cliff thk.
See Albrecht et al. (2011), TC 5, 35-44 and PISM (https://www.pism.io/).
"""

import tensorflow as tf

from igm.utils.grad.compute_divflux_slope_limiter import compute_divflux_slope_limiter

from utils import (
    neighbour_bool_any,
    neighbour_mean,
    dilate_bool,
    bulk_mask_5x5,
)


def _ocean(state):
    is_ice = state.thk > 0.0
    if hasattr(state, "water_level"):
        return tf.logical_and(tf.logical_not(is_ice), state.topg < state.water_level)
    return tf.logical_not(is_ice)


def _partial_mask(state):
    is_ice = state.thk > 0.0
    return tf.logical_and(_ocean(state), neighbour_bool_any(is_ice))


def _threshold_thickness(cfg, state, is_ice):
    p = cfg.processes.thk
    H_avg = neighbour_mean(state.thk, is_ice)
    h_avg = neighbour_mean(state.usurf, is_ice)
    grounded = (state.topg + H_avg) > h_avg
    H_threshold = tf.where(grounded, h_avg - state.topg, H_avg)
    H_threshold = tf.maximum(H_threshold, 0.0)

    if hasattr(state, "water_level"):
        Dw = tf.maximum(state.water_level - state.topg, 0.0)
        H_float = Dw / p.ratio_density
        marine = Dw > 0.0
        H_threshold = tf.where(marine, tf.minimum(H_threshold, H_float), H_threshold)
    return H_threshold


def _advect_and_route(cfg, state, is_partial):
    p = cfg.processes.thk
    divflux_front = compute_divflux_slope_limiter(
        state.ubar,
        state.vbar,
        state.thk,
        state.dx,
        state.dx,
        state.dt,
        slope_type=p.front_slope_type,
    )
    if p.interior_slope_type == p.front_slope_type:
        state.divflux = divflux_front
    else:
        divflux_interior = compute_divflux_slope_limiter(
            state.ubar,
            state.vbar,
            state.thk,
            state.dx,
            state.dx,
            state.dt,
            slope_type=p.interior_slope_type,
        )
        bulk = bulk_mask_5x5(state.thk > 0.0, state.thk.dtype)
        state.divflux = bulk * divflux_interior + (1.0 - bulk) * divflux_front
    if not hasattr(state, "smb"):
        state.smb = tf.zeros_like(state.thk)

    partial_f = tf.cast(is_partial, state.thk.dtype)

    inflow = tf.maximum(-state.divflux, 0.0)
    state.Href.assign(state.Href + state.dt * inflow * partial_f)

    thk_delta = tf.where(
        is_partial,
        tf.zeros_like(state.thk),
        state.dt * (state.smb - state.divflux),
    )
    state.thk = tf.maximum(state.thk + thk_delta, 0.0)


def _promote_and_cap(cfg, state):
    sg = cfg.processes.thk.sub_grid
    cap_factor = tf.cast(sg.href_cap_factor, state.thk.dtype)

    for _ in range(int(sg.promote_iters)):
        is_ice = state.thk > 0.0
        is_partial = _partial_mask(state)

        H_threshold = _threshold_thickness(cfg, state, is_ice)

        href_cap = tf.where(
            is_partial & (H_threshold > 0.0),
            cap_factor * H_threshold,
            state.Href,
        )
        state.Href.assign(tf.minimum(state.Href, href_cap))

        thr = tf.where(H_threshold > 0.0, H_threshold, state.Href)

        ready = is_partial & (state.Href >= thr) & (thr > 0.0)
        if not tf.reduce_any(ready):
            break

        gained = tf.where(ready, thr, tf.zeros_like(state.thk))
        state.thk = state.thk + gained
        state.Href.assign(state.Href - gained)


def _extend_thk_for_iceflow(cfg, state):
    """Pad partial cells (thk=0, next to ice) and optionally a
    halo of recently-promoted thin cells up to the interior-mean thk, so
    the iceflow solver sees a continuous thickness field. The true
    step-function column is preserved in state.thk_true."""
    p = cfg.processes.thk
    thk = state.thk

    is_ice = thk > 0.0
    is_partial = _partial_mask(state)
    thk_ref = neighbour_mean(thk, is_ice)

    extend_mask = is_partial
    halo_steps = int(p.extend_halo)
    if halo_steps > 0:
        thresh = tf.cast(p.extend_thresh, thk.dtype)
        halo_band = dilate_bool(is_partial, halo_steps)
        halo_lo = (
            halo_band & tf.logical_not(is_partial) & is_ice & (thk < thk_ref * thresh)
        )
        extend_mask = tf.logical_or(is_partial, halo_lo)

    has_ref = thk_ref > 0.0
    return tf.where(extend_mask & has_ref, thk_ref, thk)


def _apply_calving(cfg, state):
    """Apply state.calving_rate as a mass sink on Href, then
    optionally on the adjacent cliff (calve_cliff).

    state.calving_rate is the normal front retreat speed c (m/yr). The
    area-specific volume drained from a partial cell per step is
    c * H_ref * dt / dx, where H_ref is the interior-neighbour mean
    thickness (the column the partial cell would have if it were full).
    The dx scaling converts a cliff-retreat speed into the right
    Href-units sink: a column of width c*dt at full height H_ref,
    spread over a cell of size dx. With it, the net front velocity
    becomes (flux-driven advance) - c, mirroring cf_level_set.py.
    """
    p = cfg.processes.thk
    sg = p.sub_grid
    dtype = state.thk.dtype
    dx = tf.cast(state.dx, dtype)
    c = tf.cast(state.calving_rate, dtype)

    if p.only_marine and hasattr(state, "water_level"):
        c = c * tf.cast(state.topg < state.water_level, dtype)
    elif p.only_marine:
        c = tf.zeros_like(c)

    # Interior-neighbour mean thickness: the H that appears in the c * H / dx
    # retreat-mass-flux. At partial cells state.thk = 0, so take the mean over
    # ice neighbours.
    H_ref = neighbour_mean(state.thk, state.thk > 0.0)
    budget = state.dt * c * H_ref / dx

    lose_href = tf.minimum(state.Href, budget)
    state.Href.assign(state.Href - lose_href)

    if not sg.calve_cliff:
        return

    remaining = budget - lose_href
    is_ice = state.thk > 0.0
    front_full = tf.logical_and(is_ice, neighbour_bool_any(_ocean(state)))
    drain = tf.where(
        front_full, tf.minimum(state.thk, remaining), tf.zeros_like(state.thk)
    )
    state.thk = tf.maximum(state.thk - drain, 0.0)


# ---------------------------------------------------------------------------
# Public API (called from thk.py)
# ---------------------------------------------------------------------------


def initialize(cfg, state):
    if not hasattr(state, "Href"):
        state.Href = tf.Variable(tf.zeros_like(state.thk), trainable=False, name="Href")
    if not hasattr(state, "thk_true"):
        state.thk_true = tf.Variable(state.thk, trainable=False)


def update(cfg, state):
    # Restore the true step-function thk (full cells: H, partial: 0)
    # before mass transport. state.thk may have been overwritten by the
    # iceflow-facing extended view at the end of the previous step.
    state.thk = tf.identity(state.thk_true)

    is_partial = _partial_mask(state)
    _advect_and_route(cfg, state, is_partial)
    _promote_and_cap(cfg, state)

    if hasattr(state, "calving_rate"):
        _apply_calving(cfg, state)

    state.thk_true.assign(state.thk)
    state.thk = _extend_thk_for_iceflow(cfg, state)
