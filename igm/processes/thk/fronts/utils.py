#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Operations shared by the calving-front backends.

Both fronts represent the terminus as partial cells and reconstruct them from
their fully-iced surroundings, so they share the front-band flux blend, the
marine calving gate, the iceflow-facing thickness padding, and the underlying
4-neighbour reductions and mask morphology. Only the way each backend decides
which cells are partial differs, so that decision stays in the backends and is
passed in as a mask.
"""

import tensorflow as tf

from igm.utils.grad.compute_divflux_slope_limiter import compute_divflux_slope_limiter


def advect_thickness_standard(cfg, state):
    """Stock IGM mass-conservation update on thk."""
    p = cfg.processes.thk
    state.divflux = compute_divflux_slope_limiter(
        state.ubar,
        state.vbar,
        state.thk,
        state.dx,
        state.dx,
        state.dt,
        slope_type=p.slope_type,
    )
    if not hasattr(state, "smb"):
        state.smb = tf.zeros_like(state.thk)
    state.thk = tf.maximum(state.thk + state.dt * (state.smb - state.divflux), 0.0)


def blended_divflux(cfg, state, field, bulk_ice_mask):
    """Flux divergence of `field`, limited differently at the front and bulk.

    One limiter cannot do both jobs: `front_slope_type` (godunov) avoids
    pile-up at the margin, while `interior_slope_type` (superbee) is less
    diffusive in the deep interior. Setting the two equal disables the blend
    and costs one flux evaluation instead of two.
    """
    p = cfg.processes.thk
    divflux_front = compute_divflux_slope_limiter(
        state.ubar,
        state.vbar,
        field,
        state.dx,
        state.dx,
        state.dt,
        slope_type=p.front_slope_type,
    )
    if p.interior_slope_type == p.front_slope_type:
        return divflux_front

    divflux_interior = compute_divflux_slope_limiter(
        state.ubar,
        state.vbar,
        field,
        state.dx,
        state.dx,
        state.dt,
        slope_type=p.interior_slope_type,
    )
    bulk = bulk_mask_5x5(bulk_ice_mask, field.dtype)
    return bulk * divflux_interior + (1.0 - bulk) * divflux_front


def marine_calving_rate(cfg, state, dtype):
    """`state.calving_rate`, zeroed on land when `only_marine` is set."""
    rate = tf.cast(state.calving_rate, dtype)
    if not cfg.processes.thk.only_marine:
        return rate
    if hasattr(state, "water_level"):
        return rate * tf.cast(state.topg < state.water_level, dtype)
    # only_marine was requested but there is no sea level to define "marine".
    return tf.zeros_like(rate)


def extend_thk_for_iceflow(cfg, thk, partial_mask, full_mask):
    """Pad partial cells up to the mean thk of their fully-iced 4-neighbours.

    The iceflow solver reads a continuous thickness field, so the terminus
    cannot be left as a step function. With `extend_halo` > 0 the padding also
    covers a halo of recently promoted cells still thinner than
    `extend_thresh` times that mean. Callers keep the true step-function
    column in `state.thk_true`.
    """
    thk_ref = neighbour_mean(thk, full_mask)

    extend_mask = partial_mask
    halo_steps = int(cfg.processes.thk.extend_halo)
    if halo_steps > 0:
        thresh = tf.cast(cfg.processes.thk.extend_thresh, thk.dtype)
        halo_band = dilate_bool(partial_mask, halo_steps)
        halo_lo = (
            halo_band
            & tf.logical_not(partial_mask)
            & full_mask
            & (thk < thk_ref * thresh)
        )
        extend_mask = tf.logical_or(partial_mask, halo_lo)

    return tf.where(extend_mask & (thk_ref > 0.0), thk_ref, thk)


# ---------------------------------------------------------------------------
# 4-neighbour reductions and mask morphology
# ---------------------------------------------------------------------------


def neighbour_bool_any(mask):
    """True at a cell when any 4-neighbour has mask = True."""
    m = tf.cast(mask, tf.float32)
    p = tf.pad(m, [[1, 1], [1, 1]], mode="SYMMETRIC")
    s = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    return s > 0.0


def neighbour_mean(f, mask):
    """Mean of f over the 4-neighbour subset where mask is True.
    Returns 0 at cells whose neighbours all have mask = False."""
    m = tf.cast(mask, f.dtype)
    fp = tf.pad(f, [[1, 1], [1, 1]], mode="SYMMETRIC")
    mp = tf.pad(m, [[1, 1], [1, 1]], mode="SYMMETRIC")
    s = (
        fp[:-2, 1:-1] * mp[:-2, 1:-1]
        + fp[2:, 1:-1] * mp[2:, 1:-1]
        + fp[1:-1, :-2] * mp[1:-1, :-2]
        + fp[1:-1, 2:] * mp[1:-1, 2:]
    )
    n = mp[:-2, 1:-1] + mp[2:, 1:-1] + mp[1:-1, :-2] + mp[1:-1, 2:]
    return tf.where(n > 0.0, s / tf.maximum(n, 1.0), tf.zeros_like(s))


def dilate_bool(mask, steps):
    """4-connectivity dilation of a boolean mask by `steps` iterations."""
    m = tf.cast(mask, tf.float32)
    for _ in range(int(steps)):
        mp = tf.pad(m, [[1, 1], [1, 1]], mode="SYMMETRIC")
        m = tf.maximum(m, mp[:-2, 1:-1])
        m = tf.maximum(m, mp[2:, 1:-1])
        m = tf.maximum(m, mp[1:-1, :-2])
        m = tf.maximum(m, mp[1:-1, 2:])
    return m > 0.0


def bulk_mask_5x5(is_ice, dtype):
    """Deep-bulk mask: cell AND every cell in its 5x5 neighbourhood is ice.
    Used to blend a high-order divflux in the interior with a low-order
    divflux at the front band."""
    full = tf.cast(is_ice, dtype)
    fp = tf.pad(full, [[2, 2], [2, 2]], mode="SYMMETRIC")
    bulk = tf.ones_like(full)
    for di in range(-2, 3):
        for dj in range(-2, 3):
            bulk = (
                bulk
                * fp[2 + di : 2 + di + full.shape[0], 2 + dj : 2 + dj + full.shape[1]]
            )
    return bulk
