#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Shared helpers for thk / cf_level_set / cf_sub_grid."""

import tensorflow as tf

from igm.utils.grad.compute_divflux_slope_limiter import compute_divflux_slope_limiter


def update_surfaces(cfg, state):
    """Lower / upper ice surfaces. Flotation when state.water_level exists,
    else lsurf = topg."""
    p = cfg.processes.thk
    if hasattr(state, "water_level"):
        state.lsurf = tf.maximum(
            state.topg,
            -p.ratio_density * state.thk + state.water_level,
        )
    else:
        state.lsurf = tf.identity(state.topg)
    state.usurf = state.lsurf + state.thk


def advect_thickness_standard(cfg, state):
    """Stock IGM mass-conservation update on thk."""
    p = cfg.processes.thk
    state.divflux = compute_divflux_slope_limiter(
        state.ubar, state.vbar, state.thk,
        state.dx, state.dx, state.dt,
        slope_type=p.slope_type,
    )
    if not hasattr(state, "smb"):
        state.smb = tf.zeros_like(state.thk)
    state.thk = tf.maximum(
        state.thk + state.dt * (state.smb - state.divflux), 0.0
    )


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
    s = (fp[:-2, 1:-1] * mp[:-2, 1:-1] + fp[2:, 1:-1] * mp[2:, 1:-1]
         + fp[1:-1, :-2] * mp[1:-1, :-2] + fp[1:-1, 2:] * mp[1:-1, 2:])
    n = (mp[:-2, 1:-1] + mp[2:, 1:-1] + mp[1:-1, :-2] + mp[1:-1, 2:])
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
            bulk = bulk * fp[2 + di : 2 + di + full.shape[0],
                             2 + dj : 2 + dj + full.shape[1]]
    return bulk
