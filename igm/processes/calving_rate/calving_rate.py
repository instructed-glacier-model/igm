#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""
calving_rate
============

Geometry- and flow-based calving-rate field, consumed by the ``thk``
user module (level_set or sub_grid calving-front mode) as the normal
front speed ``c`` (m/yr).

Each outer step this module writes a 2D field ``state.calving_rate``
according to the law selected in ``cfg.processes.calving_rate.law``:

  zero                    c = 0
  water_depth             c = k * D_w                              linear water-depth
  eigen                   c = K2 * max(e1,0) * max(e2,0)           eigen-calving
  thickness_threshold     c = c_max where thk < Hcr, else 0        Albrecht/Ritz threshold

Symbols
  D_w   = max(water_level - topg, 0)                     water depth at bed (m)
  e1,e2 = principal horizontal strain rates of (ubar, vbar), e1 >= e2  (1/yr)

The field is zeroed on land (``topg >= water_level``) and clipped to
``[0, c_max]``. Sign convention: positive = retreat.

References
  - water_depth: empirical linear fit U_c = k * D_w.
  - eigen: Levermann et al. (2012), The Cryosphere 6, 273-286.
  - thickness_threshold: simple cutoff.
"""

import tensorflow as tf


def initialize(cfg, state):
    # "eigen" needs state.ubar / state.vbar, which iceflow hasn't produced
    # yet at init time. Seed zero in that case; update() fills it in once
    # the first iceflow solve has happened.
    if cfg.processes.calving_rate.law == "zero":
        c = tf.zeros_like(state.thk, dtype=state.thk.dtype)
    elif cfg.processes.calving_rate.law == "eigen" and not (
        hasattr(state, "ubar") and hasattr(state, "vbar")
    ):
        c = tf.zeros_like(state.thk, dtype=state.thk.dtype)
    else:
        c = _compute_field(cfg, state)

    if hasattr(state, "calving_rate"):
        state.calving_rate.assign(c)
    else:
        state.calving_rate = tf.Variable(c, trainable=False)


def update(cfg, state):
    if state.it < 0:
        return
    state.calving_rate.assign(_compute_field(cfg, state))


def finalize(cfg, state):
    pass


# ---------------------------------------------------------------------------


def _central_diff(f, dx):
    """Central differences with SYMMETRIC padding; returns (df/dx, df/dy)."""
    fp = tf.pad(f, [[1, 1], [1, 1]], mode="SYMMETRIC")
    fx = (fp[1:-1, 2:] - fp[1:-1, :-2]) / (2.0 * dx)
    fy = (fp[2:, 1:-1] - fp[:-2, 1:-1]) / (2.0 * dx)
    return fx, fy


def _principal_strain_rates(u, v, dx):
    """Eigenvalues e1 >= e2 of the symmetric 2D horizontal strain-rate tensor."""
    ux, uy = _central_diff(u, dx)
    vx, vy = _central_diff(v, dx)
    exx = ux
    eyy = vy
    exy = 0.5 * (uy + vx)
    mean = 0.5 * (exx + eyy)
    dev = tf.sqrt(tf.square(0.5 * (exx - eyy)) + tf.square(exy) + 1.0e-30)
    return mean + dev, mean - dev


def _compute_field(cfg, state):
    p = cfg.processes.calving_rate
    dtype = state.thk.dtype

    topg = tf.cast(state.topg, dtype)
    wl = tf.cast(state.water_level, dtype)
    dx = tf.cast(state.dx, dtype)

    D_w = tf.maximum(wl - topg, 0.0)

    law = p.law

    if law == "zero":
        return tf.zeros_like(topg)

    if law == "water_depth":
        c = tf.cast(p.k, dtype) * D_w

    elif law == "eigen":
        u = tf.cast(state.ubar, dtype)
        v = tf.cast(state.vbar, dtype)
        e1, e2 = _principal_strain_rates(u, v, dx)
        K2 = tf.cast(p.K2, dtype)
        c = K2 * tf.maximum(e1, 0.0) * tf.maximum(e2, 0.0)

    elif law == "thickness_threshold":
        # threshold rule: cells thinner than Hcr at the calving front
        # are removed. We implement this as c = c_max where thk < Hcr,
        # which drives _apply_calving (cf_sub_grid.py) to drain the
        # cell quickly. Acts on partial cells via Href and on adjacent
        # cliff cells via thk drain (calve_cliff).
        Hcr = tf.cast(p.Hcr, dtype)
        thk_eff = tf.cast(state.thk, dtype)
        c = tf.where(thk_eff < Hcr, tf.cast(p.c_max, dtype), tf.zeros_like(topg))

    else:
        raise ValueError(
            f"processes.calving_rate.law = {law!r} is not one of "
            "'zero', 'water_depth', 'eigen', 'thickness_threshold'."
        )

    # Zero out land (topg >= water_level) regardless of the chosen law.
    c = c * tf.cast(topg < wl, dtype)

    # Safety cap.
    c = tf.clip_by_value(c, 0.0, tf.cast(p.c_max, dtype))

    return c
