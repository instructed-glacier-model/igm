#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors


"""Level-set calving-front scheme of Bondzio et al. (2016), The Cryosphere
10, 497-510, as implemented in ISSM."""

import tensorflow as tf

from .utils import (
    advect_thickness_standard,
    blended_divflux,
    extend_thk_for_iceflow,
    marine_calving_rate,
    neighbour_bool_any,
    neighbour_mean,
)

# ---------------------------------------------------------------------------
# Level-set kernels
# ---------------------------------------------------------------------------


def _pad_sym(f):
    return tf.pad(f, [[1, 1], [1, 1]], mode="SYMMETRIC")


def _forward_backward_diffs(psi, dx):
    p = _pad_sym(psi)
    Dxm = (p[1:-1, 1:-1] - p[1:-1, :-2]) / dx
    Dxp = (p[1:-1, 2:] - p[1:-1, 1:-1]) / dx
    Dym = (p[1:-1, 1:-1] - p[:-2, 1:-1]) / dx
    Dyp = (p[2:, 1:-1] - p[1:-1, 1:-1]) / dx
    return Dxm, Dxp, Dym, Dyp


def _godunov_grad_mag(Dxm, Dxp, Dym, Dyp, sign):
    pos = tf.cast(sign > 0.0, Dxm.dtype)
    neg = 1.0 - pos
    g_plus = tf.sqrt(
        tf.square(tf.maximum(Dxm, 0.0))
        + tf.square(tf.minimum(Dxp, 0.0))
        + tf.square(tf.maximum(Dym, 0.0))
        + tf.square(tf.minimum(Dyp, 0.0))
        + 1.0e-20
    )
    g_minus = tf.sqrt(
        tf.square(tf.minimum(Dxm, 0.0))
        + tf.square(tf.maximum(Dxp, 0.0))
        + tf.square(tf.minimum(Dym, 0.0))
        + tf.square(tf.maximum(Dyp, 0.0))
        + 1.0e-20
    )
    return pos * g_plus + neg * g_minus


def _reinitialise(psi, s0, dx, n_iter):
    dtau = 0.5 * dx
    for _ in range(int(n_iter)):
        Dxm, Dxp, Dym, Dyp = _forward_backward_diffs(psi, dx)
        grad_mag = _godunov_grad_mag(Dxm, Dxp, Dym, Dyp, s0)
        psi = psi - dtau * s0 * (grad_mag - 1.0)
    return psi


def _compute_phi(state):
    """Ice-area fraction per cell: phi = clip(-psi/dx + 0.5, 0, 1)."""
    return tf.clip_by_value(-state.psi / state.dx + 0.5, 0.0, 1.0)


def _build_initial_psi(cfg, state):
    """Build state.psi as a signed-distance field consistent with thk > 0."""
    ls = cfg.processes.thk.level_set
    if hasattr(state, "psi"):
        state.psi = tf.Variable(tf.cast(state.psi, state.thk.dtype), trainable=False)
        return

    mask_inside = state.thk > 0.0
    dx_f = tf.cast(state.dx, tf.float32)
    ones = tf.ones_like(mask_inside, dtype=tf.float32)
    psi0 = tf.where(mask_inside, -dx_f * ones, dx_f * ones)
    s0 = tf.sign(psi0)
    psi0 = _reinitialise(psi0, s0, state.dx, ls.reinit_iter_initial)
    psi0 = tf.where(mask_inside, -tf.abs(psi0), tf.abs(psi0))
    state.psi = tf.Variable(psi0, trainable=False)


def _advect_psi(cfg, state):
    """HJ / Godunov step for d psi / dt + v . grad psi + F |grad psi| = 0,
    with F = -c (c = state.calving_rate; c > 0 retreat, c < 0 advance)."""
    p = cfg.processes.thk
    ls = p.level_set
    dx = state.dx
    dt = state.dt
    dtype = state.thk.dtype

    u = tf.cast(state.ubar, dtype)
    v = tf.cast(state.vbar, dtype)

    Dxm, Dxp, Dym, Dyp = _forward_backward_diffs(state.psi, dx)
    adv_x = tf.maximum(u, 0.0) * Dxm + tf.minimum(u, 0.0) * Dxp
    adv_y = tf.maximum(v, 0.0) * Dym + tf.minimum(v, 0.0) * Dyp

    # Both restrictions are multiplicative factors on the calving rate, so
    # their order does not matter.
    c = marine_calving_rate(cfg, state, dtype)
    if ls.only_near_front:
        band = tf.cast(ls.band_cells, dtype) * dx
        c = c * tf.exp(-((state.psi / band) ** 2))

    grad_plus = tf.sqrt(
        tf.square(tf.maximum(Dxm, 0.0))
        + tf.square(tf.minimum(Dxp, 0.0))
        + tf.square(tf.maximum(Dym, 0.0))
        + tf.square(tf.minimum(Dyp, 0.0))
        + 1.0e-20
    )
    grad_minus = tf.sqrt(
        tf.square(tf.minimum(Dxm, 0.0))
        + tf.square(tf.maximum(Dxp, 0.0))
        + tf.square(tf.minimum(Dym, 0.0))
        + tf.square(tf.maximum(Dyp, 0.0))
        + 1.0e-20
    )
    calving_term = tf.maximum(c, 0.0) * grad_minus + tf.minimum(c, 0.0) * grad_plus

    state.psi.assign(state.psi - dt * (adv_x + adv_y) + dt * calving_term)


def _advect_mass(cfg, state, phi_old, phi_new):
    """Fractional-area continuity on M = thk * phi."""
    p = cfg.processes.thk
    ls = p.level_set
    dtype = state.thk.dtype
    eps = tf.cast(ls.phi_eps, dtype)

    M_old = state.thk * phi_old

    state.divflux = blended_divflux(cfg, state, M_old, phi_old > (1.0 - eps))
    if not hasattr(state, "smb"):
        state.smb = tf.zeros_like(state.thk)

    M_new = tf.maximum(M_old + state.dt * (state.smb * phi_old - state.divflux), 0.0)

    thk_new = tf.where(
        phi_new > eps, M_new / tf.maximum(phi_new, eps), tf.zeros_like(M_new)
    )

    if ls.cap_partial:
        partial_mask, full_mask = _partial_and_full_masks(cfg, phi_new, thk_new)
        thk_cap = neighbour_mean(thk_new, full_mask)
        has_ref = neighbour_bool_any(full_mask)
        thk_capped = tf.where(
            partial_mask & has_ref,
            tf.minimum(thk_new, thk_cap),
            thk_new,
        )
        dx = tf.cast(state.dx, dtype)
        discarded = tf.reduce_sum((thk_new - thk_capped) * phi_new * dx * dx)
        state.cf_capped_volume.assign_add(discarded)
        state.cf_capped_volume_step.assign(discarded)
        thk_new = thk_capped

    state.thk = thk_new


def _partial_and_full_masks(cfg, phi, thk):
    """Split cells by ice-area fraction: partially and fully covered."""
    eps = tf.cast(cfg.processes.thk.level_set.phi_eps, thk.dtype)
    return phi < (1.0 - eps), (phi >= (1.0 - eps)) & (thk > 0.0)


def _reinit_maybe(cfg, state):
    ls = cfg.processes.thk.level_set
    components = state.thk_components
    components.steps_since_reinit += 1
    if ls.reinit_freq > 0 and components.steps_since_reinit >= int(ls.reinit_freq):
        s0 = state.psi / tf.sqrt(state.psi * state.psi + state.dx * state.dx)
        state.psi.assign(_reinitialise(state.psi, s0, state.dx, ls.reinit_iter))
        components.steps_since_reinit = 0


# ---------------------------------------------------------------------------
# Public API (called from thk.py)
# ---------------------------------------------------------------------------


def initialize(cfg, state):
    components = state.thk_components
    components.steps_since_reinit = 0
    components.psi_built = False
    if hasattr(state, "calving_rate"):
        _build_initial_psi(cfg, state)
        components.psi_built = True

    if not hasattr(state, "cf_capped_volume"):
        state.cf_capped_volume = tf.Variable(0.0, trainable=False)
    if not hasattr(state, "cf_capped_volume_step"):
        state.cf_capped_volume_step = tf.Variable(0.0, trainable=False)

    if not hasattr(state, "thk_true"):
        state.thk_true = tf.Variable(state.thk, trainable=False)


def update(cfg, state):
    if not hasattr(state, "calving_rate"):
        # No calving_rate on state -> plain thk update.
        advect_thickness_standard(cfg, state)
        return

    if not state.thk_components.psi_built:
        _build_initial_psi(cfg, state)
        state.thk_components.psi_built = True

    # Recover the true (M/phi) thk from last step; state.thk may have been
    # overwritten with the extended version for iceflow.
    state.thk = tf.identity(state.thk_true)

    phi_old = _compute_phi(state)
    _advect_psi(cfg, state)
    phi_new = _compute_phi(state)
    _advect_mass(cfg, state, phi_old, phi_new)
    _reinit_maybe(cfg, state)

    state.thk_true.assign(state.thk)
    partial_mask, full_mask = _partial_and_full_masks(cfg, phi_new, state.thk)
    state.thk = extend_thk_for_iceflow(cfg, state.thk, partial_mask, full_mask)
