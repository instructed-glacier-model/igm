#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State
from igm.utils.grad.compute_divflux_slope_limiter import compute_divflux_slope_limiter
from igm.utils.stag.stag import stag2x, stag2y
from igm.utils.grad.grad import pad_x, pad_y


def _get_sealevel(cfg: DictConfig, state: State) -> tf.Tensor:
    """Get sea level from state or use default from config."""
    return getattr(state, "sealevel", cfg.processes.thk.default_sealevel)


def _get_smb(cfg: DictConfig, state: State) -> tf.Tensor:
    """Get surface mass balance from state or use zero tensor."""
    return getattr(state, "smb", tf.zeros_like(state.thk))


def _compute_surfs(cfg: DictConfig, state: State) -> None:
    """Update lower and upper ice surfaces."""
    delta = cfg.processes.thk.ratio_density
    sealevel = _get_sealevel(cfg, state)
    state.lsurf = tf.maximum(state.topg, -delta * state.thk + sealevel)
    state.usurf = state.lsurf + state.thk


@tf.function()
def compute_thk_implicit_1d_x_impl(
    u: tf.Tensor,
    h: tf.Tensor,
    dx: tf.Tensor,
    dt: tf.Tensor,
    smb: tf.Tensor,
    mode_u: str = "symmetric",
    mode_h: str = "zero",
) -> tf.Tensor:
    """
    Fully implicit first-order upwind solver for ∂H/∂t + ∂(uH)/∂x = smb
    in the x-direction only (quasi-1D, no y variation assumed).

    Assembles and solves a tridiagonal system for each y-row via
    tf.linalg.tridiagonal_solve. Unconditionally stable — no CFL constraint.

    Supports mode_h in {"zero", "symmetric", "extrapolate"}.
    """

    # --- Face velocities: (ny, nx+1) ---
    u_f = stag2x(pad_x(u, mode_u))
    up = tf.nn.relu(u_f)  # u⁺ = max(u, 0)
    um = -tf.nn.relu(-u_f)  # u⁻ = min(u, 0)

    inv_dx = 1.0 / dx
    inv_dt = 1.0 / dt

    # --- Raw tridiagonal coefficients from first-order upwind ---
    #
    # Implicit equation at cell i:
    #   H[i]/dt + (u⁺_{i+½}·H[i] + u⁻_{i+½}·H[i+1]
    #            - u⁺_{i-½}·H[i-1] - u⁻_{i-½}·H[i]) / dx  =  H⁰[i]/dt + smb[i]
    #
    sub = -up[:, :-1] * inv_dx  # coeff of H[i-1], (ny, nx)
    diag = inv_dt + (up[:, 1:] - um[:, :-1]) * inv_dx  # coeff of H[i],   (ny, nx)
    sup = um[:, 1:] * inv_dx  # coeff of H[i+1], (ny, nx)
    rhs = h * inv_dt + smb  #                   (ny, nx)

    # --- Incorporate ghost-cell boundary conditions (mirrors pad_x logic) ---
    #
    # sub[:, 0]  would couple to H_ghost_left  → absorb into diag/sup, then zero out
    # sup[:, -1] would couple to H_ghost_right → absorb into diag/sub, then zero out

    if mode_h == "zero":
        # Ghost H = 0: boundary couplings vanish; nothing to add
        pass

    elif mode_h == "symmetric":
        # H_ghost_left  = H[0]   → sub[0]   · H_ghost = sub[0]   · H[0]
        # H_ghost_right = H[-1]  → sup[-1]  · H_ghost = sup[-1]  · H[-1]
        diag = tf.concat(
            [diag[:, :1] + sub[:, :1], diag[:, 1:-1], diag[:, -1:] + sup[:, -1:]],
            axis=1,
        )

    elif mode_h == "extrapolate":
        # H_ghost_left  = 2·H[0] - H[1]
        #   → diag[0]  += 2·sub[0],   sup[0]  -= sub[0]
        # H_ghost_right = 2·H[-1] - H[-2]
        #   → diag[-1] += 2·sup[-1],  sub[-1] -= sup[-1]
        sub0 = sub[:, :1]  # snapshot before in-place update
        sup_end = sup[:, -1:]
        diag = tf.concat(
            [diag[:, :1] + 2.0 * sub0, diag[:, 1:-1], diag[:, -1:] + 2.0 * sup_end],
            axis=1,
        )
        sup = tf.concat([sup[:, :1] - sub0, sup[:, 1:]], axis=1)
        sub = tf.concat([sub[:, :-1], sub[:, -1:] - sup_end], axis=1)

    # Zero out the now-absorbed ghost couplings
    zeros = tf.zeros_like(sub[:, :1])
    sub = tf.concat([zeros, sub[:, 1:]], axis=1)  # sub[0]  = 0
    sup = tf.concat([sup[:, :-1], zeros], axis=1)  # sup[-1] = 0

    # --- Solve all y-rows simultaneously ---
    # compact format: diagonals[..., 0, :] = super, [1, :] = main, [2, :] = sub
    diagonals = tf.stack([sup, diag, sub], axis=1)  # (ny, 3, nx)
    h_new = tf.linalg.tridiagonal_solve(
        diagonals,
        rhs[..., tf.newaxis],  # (ny, nx, 1)
        diagonals_format="compact",
        partial_pivoting=False,  # Thomas algorithm; fine here since
    )  # diag ≥ 1/dt dominates off-diagonals
    return tf.maximum(
        tf.squeeze(h_new, axis=-1), 0.0
    )  # (ny, nx), non-negativity projection


@tf.function()
def compute_thk_implicit_1d_x(
    u: tf.Tensor,
    h: tf.Tensor,
    dx: tf.Tensor,
    dt: tf.Tensor,
    smb: tf.Tensor,
    omega: tf.Tensor,
    mode_u: str = "symmetric",
    mode_h: str = "zero",
) -> tf.Tensor:
    """
    Weighted implicit/explicit (theta-method) solver for ∂H/∂t + ∂(uH)/∂x = smb.

    omega=0 : fully explicit (forward Euler, CFL applies)
    omega=1 : fully implicit (backward Euler, unconditionally stable)
    omega>1 : over-implicit (extra damping of fast modes)

    omega is a tf.Tensor, allowing runtime variation (e.g. adaptive time-stepping).
    Solves one tridiagonal system per time step via Thomas algorithm.
    Boundary ghost cells handled algebraically — consistent with pad_x modes.
    """

    # --- Face velocities: (ny, nx+1) ---
    u_f = stag2x(pad_x(u, mode_u))
    up = tf.nn.relu(u_f)  # u⁺ = max(u, 0)
    um = -tf.nn.relu(-u_f)  # u⁻ = min(u, 0)

    inv_dx = 1.0 / dx
    inv_dt = 1.0 / dt

    # --- Explicit flux divergence at time n (for the (1-omega) part of the RHS) ---
    h_pad = pad_x(h, mode_h)  # (ny, nx+2)
    h_left = h_pad[:, :-2]  # H[i-1] with left ghost
    h_right = h_pad[:, 2:]  # H[i+1] with right ghost
    divflux_n = (
        up[:, 1:] * h + um[:, 1:] * h_right - up[:, :-1] * h_left - um[:, :-1] * h
    ) * inv_dx  # (ny, nx)

    # --- Tridiagonal coefficients (implicit part, scaled by omega) ---
    sub = -omega * up[:, :-1] * inv_dx  # coeff of H[i-1]
    diag = inv_dt + omega * (up[:, 1:] - um[:, :-1]) * inv_dx  # coeff of H[i]
    sup = omega * um[:, 1:] * inv_dx  # coeff of H[i+1]
    rhs = h * inv_dt + smb - (1.0 - omega) * divflux_n

    # --- Fold ghost cells into matrix coefficients (implicit part only) ---
    if mode_h == "symmetric":
        # H_ghost_left = H[0], H_ghost_right = H[-1]
        diag = tf.concat(
            [diag[:, :1] + sub[:, :1], diag[:, 1:-1], diag[:, -1:] + sup[:, -1:]],
            axis=1,
        )

    elif mode_h == "extrapolate":
        # H_ghost_left = 2H[0] - H[1], H_ghost_right = 2H[-1] - H[-2]
        sub0 = sub[:, :1]
        sup_end = sup[:, -1:]
        diag = tf.concat(
            [diag[:, :1] + 2.0 * sub0, diag[:, 1:-1], diag[:, -1:] + 2.0 * sup_end],
            axis=1,
        )
        sup = tf.concat([sup[:, :1] - sub0, sup[:, 1:]], axis=1)
        sub = tf.concat([sub[:, :-1], sub[:, -1:] - sup_end], axis=1)

    # mode_h == "zero": ghost = 0, boundary couplings vanish, nothing to do

    # Zero out the absorbed boundary couplings
    zeros = tf.zeros_like(sub[:, :1])
    sub = tf.concat([zeros, sub[:, 1:]], axis=1)  # sub[0]  = 0
    sup = tf.concat([sup[:, :-1], zeros], axis=1)  # sup[-1] = 0

    # --- Solve tridiagonal system ---
    # When omega=0: off-diagonals vanish, diag=inv_dt, solution = rhs*dt = explicit Euler
    diagonals = tf.stack([sup, diag, sub], axis=1)  # (ny, 3, nx)
    h_new = tf.linalg.tridiagonal_solve(
        diagonals,
        rhs[..., tf.newaxis],
        diagonals_format="compact",
        partial_pivoting=False,
    )
    return tf.maximum(tf.squeeze(h_new, axis=-1), 0.0)


def _update_explicit(cfg: DictConfig, state: State) -> None:
    """Update thickness using explicit slope-limiter scheme."""
    state.divflux = compute_divflux_slope_limiter(
        state.ubar,
        state.vbar,
        state.thk,
        state.dx,
        state.dx,
        state.dt,
        slope_type=cfg.processes.thk.slope_type,
        mode_h=cfg.processes.thk.flux_mode_h,
        mode_u=cfg.processes.thk.flux_mode_u,
    )
    state.thk = tf.maximum(state.thk + state.dt * (state.smb - state.divflux), 0)


def _update_implicit_1d_x(cfg: DictConfig, state: State) -> None:
    """Update thickness using implicit upwind scheme along x."""

    state.thk = compute_thk_implicit_1d_x(
        state.ubar,
        state.thk,
        state.dx,
        state.dt,
        state.smb,
        omega=cfg.processes.thk.omega,
        mode_u=cfg.processes.thk.flux_mode_u,
        mode_h=cfg.processes.thk.flux_mode_h,
    )


def initialize(cfg: DictConfig, state: State) -> None:
    if not hasattr(state, "topg"):
        raise ValueError(
            "The 'thk' module requires an initial topography ('state.topg') to be defined. "
            "Please define it through the preprocessing steps (not yet implemented)"
        )

    _compute_surfs(cfg, state)


def update(cfg: DictConfig, state: State) -> None:
    if state.it >= 0:
        if hasattr(state, "logger"):
            state.logger.info(f"Ice thickness equation at time: {state.t.numpy()}")

        state.smb = _get_smb(cfg, state)

        state.dt = 10.0
        tf.print(state.dt)

        time_scheme = cfg.processes.thk.time_scheme
        if time_scheme == "explicit":
            _update_explicit(cfg, state)
        elif time_scheme == "implicit_1d_x":
            state.thk = compute_thk_implicit_1d_x(
                state.ubar,
                state.thk,
                state.dx,
                state.dt,
                state.smb,
                omega=cfg.processes.thk.omega,
                mode_u=cfg.processes.thk.flux_mode_u,
                mode_h=cfg.processes.thk.flux_mode_h,
            )
        else:
            raise ValueError("Unknown time_scheme :(")

        _compute_surfs(cfg, state)


def finalize(cfg: DictConfig, state: State) -> None:
    pass
