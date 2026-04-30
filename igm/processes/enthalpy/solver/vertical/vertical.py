#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State

from .utils.assembly import assemble_system
from .utils.bc import compute_bc
from .utils.diffusivity import compute_diffusivity
from .utils.melt import compute_basal_melt_rate
from .utils.solver import solve_tridiagonal_system
from .utils.velocity import correct_vertical_velocity


def update_vertical(
    cfg: DictConfig,
    state: State,
    strain_heat: tf.Tensor,
    friction_heat: tf.Tensor,
    E_pmp: tf.Tensor,
    E_s: tf.Tensor,
) -> None:
    """
    Update enthalpy field for vertical advection-diffusion over a time step.

    Solves the 1D vertical enthalpy equation implicitly using a finite difference
    scheme with upwind advection. Handles boundary conditions at the surface and
    base, enforces enthalpy bounds, and computes the basal melt rate.

    Args:
        strain_heat: Volumetric strain heating rate (W m^-3).
        friction_heat: Areal frictional heating rate at the bed (W m^-2).
        E_pmp: Pressure melting point enthalpy (J kg^-1).
        E_s: Surface enthalpy boundary condition (J kg^-1).

    Updates state.E (J kg^-1) and state.basal_melt_rate (m ice yr^-1).
    """
    cfg_thermal = cfg.processes.enthalpy.thermal
    cfg_physics = cfg.processes.iceflow.physics
    cfg_solver = cfg.processes.enthalpy.solver

    rho_ice = cfg_physics.ice_density
    dz_min = cfg_physics.thr_ice_thk

    k_ice = cfg_thermal.k_ice
    c_ice = cfg_thermal.c_ice
    L_ice = cfg_thermal.L_ice
    T_ref = cfg_thermal.T_ref
    T_min = cfg_thermal.T_min
    K_ratio = cfg_thermal.K_ratio
    correct_w_for_melt = cfg_solver.correct_w_for_melt
    override_basal_at_pmp = cfg_solver.override_basal_at_pmp

    dzeta = state.iceflow.discr_v.enthalpy.dzeta
    V_U_to_E = state.iceflow.discr_v.enthalpy.V_U_to_E
    dz = dzeta * state.thk[None, ...]

    # Correct vertical velocity
    W = state.W if hasattr(state, "W") else tf.zeros_like(state.U)
    Wc = tf.einsum("ij,jkl->ikl", V_U_to_E, W)
    Wc = correct_vertical_velocity(Wc, state.basal_melt_rate, correct_w_for_melt)

    # Thermal diffusivity
    K = compute_diffusivity(state.E, E_pmp, k_ice, rho_ice, c_ice, K_ratio)

    # Source term
    f = strain_heat / rho_ice

    # Boundary conditions
    q_basal = state.basal_heat_flux + friction_heat
    dEdz_dry = -(c_ice / k_ice) * q_basal
    BCB, VB, VS = compute_bc(state.E, E_pmp, E_s, state.h_water_till, dEdz_dry)

    if override_basal_at_pmp:
        BCB = tf.zeros_like(E_s)
        VB = E_pmp[0]
        VS = E_s

    # Assemble system
    spy = 31556926.0
    L, M, U, R = assemble_system(
        state.E, state.dt * spy, tf.maximum(dz, dz_min), Wc / spy, K, f, BCB, VB, VS
    )

    # Solve system
    state.E = solve_tridiagonal_system(L, M, U, R)

    # Clamp basal enthalpy to E_pmp at dry-bed points to avoid spurious melt spikes
    E_base = tf.where(
        state.h_water_till <= 0.0, tf.minimum(state.E[0], E_pmp[0]), state.E[0]
    )
    state.E = tf.concat([E_base[None, ...], state.E[1:]], axis=0)

    # Enforce bounds
    E_min = c_ice * (T_min - T_ref)
    E_max = E_pmp + L_ice
    state.E = tf.clip_by_value(state.E, E_min, E_max)

    # Compute basal melt rate
    state.basal_melt_rate = compute_basal_melt_rate(
        state.E,
        E_pmp,
        E_s,
        state.h_water_till,
        q_basal,
        k_ice,
        c_ice,
        K_ratio,
        rho_ice,
        L_ice,
        tf.maximum(dz[0], dz_min),
    )
