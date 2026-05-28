#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State
from igm.common.fields import get_tau_ref
from igm.utils.grad.grad import grad_xy, pad_x, pad_y, pad_z
from igm.utils.grad.strain_rate import compute_eps_dot2, correct_grad_zeta_to_z


def compute_strain_heat(cfg: DictConfig, state: State) -> tf.Tensor:
    """
    Compute the volumetric strain heating rate field.

    Calculates viscous dissipation from velocity gradients and the Arrhenius
    factor using Glen's flow law.

    Returns:
        Volumetric strain heating rate (W m^-3).
    """
    cfg_physics = cfg.processes.iceflow.physics

    n = cfg_physics.exp_glen
    h_min = cfg_physics.thr_ice_thk

    vertical_discr_E = state.iceflow.discr_v.enthalpy
    zeta = vertical_discr_E.zeta
    dzeta = vertical_discr_E.dzeta
    V_U_to_E = vertical_discr_E.V_U_to_E
    dz = dzeta * state.thk[None, ...]

    return compute_strain_heat_tf(
        state.U,
        state.V,
        state.thk,
        state.usurf,
        state.arrhenius,
        n,
        V_U_to_E,
        zeta,
        state.dX,
        dz,
        h_min,
    )


@tf.function
def compute_strain_heat_tf(
    U: tf.Tensor,
    V: tf.Tensor,
    h: tf.Tensor,
    s: tf.Tensor,
    arrhenius: tf.Tensor,
    n: tf.Tensor,
    V_U_to_E: tf.Tensor,
    zeta: tf.Tensor,
    dX: tf.Tensor,
    dz: tf.Tensor,
    h_min: tf.Tensor,
    mode_pad_xy: str = "symmetric",
    mode_pad_z: str = "extrapolate",
) -> tf.Tensor:
    """
    TensorFlow function to compute volumetric strain heating.

    Args:
        U: Horizontal velocity in x-direction (m yr^-1).
        V: Horizontal velocity in y-direction (m yr^-1).
        h: Ice thickness (m)
        s: Upper-surface elevation (m)
        arrhenius: Arrhenius factor field (MPa^-n yr^-1).
        n: Glen's flow law exponent (-).
        V_U_to_E: Map velocity DOFs to values at enthalpy nodes (Ndof_E, Ndof_U).
        zeta: Normalized elevation of each node/level (-).
        dx: Horizontal grid spacing (m).
        dz: Vertical grid spacing field (m).
        h_min: Minimum ice thickness threshold (m).
        mode_pad_xy: Padding mode for horizontal boundaries.
        mode_pad_z: Padding mode for vertical boundaries.

    Returns:
        Volumetric strain heating rate (W m^-3).
    """
    spy = 31556926.0

    U_si = tf.einsum("ij,jkl->ikl", V_U_to_E, U) / spy
    V_si = tf.einsum("ij,jkl->ikl", V_U_to_E, V) / spy

    # Pad velocities in x, y, z directions
    Ui = pad_x(U_si, mode=mode_pad_xy)
    Uj = pad_y(U_si, mode=mode_pad_xy)
    Uk = pad_z(U_si, mode=mode_pad_z)

    Vi = pad_x(V_si, mode=mode_pad_xy)
    Vj = pad_y(V_si, mode=mode_pad_xy)
    Vk = pad_z(V_si, mode=mode_pad_z)

    # Effective vertical spacing for finite differences
    dz_pad = pad_z(dz, mode="symmetric")
    dz_mean = (dz_pad[:-1, :, :] + dz_pad[1:, :, :]) / 2.0
    dz_eff = tf.maximum(dz_mean, h_min)

    # Lower and upper surface gradients
    l = s - h
    dldx, dldy = grad_xy(l, dX, dX, False, "extrapolate")
    dsdx, dsdy = grad_xy(s, dX, dX, False, "extrapolate")

    # Horizontal gradients
    dudx = (Ui[:, :, 2:] - Ui[:, :, :-2]) / (2.0 * dX)
    dudy = (Uj[:, 2:, :] - Uj[:, :-2, :]) / (2.0 * dX)
    dvdx = (Vi[:, :, 2:] - Vi[:, :, :-2]) / (2.0 * dX)
    dvdy = (Vj[:, 2:, :] - Vj[:, :-2, :]) / (2.0 * dX)

    # Vertical gradients
    dudz = (Uk[2:, :, :] - Uk[:-2, :, :]) / (2.0 * dz_eff)
    dvdz = (Vk[2:, :, :] - Vk[:-2, :, :]) / (2.0 * dz_eff)

    # Correct for terrain-following coordinates
    zeta = zeta[:, None, None]
    dldx = dldx[None, :, :]
    dldy = dldy[None, :, :]
    dsdx = dsdx[None, :, :]
    dsdy = dsdy[None, :, :]
    dudx, dudy, dvdx, dvdy = correct_grad_zeta_to_z(
        dudx,
        dudy,
        dvdx,
        dvdy,
        dudz,
        dvdz,
        dldx,
        dldy,
        dsdx,
        dsdy,
        zeta,
    )

    # Effactive strain rate
    eps_dot2 = compute_eps_dot2(dudx, dvdx, dudy, dvdy, dudz, dvdz)
    eps_dot = tf.sqrt(eps_dot2)

    # Convert arrhenius units: MPa^(-n) yr^(-1) to Pa^(-n) s^(-1)
    units_conv = tf.pow(1e6, n) * spy

    arrhenius_3d = (
        tf.expand_dims(arrhenius, axis=0) if arrhenius.ndim == 2 else arrhenius
    )

    # Phi = 2 * A^(-1/n) * ε_dot^(1+1/n)
    return (
        2.0
        * tf.pow(arrhenius_3d / units_conv, -1.0 / n)
        * tf.pow(eps_dot, 1.0 + 1.0 / n)
    )


def compute_friction_heat(cfg: DictConfig, state: State) -> tf.Tensor:
    """
    Compute the areal frictional heating rate at the bed.

    Dispatches on `cfg.processes.iceflow.physics.sliding.law`:

      * weertman    : heat ∝ slidingco · |u_b|^(1+1/m)   (legacy)
      * mohr_coulomb: heat ∝ (N·tan(phi)·u_ref^(-1/m)) · |u_b|^(1+1/m)
                      i.e. the effective Weertman-style C is built from
                      state.effective_pressure and the law's friction angle.

    Returns:
        Areal frictional heating rate (W m^-2).
    """
    cfg_physics = cfg.processes.iceflow.physics
    law = cfg_physics.sliding.law

    if law == "weertman":
        m = cfg_physics.sliding.weertman.exponent
        u_regu = cfg_physics.sliding.weertman.regu
        C = get_tau_ref(state)
    elif law == "mohr_coulomb":
        cfg_mc = cfg_physics.sliding.mohr_coulomb
        m = cfg_mc.exponent
        u_regu = cfg_mc.regu
        # Build effective Weertman-style C from N * tan(phi) * u_ref^(-1/m)
        # so the existing compute_friction_heat_tf kernel (which assumes the
        # tau_b = C * u_b^(1/m) form) produces the right dissipation.
        from igm.processes.iceflow.energy.components.sliding.laws.mohr_coulomb import (
            _compute_phi as _mc_compute_phi,
            MohrCoulombParams,
        )
        params = MohrCoulombParams(
            regu=cfg_mc.regu,
            exponent=cfg_mc.exponent,
            u_ref=cfg_mc.u_ref,
            phi=cfg_mc.phi,
            phi_min=cfg_mc.phi_min,
            phi_max=cfg_mc.phi_max,
            bed_min=cfg_mc.bed_min,
            bed_max=cfg_mc.bed_max,
            tauc_min=cfg_mc.tauc_min,
            tauc_max=cfg_mc.tauc_max,
            tauc_ice_free=cfg_mc.tauc_ice_free,
            rho_ratio=cfg_physics.water_density / cfg_physics.ice_density,
            use_mask_gr=cfg_physics.sliding.use_mask_gr,
        )
        topg = state.usurf - state.thk
        phi = _mc_compute_phi(topg, params)
        tan_phi = tf.math.tan(phi * tf.constant(3.14159265358979 / 180.0, dtype=phi.dtype))
        tauc = state.effective_pressure * tan_phi
        # Match the sliding law's ordering: ice-free hard assign before clip.
        tauc = tf.where(state.thk > 0.0, tauc, cfg_mc.tauc_ice_free)
        tauc = tf.clip_by_value(tauc, cfg_mc.tauc_min, cfg_mc.tauc_max)
        C = tauc * tf.pow(tf.cast(cfg_mc.u_ref, tauc.dtype), -1.0 / m)
    else:
        raise ValueError(
            f"compute_friction_heat: unsupported sliding law '{law}' "
            "(supported: 'weertman', 'mohr_coulomb')"
        )

    V_b = state.iceflow.discr_v.V_b

    return compute_friction_heat_tf(
        state.U,
        state.V,
        C,
        state.thk,
        state.usurf,
        state.dX,
        m,
        u_regu,
        V_b,
    )


@tf.function
def compute_friction_heat_tf(
    U: tf.Tensor,
    V: tf.Tensor,
    C: tf.Tensor,
    h: tf.Tensor,
    s: tf.Tensor,
    dx: tf.Tensor,
    m: tf.Tensor,
    u_regu: tf.Tensor,
    V_b: tf.Tensor,
) -> tf.Tensor:
    """
    TensorFlow function to compute basal frictional heating.

    Args:
        U: Horizontal velocity in x-direction (m yr^-1).
        V: Horizontal velocity in y-direction (m yr^-1).
        C: Sliding coefficient field (MPa m^-1/m yr^1/m).
        h: Ice thickness (m)
        s: Upper-surface elevation (m)
        dx: Horizontal grid spacing (m).
        m: Weertman sliding law exponent (-).
        u_regu: Velocity regularization (m yr^-1).
        V_b: Basal extraction vector: dofs -> basal (-).

    Returns:
        Areal frictional heating rate (W m^-2).
    """

    # Retrieve basal velocity
    ux_b = tf.einsum("j,jkl->kl", V_b, U)
    uy_b = tf.einsum("j,jkl->kl", V_b, V)

    # Compute bed gradient ∇b
    b = s - h
    dbdx, dbdy = grad_xy(b, dx, dx, False, "extrapolate")

    # Compute basal velocity magnitude (with norm M and regularization)
    u_corr_b = ux_b * dbdx + uy_b * dbdy
    u_b = tf.sqrt(ux_b * ux_b + uy_b * uy_b + u_regu * u_regu + u_corr_b * u_corr_b)

    # Transform to SI units
    spy = 31556926.0
    u_b_si = u_b / spy
    C_si = C * 1e6 * spy ** (1.0 / m)

    # C |u_b|^(1/m + 1)
    return C_si * tf.pow(u_b_si, 1.0 / m + 1.0)
