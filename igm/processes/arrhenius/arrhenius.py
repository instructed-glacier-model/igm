#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file
#
# Standalone process module computing the vertically-averaged Arrhenius
# factor `state.arrhenius` (MPa^-n yr^-1) used by the iceflow solver.
#
# Reads state.E (3D enthalpy, J kg^-1) produced by the enthalpy module and
# derives temperature and water content on-the-fly via compute_pmp /
# compute_temperature.  Place `arrhenius` AFTER `enthalpy` in the
# `override /processes` list so that each timestep's E is fresh.

import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State

from ..enthalpy.temperature import compute_pa, compute_pmp, compute_temperature


def initialize(cfg: DictConfig, state: State) -> None:
    """Compute state.arrhenius from state.E at t=0 (if available)."""
    if hasattr(state, "E"):
        E_pmp, _ = compute_pmp(cfg, state)
        T, omega = compute_temperature(cfg, state, E_pmp)
        compute_arrhenius(cfg, state, T, omega)


def update(cfg: DictConfig, state: State) -> None:
    """Recompute state.arrhenius from the current state.E."""
    if hasattr(state, "logger"):
        state.logger.info(f"Update ARRHENIUS at iteration : {state.it}")

    if not hasattr(state, "E"):
        raise RuntimeError(
            "arrhenius.update: state.E is required. "
            "Make sure the enthalpy module runs BEFORE the arrhenius module "
            "in your `override /processes` list."
        )

    E_pmp, _ = compute_pmp(cfg, state)
    T, omega = compute_temperature(cfg, state, E_pmp)
    compute_arrhenius(cfg, state, T, omega)


def finalize(cfg: DictConfig, state: State) -> None:
    pass


def compute_arrhenius(
    cfg: DictConfig, state: State, T: tf.Tensor, omega: tf.Tensor
) -> None:
    """
    Compute the vertically-averaged Arrhenius factor for ice flow.

    Calculates the 3D Arrhenius factor A scaled by an enhancement factor, then performs
    vertical averaging over B = A^(-1/n) (related to viscosity) rather than over A directly.
    The final result is converted back: A_avg = (Σ B_i * w_i)^(-n).

    This approach is physically motivated since B is proportional to viscosity,
    and averaging viscosity is more appropriate than averaging the rate factor.

    Args:
        T: Temperature field (K).
        omega: Water content fraction (-).

    Updates state.arrhenius (MPa^-n yr^-1).
    """
    cfg_physics = cfg.processes.iceflow.physics
    E = cfg_physics.viscosity.enhancement_factor
    n = cfg_physics.viscosity.exponent
    weights = state.iceflow.discr_v.enthalpy.weights

    # Compute 3D Arrhenius factor with enhancement
    A = E * compute_arrhenius_3d(cfg, state, T, omega)

    # Average over B = A^(-1/n) and convert back
    B = tf.pow(A, -1.0 / n)
    B_avg = tf.reduce_sum(B * weights, axis=0)
    state.arrhenius = tf.pow(B_avg, -n)


def compute_arrhenius_3d(
    cfg: DictConfig, state: State, T: tf.Tensor, omega: tf.Tensor
) -> tf.Tensor:
    """
    Compute the 3D Arrhenius factor field for ice viscosity.

    Uses a two-regime Arrhenius law (cold/warm ice) with water content enhancement
    to compute the rate factor throughout the ice column.

    Args:
        T: Temperature field (K).
        omega: Water content fraction (-).

    Returns:
        3D Arrhenius factor field (MPa^-n yr^-1).
    """
    T_pa = compute_pa(cfg, state, T)

    cfg_arrhenius = cfg.processes.arrhenius

    return compute_arrhenius_3d_tf(
        omega,
        T_pa,
        cfg_arrhenius.T_threshold,
        cfg_arrhenius.A_cold,
        cfg_arrhenius.A_warm,
        cfg_arrhenius.Q_cold,
        cfg_arrhenius.Q_warm,
        cfg_arrhenius.omega_coef,
        cfg_arrhenius.omega_max,
        cfg_arrhenius.R,
    )


@tf.function()
def compute_arrhenius_3d_tf(
    omega: tf.Tensor,
    T_pa: tf.Tensor,
    T_threshold: tf.Tensor,
    A_cold: tf.Tensor,
    A_warm: tf.Tensor,
    Q_cold: tf.Tensor,
    Q_warm: tf.Tensor,
    omega_coef: tf.Tensor,
    omega_max: tf.Tensor,
    R: tf.Tensor,
) -> tf.Tensor:
    """
    TensorFlow function to compute the 3D Arrhenius factor.

    Args:
        omega: Water content fraction (-).
        T_pa: Pressure-adjusted temperature (K).
        T_threshold: Temperature threshold between cold and warm regimes (K).
        A_cold: Pre-exponential factor for cold ice (Pa^-n s^-1).
        A_warm: Pre-exponential factor for warm ice (Pa^-n s^-1).
        Q_cold: Activation energy for cold ice (J mol^-1).
        Q_warm: Activation energy for warm ice (J mol^-1).
        omega_coef: Water content enhancement coefficient (-).
        omega_max: Maximum water content for enhancement (-).
        R: Universal gas constant (J mol^-1 K^-1).

    Returns:
        3D Arrhenius factor field (MPa^-n yr^-1).
    """
    A = tf.where(T_pa < T_threshold, A_cold, A_warm)
    Q = tf.where(T_pa < T_threshold, Q_cold, Q_warm)

    omega_factor = 1.0 + omega_coef * tf.minimum(omega, omega_max)

    spy = 31556926.0

    return omega_factor * A * tf.math.exp(-Q / (R * T_pa)) * spy * 1e18
