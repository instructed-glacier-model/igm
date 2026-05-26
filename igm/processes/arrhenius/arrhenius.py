#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file
#
# Standalone process module computing the vertically-averaged Arrhenius
# factor `state.arrhenius` (MPa^-n yr^-1) used by the iceflow solver.
#
# Reads its inputs from state:
#   - state.T      : 3D ice temperature (K)            — produced by the
#                    enthalpy module (or any other module that exposes them)
#   - state.omega  : 3D ice water content fraction (-)
#
# So if you use enthalpy, place `arrhenius` AFTER `enthalpy` in the
# `override /processes` list so that each timestep's T, omega are fresh.

import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State

from ..enthalpy.temperature import compute_pa


def initialize(cfg: DictConfig, state: State) -> None:
    """Compute state.arrhenius from state.T, state.omega at t=0 (if available)."""
    if hasattr(state, "T") and hasattr(state, "omega"):
        compute_arrhenius(cfg, state, state.T, state.omega)


def update(cfg: DictConfig, state: State) -> None:
    """Recompute state.arrhenius from the current state.T, state.omega."""
    if hasattr(state, "logger"):
        state.logger.info(f"Update ARRHENIUS at iteration : {state.it}")

    if not (hasattr(state, "T") and hasattr(state, "omega")):
        raise RuntimeError(
            "arrhenius.update: state.T and state.omega are required. "
            "Make sure the enthalpy module (or another producer of these "
            "fields) runs BEFORE the arrhenius module in your "
            "`override /processes` list."
        )

    compute_arrhenius(cfg, state, state.T, state.omega)


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
    E = cfg_physics.enhancement_factor
    n = cfg_physics.exp_glen
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
