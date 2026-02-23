#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Tuple
from omegaconf import DictConfig

from igm.common import State

from .utils import compute_pmp_tf, compute_T_tf, compute_omega_tf, compute_pa_tf


def compute_temperature(
    cfg: DictConfig, state: State, E_pmp: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Compute temperature and water content fields from enthalpy.

    Converts the enthalpy field to temperature and water content using
    the pressure melting point as the threshold between cold and temperate ice.

    Args:
        E_pmp: Pressure melting point enthalpy (J kg^-1).

    Returns T (K) and omega (-).
    """
    cfg_thermal = cfg.processes.enthalpy.thermal

    c_ice = cfg_thermal.c_ice
    L_ice = cfg_thermal.L_ice
    T_ref = cfg_thermal.T_ref

    T = compute_T_tf(state.E, E_pmp, T_ref, c_ice)
    omega = compute_omega_tf(state.E, E_pmp, L_ice)

    return T, omega


def compute_pmp(cfg: DictConfig, state: State) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Compute the pressure melting point enthalpy field.

    Calculates the pressure-dependent melting point throughout the ice column
    using the Clausius-Clapeyron relation.

    Returns E_pmp (J kg^-1) and T_pmp (K).
    """
    cfg_physics = cfg.processes.iceflow.physics
    cfg_thermal = cfg.processes.enthalpy.thermal

    rho_ice = cfg_physics.ice_density
    g = cfg_physics.gravity_cst

    beta = cfg_thermal.beta
    c_ice = cfg_thermal.c_ice
    T_pmp_ref = cfg_thermal.T_pmp_ref
    T_ref = cfg_thermal.T_ref

    depth_ice = state.iceflow.discr_v.enthalpy.depth * state.thk[None, ...]

    T_pmp, E_pmp = compute_pmp_tf(rho_ice, g, depth_ice, beta, c_ice, T_pmp_ref, T_ref)
    return E_pmp, T_pmp


def compute_pa(cfg: DictConfig, state: State, T: tf.Tensor) -> tf.Tensor:
    """
    Compute the pressure-adjusted temperature field.

    Adjusts the temperature field for the pressure-melting point depression
    using the Clausius-Clapeyron relation.

    Args:
        T: Temperature field (K).

    Returns:
        Pressure-adjusted temperature (K).
    """
    cfg_physics = cfg.processes.iceflow.physics
    cfg_thermal = cfg.processes.enthalpy.thermal

    rho_ice = cfg_physics.ice_density
    g = cfg_physics.gravity_cst
    beta = cfg_thermal.beta

    depth_ice = state.iceflow.discr_v.enthalpy.depth * state.thk[None, ...]

    return compute_pa_tf(T, beta, rho_ice, g, depth_ice)
