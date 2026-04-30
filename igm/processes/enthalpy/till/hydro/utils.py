#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State


def update_h_water_till(cfg: DictConfig, state: State) -> tf.Tensor:
    """
    Update the till water layer thickness over a time step.

    Evolves the water layer based on basal melt rate and drainage,
    clamped to valid bounds, with zero water in ice-free areas.

    Returns:
        Updated till water layer thickness (m water).
    """
    h_water_till = state.h_water_till
    basal_melt_rate = state.basal_melt_rate
    h_ice = state.thk
    dt = state.dt

    cfg_hydro = cfg.processes.enthalpy.till.hydro
    cfg_physics = cfg.processes.iceflow.physics
    cfg_drainage = cfg.processes.enthalpy.drainage

    drainage_rate = cfg_hydro.drainage_rate
    h_water_till_max = cfg_hydro.h_water_till_max
    rho_ice = cfg_physics.ice_density
    rho_water = cfg_drainage.water_density

    return update_h_water_till_tf(
        h_water_till,
        h_water_till_max,
        basal_melt_rate,
        drainage_rate,
        h_ice,
        rho_ice,
        rho_water,
        dt,
    )


@tf.function()
def update_h_water_till_tf(
    h_water_till: tf.Tensor,
    h_water_till_max: tf.Tensor,
    basal_melt_rate: tf.Tensor,
    drainage_rate: tf.Tensor,
    h_ice: tf.Tensor,
    rho_ice: tf.Tensor,
    rho_water: tf.Tensor,
    dt: tf.Tensor,
) -> tf.Tensor:
    """
    TensorFlow function to update till water layer thickness.

    Args:
        h_water_till: Current till water layer thickness (m water).
        h_water_till_max: Maximum till water layer thickness (m water).
        basal_melt_rate: Basal melt rate (m ice yr^-1).
        drainage_rate: Till drainage rate (m water yr^-1).
        h_ice: Ice thickness field (m).
        rho_ice: Ice density (kg m^-3).
        rho_water: Water density (kg m^-3).
        dt: Time step (yr).

    Returns:
        Updated till water layer thickness (m water).
    """
    h_water_till += dt * (rho_ice / rho_water * basal_melt_rate - drainage_rate)
    h_water_till = tf.clip_by_value(h_water_till, 0.0, h_water_till_max)
    return tf.where(h_ice > 0.0, h_water_till, 0.0)


def compute_N(cfg: DictConfig, state: State) -> tf.Tensor:
    """
    Compute the effective pressure field from till water content.

    Uses the Tulaczyk et al. (2000) parameterization relating till water
    saturation to effective pressure through void ratio changes.

    Returns:
        Effective pressure field (Pa).
    """
    cfg_physics = cfg.processes.iceflow.physics
    cfg_hydro = cfg.processes.enthalpy.till.hydro

    rho_ice = cfg_physics.ice_density
    g = cfg_physics.gravity_cst

    h_water_till_max = cfg_hydro.h_water_till_max
    N_ref = cfg_hydro.N_ref
    e_ref = cfg_hydro.e_ref
    C_c = cfg_hydro.C_c
    delta = cfg_hydro.delta

    h_water_till = state.h_water_till
    h_ice = state.thk

    return compute_N_tf(
        h_water_till, h_water_till_max, rho_ice, g, h_ice, N_ref, e_ref, C_c, delta
    )


@tf.function()
def compute_N_tf(
    h_water_till: tf.Tensor,
    h_water_till_max: tf.Tensor,
    rho_ice: tf.Tensor,
    g: tf.Tensor,
    h_ice: tf.Tensor,
    N_ref: tf.Tensor,
    e_ref: tf.Tensor,
    C_c: tf.Tensor,
    delta: tf.Tensor,
) -> tf.Tensor:
    """
    TensorFlow function to compute effective pressure from till saturation.

    Args:
        h_water_till: Till water layer thickness (m).
        h_water_till_max: Maximum till water layer thickness (m).
        rho_ice: Ice density (kg m^-3).
        g: Gravitational acceleration (m s^-2).
        h_ice: Ice thickness field (m).
        N_ref: Reference effective pressure (Pa).
        e_ref: Reference void ratio (-).
        C_c: Till compressibility coefficient (-).
        delta: Minimum effective pressure fraction (-).

    Returns:
        Effective pressure field (Pa).
    """
    s = h_water_till / h_water_till_max
    p_ice = rho_ice * g * h_ice

    N = N_ref * ((delta * p_ice / N_ref) ** s) * 10 ** (e_ref * (1 - s) / C_c)

    return tf.minimum(p_ice, N)
