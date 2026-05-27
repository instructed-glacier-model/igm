#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file
#
# Tulaczyk-style till hydrology + effective-pressure parameterization,
# as used by Van Pelt & Bueler (2015). Decoupled out of the enthalpy
# module so it can be picked as a method of the `effective_pressure`
# umbrella.
#
# State contract:
#   reads:   state.basal_melt_rate (m ice/yr), state.thk, state.dt
#   writes:  state.h_water_till (m water), state.effective_pressure (MPa)

import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State


def initialize(cfg: DictConfig, state: State) -> None:
    """Initial till water and effective pressure."""
    if not hasattr(state, "h_water_till"):
        state.h_water_till = tf.zeros_like(state.thk)
    if not hasattr(state, "basal_melt_rate"):
        state.basal_melt_rate = tf.zeros_like(state.thk)

    state.effective_pressure = _compute_N_MPa(cfg, state)


def update(cfg: DictConfig, state: State) -> None:
    """Evolve till water by one step, then recompute N (MPa)."""
    state.h_water_till = _update_h_water_till(cfg, state)
    state.effective_pressure = _compute_N_MPa(cfg, state)


def _update_h_water_till(cfg: DictConfig, state: State) -> tf.Tensor:
    """Update till water layer thickness (m water) over one time step."""
    cfg_vpb = cfg.processes.effective_pressure.vanpelt_bueler
    cfg_phys = cfg.processes.iceflow.physics

    return update_h_water_till_tf(
        state.h_water_till,
        cfg_vpb.h_water_till_max,
        state.basal_melt_rate,
        cfg_vpb.drainage_rate,
        state.thk,
        cfg_phys.ice_density,
        cfg_vpb.water_density,
        state.dt,
    )


def _compute_N_MPa(cfg: DictConfig, state: State) -> tf.Tensor:
    """Tulaczyk effective-pressure parameterization, returned in MPa.

    The legacy enthalpy/till/hydro implementation worked in Pa; we keep
    the formula identical but apply a single Pa->MPa scale at the
    output. Equivalently, p_ice is computed in MPa from the start
    (which is what this implementation does).
    """
    cfg_vpb = cfg.processes.effective_pressure.vanpelt_bueler
    cfg_phys = cfg.processes.iceflow.physics

    return compute_N_MPa_tf(
        state.h_water_till,
        cfg_vpb.h_water_till_max,
        cfg_phys.ice_density,
        cfg_phys.gravity_cst,
        state.thk,
        cfg_vpb.N_ref,
        cfg_vpb.e_ref,
        cfg_vpb.C_c,
        cfg_vpb.delta,
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
    """TensorFlow function to update till water layer thickness."""
    h_water_till = h_water_till + dt * (
        rho_ice / rho_water * basal_melt_rate - drainage_rate
    )
    h_water_till = tf.clip_by_value(h_water_till, 0.0, h_water_till_max)
    return tf.where(h_ice > 0.0, h_water_till, 0.0)


@tf.function()
def compute_N_MPa_tf(
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
    """Tulaczyk parameterization for effective pressure (MPa).

    N = N_ref * (delta * p_ice / N_ref)^s * 10^((e_ref * (1 - s)) / C_c)
    capped above by p_ice.

    Args (note units):
        h_water_till:     (m water)
        h_water_till_max: (m water)
        rho_ice:          (kg m^-3)
        g:                (m s^-2)
        h_ice:            (m)
        N_ref:            (MPa)
        e_ref, C_c, delta: dimensionless
    """
    PA_TO_MPA = tf.cast(1.0e-6, h_ice.dtype)

    s = h_water_till / h_water_till_max
    p_ice = rho_ice * g * h_ice * PA_TO_MPA   # (MPa)

    N = N_ref * tf.pow(delta * p_ice / N_ref, s) * tf.pow(
        10.0, e_ref * (1.0 - s) / C_c
    )

    return tf.minimum(p_ice, N)
