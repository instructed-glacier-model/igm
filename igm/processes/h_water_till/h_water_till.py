#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file
#
# Till water layer evolution.  Integrates state.h_water_till forward
# in time using the Tulaczyk (2000) drainage ODE driven by the basal
# melt rate produced by the enthalpy solver.
#
# State contract:
#   reads:  state.h_water_till (m water), state.basal_melt_rate (m ice/yr),
#           state.thk (m), state.dt (yr)
#   writes: state.h_water_till (m water)

import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State


def initialize(cfg: DictConfig, state: State) -> None:
    """Initialize till water layer to zero if not already set."""
    if not hasattr(state, "h_water_till"):
        state.h_water_till = tf.zeros_like(state.thk)
    if not hasattr(state, "basal_melt_rate"):
        state.basal_melt_rate = tf.zeros_like(state.thk)


def update(cfg: DictConfig, state: State) -> None:
    """Evolve till water layer by one time step."""
    state.h_water_till = _update_h_water_till(cfg, state)


def finalize(cfg: DictConfig, state: State) -> None:
    pass


def _update_h_water_till(cfg: DictConfig, state: State) -> tf.Tensor:
    cfg_hwt = cfg.processes.h_water_till
    cfg_phys = cfg.processes.iceflow.physics

    return update_h_water_till_tf(
        state.h_water_till,
        cfg_hwt.h_water_till_max,
        state.basal_melt_rate,
        cfg_hwt.drainage_rate,
        state.thk,
        cfg_phys.ice_density,
        cfg_hwt.water_density,
        state.dt,
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
