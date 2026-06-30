#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file
#
# Tulaczyk till hydrology: water layer ODE and effective-pressure parameterisation.
# Called from subglacial_hydrology.py when mode == "till_storage".

import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State


# ---------------------------------------------------------------------------
# Till water layer evolution
# ---------------------------------------------------------------------------

def update_h_water_till(cfg: DictConfig, state: State) -> tf.Tensor:
    cfg_ts = cfg.processes.subglacial_hydrology.till_storage
    return update_h_water_till_tf(
        state.h_water_till,
        cfg_ts.h_water_till_max,
        state.basal_melt_rate,
        cfg_ts.drainage_rate,
        state.thk,
        cfg.processes.iceflow.physics.ice_density,
        cfg_ts.water_density,
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
    """Advance the till water layer by one time step (Tulaczyk 2000 ODE).

    W^{n+1} = clip(W^n + dt * (rho_ice/rho_water * melt - drainage), 0, W_max)

    Ice-free cells are zeroed regardless of the ODE result.
    """
    h = h_water_till + dt * (rho_ice / rho_water * basal_melt_rate - drainage_rate)
    h = tf.clip_by_value(h, 0.0, h_water_till_max)
    return tf.where(h_ice > 0.0, h, 0.0)


# ---------------------------------------------------------------------------
# Effective pressure from till saturation
# ---------------------------------------------------------------------------

def compute_N_MPa(cfg: DictConfig, state: State) -> tf.Tensor:
    cfg_ts = cfg.processes.subglacial_hydrology.till_storage
    return compute_N_MPa_tf(
        state.h_water_till,
        cfg_ts.h_water_till_max,
        cfg.processes.iceflow.physics.ice_density,
        cfg.processes.iceflow.physics.gravity_cst,
        state.thk,
        cfg_ts.N_ref,
        cfg_ts.e_ref,
        cfg_ts.C_c,
        cfg_ts.delta,
    )


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
    """Tulaczyk (2000) effective pressure parameterisation (MPa).

    N = N_ref * (delta * p_ice / N_ref)^s * 10^((e_ref * (1 - s)) / C_c)
    capped above by p_ice, where s = h_water_till / h_water_till_max.

    Units: h_water_till / h_water_till_max (m), rho_ice (kg m^-3),
           g (m s^-2), h_ice (m), N_ref (MPa), e_ref / C_c / delta (dimless).
    """
    PA_TO_MPA = tf.cast(1.0e-6, h_ice.dtype)
    s = h_water_till / h_water_till_max
    p_ice = rho_ice * g * h_ice * PA_TO_MPA  # MPa
    N = N_ref * tf.pow(delta * p_ice / N_ref, s) * tf.pow(10.0, e_ref * (1.0 - s) / C_c)
    return tf.minimum(p_ice, N)
