#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file
#
# Tulaczyk effective-pressure parameterisation driven by till saturation.
# Helper functions called from effective_pressure.py when mode == "till_storage".

import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State


def compute_N_MPa(cfg: DictConfig, state: State) -> tf.Tensor:
    """Tulaczyk effective-pressure parameterization, returned in MPa."""
    cfg_ts = cfg.processes.effective_pressure.till_storage
    cfg_hwt = cfg.processes.h_water_till
    cfg_phys = cfg.processes.iceflow.physics

    return compute_N_MPa_tf(
        state.h_water_till,
        cfg_hwt.h_water_till_max,
        cfg_phys.ice_density,
        cfg_phys.gravity_cst,
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
