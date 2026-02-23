#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State

from .utils import compute_fraction_drained


def update_drainage(cfg: DictConfig, state: State, E_pmp: tf.Tensor) -> None:
    """
    Update enthalpy field by draining excess water from the ice column.

    Removes water content exceeding threshold values and adds the drained
    water to the basal melt rate. Skips drainage if dt is zero or drainage
    is disabled in configuration.

    Args:
        E_pmp: Pressure melting point enthalpy (J kg^-1).

    Updates state.E (J kg^-1) and state.basal_melt_rate (m yr^-1).
    """
    cfg_physics = cfg.processes.iceflow.physics
    cfg_thermal = cfg.processes.enthalpy.thermal
    cfg_drainage = cfg.processes.enthalpy.drainage

    rho_ice = cfg_physics.ice_density

    L_ice = cfg_thermal.L_ice

    rho_water = cfg_drainage.water_density
    drain_ice_column = cfg_drainage.drain_ice_column
    omega_target = cfg_drainage.omega_target
    omega_threshold_1 = cfg_drainage.omega_threshold_1
    omega_threshold_2 = cfg_drainage.omega_threshold_2
    omega_threshold_3 = cfg_drainage.omega_threshold_3

    if (state.dt == 0.0) or not drain_ice_column:
        return

    dzeta = state.iceflow.discr_v.enthalpy.dzeta
    dz = dzeta * state.thk[None, ...]

    fraction_drained, h_drained = compute_fraction_drained(
        state.E,
        E_pmp,
        L_ice,
        omega_target,
        omega_threshold_1,
        omega_threshold_2,
        omega_threshold_3,
        dz,
        state.dt,
    )

    state.E -= fraction_drained * L_ice

    state.basal_melt_rate += (rho_ice / rho_water) * h_drained / state.dt
