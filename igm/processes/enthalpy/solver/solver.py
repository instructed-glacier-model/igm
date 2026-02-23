#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State

from .horizontal import update_horizontal
from .vertical import update_vertical
from .drainage import update_drainage


def update_enthalpy(
    cfg: DictConfig,
    state: State,
    strain_heat: tf.Tensor,
    friction_heat: tf.Tensor,
    E_pmp: tf.Tensor,
    E_s: tf.Tensor,
) -> None:
    """
    Update the enthalpy field over a time step.

    Performs the complete enthalpy evolution including horizontal advection
    (explicit), vertical advection-diffusion (implicit), and water drainage.

    Args:
        strain_heat: Volumetric strain heating rate (W m^-3).
        friction_heat: Areal frictional heating rate at the bed (W m^-2).
        E_pmp: Pressure melting point enthalpy (J kg^-1).
        E_s: Surface enthalpy boundary condition (J kg^-1).

    Updates state.E (J kg^-1) and state.basal_melt_rate (m yr^-1).
    """
    # Horizontal advection (explicit)
    update_horizontal(cfg, state)

    # Vertical advection-diffusion (implicit)
    update_vertical(cfg, state, strain_heat, friction_heat, E_pmp, E_s)

    # Drainage
    update_drainage(cfg, state, E_pmp)

    # Prevent refreezing if needed
    allow_basal_refreezing = cfg.processes.enthalpy.solver.allow_basal_refreezing
    if not allow_basal_refreezing:
        state.basal_melt_rate = tf.maximum(state.basal_melt_rate, 0.0)
