#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Basal boundary condition of the ice in contact with the ocean.

Under floating ice (and ice-free ocean), the base of the column is in contact
with sea water and its enthalpy is prescribed (Dirichlet), as in PISM: that of ice at the shelf-base temperature, without water. The
shelf-base temperature is the freezing point of sea water at the ice draft,
T_f(S, z_b), when the ``ocean`` process provides the salinity (with its
freezing-point coefficients, as for the sub-shelf melt), and otherwise the
pressure-melting point of ice, as PISM does without an ocean model. It never
exceeds the pressure-melting point. Geothermal and frictional heat do not
reach floating ice, and the basal melt rate of the energy model is 0 there:
the ocean-induced melt is that of the ``bmb`` process.
"""

from typing import Tuple

import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State
from igm.processes.ocean.ocean import ocean_depth
from igm.processes.ocean.seawater import freezing_temperature
from igm.processes.thk.masks import compute_grounded_mask, no_ocean_like

from ..temperature.utils import compute_E_cold_tf


def compute_shelf(
    cfg: DictConfig, state: State, T_pmp: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Columns in contact with the ocean and their basal enthalpy (J kg-1).

    Args:
        T_pmp: Pressure-melting temperature (K), 3-D with the base first.

    Returns:
        The bool mask of the columns in contact with the ocean (floating ice
        and ice-free ocean), and the Dirichlet value of their basal enthalpy.
    """
    cfg_physics = cfg.processes.iceflow.physics
    cfg_thermal = cfg.processes.enthalpy.thermal
    water_level = getattr(state, "water_level", None)
    if water_level is None:
        water_level = no_ocean_like(state.thk)

    rho_ratio = cfg_physics.water_density / cfg_physics.ice_density
    ocean = ~compute_grounded_mask(state.thk, state.topg, water_level, rho_ratio)

    T_base = T_pmp[0]
    if "ocean" in cfg.processes and hasattr(state, "ocean_salinity"):
        draft = ocean_depth(state.thk, state.lsurf, state.topg, water_level)
        T_freezing = freezing_temperature(cfg, state.ocean_salinity, draft)
        T_base = T_freezing + cfg_thermal.T_pmp_ref

    E_shelf = compute_E_cold_tf(T_base, T_pmp[0], cfg_thermal.T_ref, cfg_thermal.c_ice)
    return ocean, E_shelf
