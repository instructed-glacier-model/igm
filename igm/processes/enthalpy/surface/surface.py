#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from omegaconf import DictConfig
from typing import Tuple

from igm.common import State

from .utils import compute_T_s_tf
from ..temperature.utils import compute_E_cold_tf


def compute_surface(cfg: DictConfig, state: State) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Compute surface enthalpy boundary condition.

    Derives the ice surface temperature from air temperature (with offset),
    capped at the pressure melting point, and computes the corresponding
    surface enthalpy.

    Returns E_s (J kg^-1).
    """
    cfg_thermal = cfg.processes.enthalpy.thermal
    cfg_surface = cfg.processes.enthalpy.surface

    T_offset = cfg_surface.T_offset
    c_ice = cfg_thermal.c_ice
    T_pmp_ref = cfg_thermal.T_pmp_ref
    T_ref = cfg_thermal.T_ref

    T_s = compute_T_s_tf(state.air_temp, T_offset, T_pmp_ref)
    E_s = compute_E_cold_tf(T_s, T_pmp_ref, T_ref, c_ice)

    return E_s, T_s
