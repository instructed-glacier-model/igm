#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Tuple
from omegaconf import DictConfig

from igm.common import State

from .utils import compute_strain_heat, compute_friction_heat


def compute_dissipation(cfg: DictConfig, state: State) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Compute heat sources from viscous dissipation and basal friction.

    Calculates the volumetric strain heating rate throughout the ice column
    and the areal frictional heating rate at the bed.

    Returns:
        Tuple of strain_heat (W m^-3) and friction_heat (W m^-2).
        Neither is stored on state, keeping GPU memory lean.
    """
    strain_heat = compute_strain_heat(cfg, state)
    friction_heat = compute_friction_heat(cfg, state)
    return strain_heat, friction_heat
