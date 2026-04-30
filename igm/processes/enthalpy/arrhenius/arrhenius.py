#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State

from .utils import compute_arrhenius_3d


def compute_arrhenius(
    cfg: DictConfig, state: State, T: tf.Tensor, omega: tf.Tensor
) -> None:
    """
    Compute the vertically-averaged Arrhenius factor for ice flow.

    Calculates the 3D Arrhenius factor A scaled by an enhancement factor, then performs
    vertical averaging over B = A^(-1/n) (related to viscosity) rather than over A directly.
    The final result is converted back: A_avg = (Σ B_i * w_i)^(-n).

    This approach is physically motivated since B is proportional to viscosity,
    and averaging viscosity is more appropriate than averaging the rate factor.

    Args:
        T: Temperature field (K).
        omega: Water content fraction (-).

    Updates state.arrhenius (MPa^-n yr^-1).
    """
    cfg_physics = cfg.processes.iceflow.physics
    E = cfg_physics.enhancement_factor
    n = cfg_physics.exp_glen
    weights = state.iceflow.discr_v.enthalpy.weights

    # Compute 3D Arrhenius factor with enhancement
    A = E * compute_arrhenius_3d(cfg, state, T, omega)

    # Average over B = A^(-1/n) and convert back
    B = tf.pow(A, -1.0 / n)
    B_avg = tf.reduce_sum(B * weights, axis=0)
    state.arrhenius = tf.pow(B_avg, -n)
