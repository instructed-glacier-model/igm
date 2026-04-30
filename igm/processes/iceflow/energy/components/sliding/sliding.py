#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from typing import Any, Dict
from omegaconf import DictConfig
import tensorflow as tf

from ..energy import EnergyComponent


class SlidingComponent(EnergyComponent):
    """Energy component representing frictional energy."""

    pass


def mask_gr(h: tf.Tensor, topg: tf.Tensor, rho_ratio: tf.Tensor) -> tf.Tensor:
    """Compute grounding mask: 1 where grounded, 0 where floating."""
    phi = h + rho_ratio * topg
    return tf.cast(phi > 0.0, dtype=h.dtype)


def get_sliding_params_args(cfg: DictConfig) -> Dict[str, Any]:
    """Extract friction parameters from configuration."""

    cfg_physics = cfg.processes.iceflow.physics

    law = cfg_physics.sliding.law

    args = dict(cfg_physics.sliding[law])
    args["rho_ratio"] = cfg_physics.water_density / cfg_physics.ice_density
    args["use_mask_gr"] = cfg_physics.sliding.use_mask_gr

    return args
