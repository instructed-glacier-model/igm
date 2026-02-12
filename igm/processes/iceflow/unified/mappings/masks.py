#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from omegaconf import DictConfig
from typing import Any, Dict

from igm.common import State
from igm.utils.math.precision import normalize_precision


def mask_gr_friction(
    inputs: tf.Tensor,
    idx_C: int,
) -> tf.Tensor:
    C = inputs[..., idx_C : idx_C + 1]
    return tf.cast(C > 0.0, dtype=inputs.dtype)


def mask_gr_flotation(
    inputs: tf.Tensor,
    idx_thk: int,
    rho_ratio: tf.Tensor,
    topg: tf.Tensor,
    thk_c: tf.Tensor,
) -> tf.Tensor:
    thk = inputs[..., idx_thk : idx_thk + 1]
    phi = thk + rho_ratio * topg
    return tf.sigmoid(phi / thk_c)


def interface_friction(
    cfg: DictConfig, state: State, mask_gr_cfg: DictConfig
) -> Dict[str, Any]:
    inputs = list(cfg.processes.iceflow.unified.inputs)
    field = mask_gr_cfg.get("field", "slidingco")
    return {"idx_C": inputs.index(field)}


def interface_flotation(
    cfg: DictConfig, state: State, mask_gr_cfg: DictConfig
) -> Dict[str, Any]:
    inputs = list(cfg.processes.iceflow.unified.inputs)
    cfg_physics = cfg.processes.iceflow.physics
    precision = cfg.processes.iceflow.numerics.precision
    dtype = normalize_precision(precision)

    # topg from state: (Ny, Nx) → (1, Ny, Nx, 1) for broadcasting
    topg = tf.constant(state.topg, dtype=dtype)
    topg = tf.reshape(topg, [1, topg.shape[0], topg.shape[1], 1])

    return {
        "idx_thk": inputs.index("thk"),
        "topg": topg,
        "rho_ratio": cfg_physics.water_density / cfg_physics.ice_density,
        "thk_c": mask_gr_cfg.smoothing,
    }


masks_gr = {
    "friction": mask_gr_friction,
    "flotation": mask_gr_flotation,
}

interfaces_mask_gr = {
    "friction": interface_friction,
    "flotation": interface_flotation,
}
