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


def mask_x_split(
    inputs: tf.Tensor,
    x_mask: tf.Tensor,
) -> tf.Tensor:
    """Hard binary mask: 1 where x < split_x, 0 elsewhere.

    x_mask is precomputed by interface_x_split and broadcast-compatible
    with inputs shape (batch, Ny, Nx, 1).
    """
    return tf.cast(x_mask, dtype=inputs.dtype)


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


def interface_x_split(
    cfg: DictConfig, state: State, mask_gr_cfg: DictConfig
) -> Dict[str, Any]:
    """Precompute a spatial mask splitting the domain at x = split_x.

    Parameters
    ----------
    split_x : float
        x-coordinate of the split in metres (e.g. 400 km → 400000).
    smoothing : float, optional
        Transition half-width in metres. When set, the mask is
        sigmoid((split_x - x) / smoothing), tapering from 1 (left
        subdomain) to 0 (right subdomain) over ~2×smoothing metres.
        When absent or 0, a hard binary step is used instead.
    """
    precision = cfg.processes.iceflow.numerics.precision
    dtype = normalize_precision(precision)

    split_x = float(mask_gr_cfg.split_x)
    smoothing = float(mask_gr_cfg.get("smoothing", 0.0))

    # state.x is (Nx,) in metres; reshape to (1, 1, Nx, 1) for broadcasting
    # against inputs (batch, Ny, Nx, channels).
    x = tf.constant(state.x, dtype=dtype)
    x = tf.reshape(x, [1, 1, -1, 1])

    if smoothing > 0.0:
        x_mask = tf.sigmoid((split_x - x) / smoothing)
    else:
        x_mask = tf.cast(x < split_x, dtype=dtype)

    return {"x_mask": x_mask}


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
    "x_split": mask_x_split,
}

interfaces_mask_gr = {
    "friction": interface_friction,
    "flotation": interface_flotation,
    "x_split": interface_x_split,
}
