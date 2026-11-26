#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State
from igm.processes.iceflow.energy.components.sliding import SlidingComponents
from igm.processes.thk.masks import weights_gr


def get_inputs_from_state(cfg: DictConfig, state: State) -> tf.Tensor:
    """Stack the unified inputs read from ``state`` into a [H, W, C] tensor.

    With ``physics.sliding.use_mask_gr`` and ``use_subgrid_gl``, the
    ``friction_inputs`` of the sliding law are multiplied by the grounded
    weights of the nodes (``weights_gr``, from ``state.topg``); the inputs
    keep their names and number.
    """
    cfg_physics = cfg.processes.iceflow.physics
    cfg_sliding = cfg_physics.sliding
    fields = {
        name: getattr(state, name) for name in cfg.processes.iceflow.unified.inputs
    }
    if cfg_sliding.use_mask_gr and cfg_sliding.use_subgrid_gl:
        weight = weights_gr(
            state.thk,
            state.topg,
            state.water_level,
            cfg_physics.water_density / cfg_physics.ice_density,
        )
        for name in SlidingComponents[cfg_sliding.law].friction_inputs:
            if name in fields:
                fields[name] = fields[name] * weight
    return tf.stack(list(fields.values()), axis=-1)
