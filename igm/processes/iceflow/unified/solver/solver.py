#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from typing import Any, Dict, Tuple
import tensorflow as tf
import os
import warnings
import igm
from ..mappings import Mappings
from ..optimizers import Optimizer

from igm.processes.iceflow.utils.data_preprocessing import (
    match_fieldin_dimensions,
    split_into_patches,
    pertubate_X,
)

from igm.processes.iceflow.utils.data_preprocessing import (
    fieldin_to_X_2d,
    fieldin_to_X_3d,
)


def get_inputs_from_state(cfg, state) -> tf.Tensor:

    # TO DO: full data prep?

    cfg_physics = cfg.processes.iceflow.physics
    cfg_unified = cfg.processes.iceflow.unified

    if len(cfg_unified.inputs) != len(cfg_unified.inputs_scales):
        raise ValueError("❌ The inputs and input scales should have the same size.")

    inputs = [vars(state)[input] for input in cfg_unified.inputs]

    inputs = [input / scale for input, scale in zip(inputs, cfg_unified.inputs_scales)]

    if cfg_physics.dim_arrhenius == 3:
        inputs = match_fieldin_dimensions(inputs)
        inputs = fieldin_to_X_3d(cfg_physics.dim_arrhenius, inputs)
    elif cfg_physics.dim_arrhenius == 2:
        inputs = tf.stack(inputs, axis=-1)
        inputs = fieldin_to_X_2d(inputs)
    else:
        raise ValueError("❌ Invalid Arrhenius dimension value.")

    if cfg_unified.perturbate:
        inputs = pertubate_X(cfg, inputs)

    inputs = split_into_patches(
        inputs,
        cfg.processes.iceflow.unified.framesizemax,
        cfg.processes.iceflow.unified.split_patch_method,
    )

    return inputs


@tf.function(jit_compile=False)
def solver_iceflow(optimizer: Optimizer, inputs: tf.Tensor) -> tf.Tensor:
    return optimizer.minimize(inputs)


def solve_iceflow(cfg, state, init: bool = False):

    cfg_unified = cfg.processes.iceflow.unified

    # TO DO: nbit and lr should be set within the optimizer
    if init:
        nbit = cfg_unified.nbit_init
        lr = cfg_unified.lr_init
    else:
        warm_up = int(state.it <= cfg_unified.warm_up_it)
        if warm_up:
            nbit = cfg_unified.nbit_init
            lr = cfg_unified.lr_init
        else:
            nbit = cfg_unified.nbit
            lr = cfg_unified.lr
    state.optimizer.lr = lr

    # Get inputs for mapping
    inputs = get_inputs_from_state(cfg, state)

    # Get optimizer
    optimizer = state.optimizer

    # Set its parameters
    # optimizer.update_parameters()

    # Optimize
    cost = optimizer.minimize(inputs)

    # Save cost
    state.cost = cost
