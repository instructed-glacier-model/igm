#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from omegaconf import DictConfig
import tensorflow as tf

from igm.common.core import State
from ..optimizers import Interfaces, Status

from igm.processes.iceflow.utils.data_preprocessing import (
    match_fieldin_dimensions,
    split_into_patches_X,
    pertubate_X,
)

from igm.processes.iceflow.utils.data_preprocessing import (
    fieldin_to_X_2d,
    fieldin_to_X_3d,
)


def get_status(cfg: DictConfig, state: State, init: bool = False) -> Status:

    cfg_unified = cfg.processes.iceflow.unified
    warm_up_it = cfg_unified.warm_up_it
    retrain_freq = cfg_unified.retrain_freq

    if init:
        status = Status.INIT
    elif state.it <= warm_up_it:
        status = Status.WARM_UP
    elif retrain_freq > 0 and state.it > 0 and state.it % retrain_freq == 0:
        status = Status.DEFAULT
    else:
        status = Status.IDLE

    return status


def get_inputs_from_state(cfg: DictConfig, state: State) -> tf.Tensor:

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

    inputs = split_into_patches_X(
        inputs,
        cfg_unified.framesizemax,
        cfg_unified.split_patch_method,
    )

    return inputs


def solve_iceflow(cfg: DictConfig, state: State, init: bool = False) -> None:

    # Get status: should we optimize again?
    status = get_status(cfg, state, init)

    # Get optimizer
    optimizer = state.optimizer

    # Set optimizer parameters
    set_optimizer_params = Interfaces[optimizer.name].set_optimizer_params
    do_solve = set_optimizer_params(cfg, status, optimizer)

    # Optimize and save cost
    if do_solve:
        inputs = get_inputs_from_state(cfg, state)
        state.cost = optimizer.minimize(inputs)
