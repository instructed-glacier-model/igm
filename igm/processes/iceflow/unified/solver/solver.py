#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State

from ..optimizers import InterfaceOptimizers, Status
from igm.processes.iceflow.utils.data_preprocessing import (
    fieldin_state_to_X,
    split_field_into_patches,
)
from igm.processes.iceflow.unified.mappings.normalizer import (
    IdentityNormalizer,
    NetworkNormalizer,
)


def get_status(cfg: DictConfig, state: State, init: bool = False) -> Status:

    cfg_unified = cfg.processes.iceflow.unified
    nbit_warmup = cfg_unified.nbit_warmup
    retrain_freq = cfg_unified.retrain_freq

    if init:
        status = Status.INIT
    elif state.it <= nbit_warmup:
        status = Status.WARM_UP
    elif retrain_freq > 0 and state.it > 0 and state.it % retrain_freq == 0:
        status = Status.DEFAULT
    else:
        status = Status.IDLE

    return status


def get_solver_inputs_from_state(cfg: DictConfig, state: State) -> tf.Tensor:
    """Returns [N, ly, lx, C] non-overlapping patches, same strategy as emulated approach."""
    X = fieldin_state_to_X(cfg, state)

    framesizemax = cfg.processes.iceflow.unified.data_preparation.framesizemax
    return split_field_into_patches(X, framesizemax)


def solve_iceflow(cfg: DictConfig, state: State, init: bool = False) -> None:

    # Get status: should we optimize again?
    status = get_status(cfg, state, init)

    # Get optimizer
    optimizer = state.iceflow.optimizer

    # Set optimizer parameters
    set_optimizer_params = InterfaceOptimizers[optimizer.name].set_optimizer_params
    do_solve = set_optimizer_params(cfg, status, optimizer)

    # Optimize and save cost
    if do_solve:
        inputs = get_solver_inputs_from_state(cfg, state)

        # TODO: check this, and probably do it elsewhere
        optimizer.map.update_normalizer(inputs)

        state.cost = optimizer.minimize(inputs)
