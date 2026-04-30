#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from omegaconf import DictConfig
import tensorflow as tf

from igm.common import State
from ..optimizers import InterfaceOptimizers, Status
from igm.processes.iceflow.utils.data_preprocessing import (
    fieldin_state_to_X,
    split_field_into_patches,
)
from igm.utils.math.precision import normalize_precision

from ..mappings.normalizer import is_distribution_shifted
from .patch_selection import select_patches


def get_status(
    cfg: DictConfig,
    state: State,
    init: bool = False,
    distribution_shifted: bool = False,
) -> Status:
    cfg_unified = cfg.processes.iceflow.unified
    nbit_warmup = cfg_unified.nbit_warmup
    retrain_freq = cfg_unified.retrain_freq

    if init:
        return Status.INIT
    elif state.it <= nbit_warmup:
        return Status.WARM_UP
    elif retrain_freq > 0 and state.it > 0 and state.it % retrain_freq == 0:
        return Status.DEFAULT
    elif state.it > 0 and distribution_shifted:
        print(
            "Retraining due to distribution shift!"
        )  # temporary measure to make debugging more clear for users
        return Status.DEFAULT
    # elif state.it > 0 and cfg_unified.mapping == "identity": # in theory, we might want to require solving at each time step for identity
    # return Status.DEFAULT

    return Status.IDLE


def get_solver_inputs_from_state(cfg: DictConfig, state: State) -> tf.Tensor:
    """Returns [N, ly, lx, C] non-overlapping patches, same strategy as emulated approach."""
    X = fieldin_state_to_X(cfg, state)
    framesizemax = cfg.processes.iceflow.unified.data_preparation.framesizemax
    dtype = normalize_precision(cfg.processes.iceflow.numerics.precision)
    return tf.cast(split_field_into_patches(X, framesizemax), dtype)


def should_normalize(cfg: DictConfig) -> bool:
    """Covers the 3 main situations in which the NN has its inputs normalized (or not).
    1. If we are using the identity mapping, do NOT NORMALIZE.
    2. If we are using any type of fixed or none transformation, DO NOT NORMALIZE.
    3. If we are using Sebastian's pretraining, do NOT NORMALIZE.
    """
    is_network = cfg.processes.iceflow.unified.mapping.lower() == "network"
    is_fixed_normalization = (
        cfg.processes.iceflow.unified.normalization.method.lower() in ("fixed", "none")
    )
    is_pretraining_SR = "pretraining" in cfg.processes.keys()
    is_pretrained_GJ = cfg.processes.iceflow.unified.network.pretrained

    if (
        not is_network
        or is_fixed_normalization
        or is_pretraining_SR
        or is_pretrained_GJ
    ):
        return False

    return True


def solve_iceflow(cfg: DictConfig, state: State, init: bool = False) -> None:

    # Get optimizer
    cfg_unified = cfg.processes.iceflow.unified
    optimizer = state.iceflow.optimizer

    # Set optimizer parameters
    set_optimizer_params = InterfaceOptimizers[optimizer.name].set_optimizer_params
    inputs = get_solver_inputs_from_state(cfg, state)
    mapping = getattr(optimizer.map, "network", optimizer.map)

    is_should_normalize = should_normalize(cfg)

    distribution_shifted = (
        is_distribution_shifted(mapping, inputs, init, cfg_unified.retrain_threshold)
        if is_should_normalize
        else False
    )

    status = get_status(cfg, state, init, distribution_shifted)
    do_solve = set_optimizer_params(cfg, status, optimizer)

    # Adaptive patch selection: filter patches by |dh/dt| if enabled
    if do_solve and status == Status.DEFAULT:
        inputs = select_patches(cfg, state, inputs)

    # Optimize and save cost
    if do_solve:

        state.cost = optimizer.minimize(inputs)
