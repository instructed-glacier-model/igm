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
    """Covers the main situations in which the NN has its inputs normalized (or not).
    1. If we are using the identity mapping, do NOT NORMALIZE.
    2. If we are using any type of fixed or none transformation, DO NOT NORMALIZE.
    3. If we are using Sebastian's pretraining, do NOT NORMALIZE.
    4. For composite mappings, normalize if any sub-network is not pretrained and uses adaptive normalization.
    """
    mapping_name = cfg.processes.iceflow.unified.mapping.lower()
    is_pretraining_SR = "pretraining" in cfg.processes.keys()
    is_fixed_normalization = (
        cfg.processes.iceflow.unified.normalization.method.lower() in ("fixed", "none")
    )

    if is_pretraining_SR or is_fixed_normalization:
        return False

    if mapping_name == "network":
        is_pretrained_GJ = cfg.processes.iceflow.unified.network.pretrained
        return not is_pretrained_GJ

    if mapping_name == "composite":
        cfg_composite = cfg.processes.iceflow.unified.composite
        for sub_key in ("gr", "fl"):
            sub = getattr(cfg_composite, sub_key, None)
            if sub is None:
                continue
            net = getattr(sub, "network", None)
            if net is None or getattr(net, "pretrained", False):
                continue
            norm = getattr(sub, "normalization", None)
            if norm and getattr(norm, "method", "").lower() == "adaptive":
                return True
        return False

    return False


def get_networks_for_normalization(optimizer_map) -> list:
    """Returns the network model(s) whose normalizer stats need updating.

    For a single MappingNetwork, returns [network].
    For a MappingComposite, returns [gr.network, fl.network].
    """
    if hasattr(optimizer_map, "mapping_gr") and hasattr(optimizer_map, "mapping_fl"):
        networks = []
        for sub_map in (optimizer_map.mapping_gr, optimizer_map.mapping_fl):
            net = getattr(sub_map, "network", None)
            if net is not None:
                networks.append(net)
        return networks
    return [getattr(optimizer_map, "network", optimizer_map)]


def solve_iceflow(cfg: DictConfig, state: State, init: bool = False) -> None:

    # Get optimizer
    cfg_unified = cfg.processes.iceflow.unified
    optimizer = state.iceflow.optimizer

    # Set optimizer parameters
    set_optimizer_params = InterfaceOptimizers[optimizer.name].set_optimizer_params
    inputs = get_solver_inputs_from_state(cfg, state)

    is_should_normalize = should_normalize(cfg)

    distribution_shifted = False
    if is_should_normalize:
        for net in get_networks_for_normalization(optimizer.map):
            shifted = is_distribution_shifted(net, inputs, init, cfg_unified.retrain_threshold)
            distribution_shifted = distribution_shifted or shifted

    status = get_status(cfg, state, init, distribution_shifted)
    do_solve = set_optimizer_params(cfg, status, optimizer)

    # Adaptive patch selection: filter patches by |dh/dt| if enabled
    if do_solve and status == Status.DEFAULT:
        inputs = select_patches(cfg, state, inputs)

    # Optimize and save cost
    if do_solve:

        state.cost = optimizer.minimize(inputs)
