#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from omegaconf import DictConfig
import tensorflow as tf
from typing import List, Tuple

from igm.common import State

from ..optimizers import InterfaceOptimizers, Status
from igm.processes.iceflow.utils.data_preprocessing import (
    fieldin_state_to_X,
)

from igm.utils.math.precision import normalize_precision
from igm.processes.iceflow.unified.mappings import Mappings
from igm.processes.iceflow.emulate.utils.normalizations import FixedAffineLayer
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
    elif retrain_freq > 0 and state.it > 0 and state.it % retrain_freq == 0:
        status = Status.DEFAULT
    else:
        status = Status.IDLE

    return status


def get_solver_inputs_from_state(cfg: DictConfig, state: State) -> tf.Tensor:
    """Returns [N, H, W, C] patches (sampler handles batching/augmentation)."""
    X = fieldin_state_to_X(cfg, state)

    # dtype = normalize_precision(cfg.processes.iceflow.numerics.precision) # ! confirm double precision is working properly

    # Create patches using the patching object
    patches = state.iceflow.patching.generate_patches(X)

    return patches


def solve_iceflow(cfg: DictConfig, state: State, init: bool = False) -> None:

    # Get status: should we optimize again?
    status = get_status(cfg, state, init)

    # Get optimizer
    optimizer = state.iceflow.optimizer

    # Set optimizer parameters
    set_optimizer_params = InterfaceOptimizers[optimizer.name].set_optimizer_params
    do_solve = set_optimizer_params(cfg, status, optimizer)

    # TODO: Clean up this if statement
    # AND...  include the compute stats and set stats in the status function instead (in general, we should think about this whole function)
    if hasattr(optimizer.map, "network"):
        mapping = optimizer.map.network
    else:
        mapping = optimizer.map

    # Optimize and save cost
    if do_solve:
        inputs = get_solver_inputs_from_state(cfg, state)

        if ( # this became a bit of a mess now... 
            cfg.processes.iceflow.unified.mapping == "network" # per thomas' request
            and (
                (not cfg.processes.iceflow.unified.network.pretrained)
                or cfg.processes.iceflow.unified.network.old_format # as discussed with brandon
            )
        ):
            # # Compute and set stats can be linked inside the class, so we can get rid of the liskov violation here (when means in one transformation is not means in another)
            means, variances = mapping.input_normalizer.compute_stats(inputs)
            mapping.input_normalizer.set_stats(means, variances)

        state.cost = optimizer.minimize(inputs)
