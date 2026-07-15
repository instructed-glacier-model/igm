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
from .credit_tracker import (
    update_credit_observer,
    log_and_maybe_reset_credit_observer,
)


def get_status(
    cfg: DictConfig,
    state: State,
    init: bool = False,
    distribution_shifted: bool = False,
) -> Status:
    cfg_unified = cfg.processes.iceflow.unified
    nbit_warmup = cfg_unified.nbit_warmup
    retrain_freq = cfg_unified.retrain_freq

    # adaptive_time decides WHEN to retrain: "none" = retrain_freq only,
    # "shift_distribution" = retrain_freq + distribution-shift early trigger,
    # "credit" = per-step input-change accumulator (see credit_tracker.py).
    cfg_at = getattr(cfg_unified, "adaptive_time", None)
    method = str(getattr(cfg_at, "method", "none")).lower() if cfg_at is not None else "none"

    if init:
        return Status.INIT
    elif state.it <= nbit_warmup:
        return Status.WARM_UP
    elif method == "credit":
        # retrain once the credit accumulated by update_credit_observer
        # exceeds kappa, or after retrain_freq_max steps (safety ceiling)
        if state.it > 0 and (
            float(getattr(state, "_ct_credit", 0.0)) >= cfg_at.credit.kappa
            or int(getattr(state, "_ct_steps_since_retrain", 0)) >= cfg_at.credit.retrain_freq_max
        ):
            return Status.DEFAULT
    elif retrain_freq > 0 and state.it > 0 and state.it % retrain_freq == 0:
        return Status.DEFAULT
    elif method == "shift_distribution" and state.it > 0 and distribution_shifted:
        print(
            "Retraining due to distribution shift!"
        )  # temporary measure to make debugging more clear for users
        return Status.DEFAULT
    # elif state.it > 0 and cfg_unified.mapping == "identity": # in theory, we might want to require solving at each time step for identity
    # return Status.DEFAULT

    return Status.IDLE


def get_solver_inputs_from_state(cfg: DictConfig, state: State) -> tf.Tensor:
    """Returns [N, ly, lx, C] non-overlapping patches, same strategy as emulated approach."""
    X = fieldin_state_to_X(state, cfg.processes.iceflow.unified.inputs)
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

    # Per-step credit observer (no-op unless adaptive_time activates it)
    update_credit_observer(cfg, state)

    status = get_status(cfg, state, init, distribution_shifted)
    do_solve = set_optimizer_params(cfg, status, optimizer)

    # Adaptive patch selection: train on a subset of patches where the inputs
    # change the most. Applied to INIT/WARM_UP/DEFAULT alike so the optimizer
    # graph is XLA-compiled once on a fixed (bs, ly, lx, C) shape.
    use_adaptive = bool(getattr(cfg_unified.adaptive_patching, "enabled", False))

    # Optimize and save cost
    if do_solve and use_adaptive:
        training_inputs, bs = select_patches(cfg, state, inputs)
        for start in range(0, int(training_inputs.shape[0]), bs):
            state.cost = optimizer.minimize(training_inputs[start:start + bs])
    elif do_solve:
        state.cost = optimizer.minimize(inputs)

    # Log + reset the credit accumulator if a retrain fired (no-op unless active)
    log_and_maybe_reset_credit_observer(cfg, state, status, do_solve)
