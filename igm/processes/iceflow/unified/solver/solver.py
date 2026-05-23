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
from .patch_selection import (
    select_patches,
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

    if init:
        return Status.INIT
    elif state.it <= nbit_warmup:
        return Status.WARM_UP

    # STAGE 2 of adaptive_unique_strategy: when adaptive_training.enabled is true,
    # the credit accumulator replaces retrain_freq + retrain_threshold as the
    # DEFAULT trigger. See clever-patch/adaptive_unique_strategy.md §7.
    cfg_at = getattr(cfg_unified, "adaptive_training", None)
    if cfg_at is not None and bool(getattr(cfg_at, "enabled", False)):
        credit = float(getattr(state, "_ct_credit", 0.0))
        steps = int(getattr(state, "_ct_steps_since_retrain", 0))
        kappa = float(getattr(cfg_at, "kappa", 1.0e-3))
        rfreq_max = int(getattr(cfg_at, "retrain_freq_max", 100))
        if state.it > 0 and (credit >= kappa or steps >= rfreq_max):
            return Status.DEFAULT
        return Status.IDLE

    # Legacy: retrain_freq-based + distribution-shifted trigger.
    if retrain_freq > 0 and state.it > 0 and state.it % retrain_freq == 0:
        return Status.DEFAULT
    elif state.it > 0 and distribution_shifted:
        print(
            "Retraining due to distribution shift!"
        )  # temporary measure to make debugging more clear for users
        return Status.DEFAULT

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

    # Per-step credit observer (STAGE 1 of adaptive_unique_strategy migration).
    # Updates state._ct_* attributes and accumulates Δ. Observation-only at
    # this stage — get_status below is untouched, retrain decisions still
    # come from retrain_freq. Controlled by
    # cfg.processes.iceflow.unified.adaptive_training.enabled_observation.
    update_credit_observer(cfg, state)

    status = get_status(cfg, state, init, distribution_shifted)
    do_solve = set_optimizer_params(cfg, status, optimizer)

    # Adaptive patch selection — applies to INIT, WARM_UP, and DEFAULT alike so
    # that the optimizer graph is JIT-compiled on a single (bs, ly, lx, C) shape
    # for the whole run. Prior to 2026-05-20 INIT used the splitter's framesizemax-
    # tiled tensor while DEFAULT used patch_size-tiled patches, which triggered an
    # expensive XLA recompile at the first DEFAULT step on small grids and could
    # hang for >10 minutes when the shape change was drastic (e.g. 1×244×179 →
    # 75×30×29). Routing INIT through select_patches uses the same patch
    # geometry and (bs, ly, lx, C) shape as the subsequent DEFAULT calls.
    cfg_ap = cfg.processes.iceflow.unified.adaptive_patching
    use_adaptive = do_solve and bool(getattr(cfg_ap, "enabled", False))

    # Optimize and save cost
    if do_solve and use_adaptive:
        # `select_patches` returns the full training tensor of shape
        # [N_train, ly, lx, C] (already shuffled / scored / sampled per
        # the new 4-step pipeline) plus the fixed batch size `bs`. We
        # loop over N_train/bs slices, each of fixed shape [bs, ly, lx, C],
        # so the optimizer graph compiles once.
        training_inputs, bs = select_patches(cfg, state, inputs)
        n_train = int(training_inputs.shape[0])
        for start in range(0, n_train, bs):
            batch = training_inputs[start:start + bs]
            state.cost = optimizer.minimize(batch)
    elif do_solve:
        state.cost = optimizer.minimize(inputs)

    # Per-step credit observer — log line + reset accumulators if retrain fired.
    # No-op when adaptive_training.enabled_observation is false.
    log_and_maybe_reset_credit_observer(cfg, state, status, do_solve)
