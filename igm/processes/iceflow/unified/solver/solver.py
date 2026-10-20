#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import glob
import math
import os

from omegaconf import DictConfig
import tensorflow as tf

from igm.common import State
from ..optimizers import InterfaceOptimizers, Status
from igm.processes.iceflow.utils.data_preprocessing import split_field_into_patches
from igm.utils.math.precision import normalize_precision

from ..mappings.normalizer import is_distribution_shifted
from ..utils import get_inputs_from_state
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
    method = (
        str(getattr(cfg_at, "method", "none")).lower() if cfg_at is not None else "none"
    )

    if init:
        return Status.INIT
    elif state.it <= nbit_warmup:
        return Status.WARM_UP
    elif method == "credit":
        # retrain once the credit accumulated by update_credit_observer
        # exceeds kappa, or after retrain_freq_max steps (safety ceiling)
        if state.it > 0 and (
            float(getattr(state, "_ct_credit", 0.0)) >= cfg_at.credit.kappa
            or int(getattr(state, "_ct_steps_since_retrain", 0))
            >= cfg_at.credit.retrain_freq_max
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
    X = get_inputs_from_state(cfg, state)
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
    do_solve = set_optimizer_params(cfg, status, optimizer, getattr(state, "t", None))
    if do_solve and optimizer.name == "cg_newton":
        optimizer.cold_restart(getattr(state, "t", None), init)

    # Adaptive patch selection: train on a subset of patches where the inputs
    # change the most. Applied to INIT/WARM_UP/DEFAULT alike so the optimizer
    # graph is XLA-compiled once on a fixed (bs, ly, lx, C) shape.
    use_adaptive = bool(getattr(cfg_unified.adaptive_patching, "enabled", False))

    # Retrain trust region (see iceflow.yaml `retrain_guard`): snapshot the
    # weights and the physics cost on the CURRENT inputs before a regular
    # retrain, so a step that worsens the cost can be rejected outright.
    guard_cfg = getattr(cfg_unified, "retrain_guard", None)
    use_guard = (
        do_solve
        and status == Status.DEFAULT
        and guard_cfg is not None
        and bool(guard_cfg.enabled)
        and cfg_unified.mapping == "network"
    )
    rejected = False
    if use_guard:
        guard_model = state.iceflow_model
        weights_before = guard_model.get_weights()
        cost_before = float(optimizer._get_cost(inputs).numpy())

    # Optimize and save cost
    if do_solve and use_adaptive:
        training_inputs, bs = select_patches(cfg, state, inputs)
        for start in range(0, int(training_inputs.shape[0]), bs):
            state.cost = optimizer.minimize(training_inputs[start : start + bs])
    elif do_solve:
        state.cost = optimizer.minimize(inputs)

    if use_guard:
        cost_after = float(optimizer._get_cost(inputs).numpy())
        rejected = (not math.isfinite(cost_after)) or (
            cost_after
            > cost_before + float(guard_cfg.slack) * (abs(cost_before) + 1e-30)
        )
        if rejected:
            guard_model.set_weights(weights_before)
            state.iceflow.retrains_rejected = (
                getattr(state.iceflow, "retrains_rejected", 0) + 1
            )
            print(
                f"retrain guard t={float(getattr(state, 't', 0.0)):.1f}: "
                f"cost {cost_before:.6e} -> {cost_after:.6e} REJECTED, weights "
                f"rolled back (total rejected {state.iceflow.retrains_rejected})"
            )
        elif bool(getattr(guard_cfg, "verbose", False)):
            print(
                f"retrain guard t={float(getattr(state, 't', 0.0)):.1f}: "
                f"cost {cost_before:.6e} -> {cost_after:.6e} ok"
            )

    # Rolling weight checkpoints (see iceflow.yaml `checkpoint`): every `freq`
    # accepted retrains, save the weights and prune to the `keep` most recent.
    ckpt_cfg = getattr(cfg_unified, "checkpoint", None)
    if (
        do_solve
        and status == Status.DEFAULT
        and not rejected
        and ckpt_cfg is not None
        and bool(ckpt_cfg.enabled)
        and cfg_unified.mapping == "network"
    ):
        n_accepted = getattr(state.iceflow, "retrains_accepted", 0) + 1
        state.iceflow.retrains_accepted = n_accepted
        if n_accepted % int(ckpt_cfg.freq) == 0:
            os.makedirs(str(ckpt_cfg.dir), exist_ok=True)
            path = os.path.join(
                str(ckpt_cfg.dir),
                f"iceflow_t{float(getattr(state, 't', 0.0)):.1f}.weights.h5",
            )
            state.iceflow_model.save_weights(path)
            saved = sorted(
                glob.glob(os.path.join(str(ckpt_cfg.dir), "iceflow_t*.weights.h5")),
                key=os.path.getmtime,
            )
            for old_path in saved[: -int(ckpt_cfg.keep)]:
                os.remove(old_path)

    # Log + reset the credit accumulator if a retrain fired (no-op unless active)
    log_and_maybe_reset_credit_observer(cfg, state, status, do_solve)
