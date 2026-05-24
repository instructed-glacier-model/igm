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

    if init:
        return Status.INIT
    elif state.it <= nbit_warmup:
        return Status.WARM_UP

    # Temporal lever (B) — dispatch on `adaptive_time.method`. Default
    # "none" → legacy `retrain_freq` modulo behaviour (no new params used,
    # old yamls work unchanged). Other methods use ONLY their own new
    # params (credit.kappa / credit.retrain_freq_max); they continue to
    # read top-level `retrain_freq` / `retrain_threshold` for their semantics.
    cfg_at = getattr(cfg_unified, "adaptive_time", None)
    method = (str(getattr(cfg_at, "method", "none")).lower()
              if cfg_at is not None else "none")

    if method == "credit":
        # Credit accumulator (`credit_tracker.py`). state._ct_credit is
        # updated each step in `update_credit_observer` BEFORE get_status.
        credit_cfg = getattr(cfg_at, "credit", None)
        credit = float(getattr(state, "_ct_credit", 0.0))
        steps = int(getattr(state, "_ct_steps_since_retrain", 0))
        kappa = float(getattr(credit_cfg, "kappa", 1.0e-3)) if credit_cfg is not None else 1.0e-3
        rfreq_max = int(getattr(credit_cfg, "retrain_freq_max", 100)) if credit_cfg is not None else 100
        if state.it > 0 and (credit >= kappa or steps >= rfreq_max):
            return Status.DEFAULT
        return Status.IDLE

    if method == "shift_distribution":
        # Hotelling-T² check on the input normaliser. The threshold is the
        # top-level `retrain_threshold`, consumed by `is_distribution_shifted`
        # in `solve_iceflow` (caller) and passed in as `distribution_shifted`.
        if state.it > 0 and distribution_shifted:
            return Status.DEFAULT
        return Status.IDLE

    # method == "none" (default) → legacy: retrain every `retrain_freq` steps.
    if retrain_freq > 0 and state.it > 0 and state.it % retrain_freq == 0:
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

    cfg_unified = cfg.processes.iceflow.unified
    optimizer = state.iceflow.optimizer
    set_optimizer_params = InterfaceOptimizers[optimizer.name].set_optimizer_params
    inputs = get_solver_inputs_from_state(cfg, state)
    mapping = getattr(optimizer.map, "network", optimizer.map)

    # Distribution-shift check (legacy Hotelling-T² on normaliser stats).
    distribution_shifted = (
        is_distribution_shifted(mapping, inputs, init, cfg_unified.retrain_threshold)
        if should_normalize(cfg) else False
    )

    # B (temporal): per-step credit observer. No-op unless adaptive_time activates it.
    update_credit_observer(cfg, state)

    status = get_status(cfg, state, init, distribution_shifted)
    do_solve = set_optimizer_params(cfg, status, optimizer)

    # C (spatial): adaptive_patching applies to INIT/WARM_UP/DEFAULT alike so the
    # optimizer graph is XLA-compiled once on a fixed (bs, ly, lx, C) shape.
    cfg_ap = cfg.processes.iceflow.unified.adaptive_patching
    use_adaptive = do_solve and bool(getattr(cfg_ap, "enabled", False))

    if do_solve and use_adaptive:
        training_inputs, bs = select_patches(cfg, state, inputs)
        n_train = int(training_inputs.shape[0])
        for start in range(0, n_train, bs):
            state.cost = optimizer.minimize(training_inputs[start:start + bs])
    elif do_solve:
        state.cost = optimizer.minimize(inputs)

    # B: log + reset credit accumulator if a retrain fired. No-op unless active.
    log_and_maybe_reset_credit_observer(cfg, state, status, do_solve)
