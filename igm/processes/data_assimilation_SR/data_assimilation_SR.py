#!/usr/bin/env python3
# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from __future__ import annotations

import keras
import tensorflow as tf

from igm.processes.iceflow.unified.halt import Halt
from igm.processes.iceflow.unified.halt.criteria import Criteria
from igm.processes.iceflow.unified.halt.metrics import Metrics
from igm.processes.iceflow.unified.mappings.data_assimilation import MappingDataAssimilation
from igm.processes.iceflow.unified.mappings.interfaces.data_assimilation import InterfaceDataAssimilation
from igm.processes.iceflow.unified.optimizers.interfaces import InterfaceLBFGS, InterfaceSPG
from igm.processes.iceflow.unified.optimizers.lbfgs_DA import OptimizerLBFGSBoundsDA
from igm.utils.math.precision import normalize_precision

from igm.processes.iceflow.unified.optimizers.spectral_projected_gradient import (
    OptimizerSpectralProjectedGradient,
)

from .phase_runner import (
    DataAssimilationRuntime,
    build_cost_and_objective,
    reset_da_run_state,
    run_da_phase,
)
from .retraining import (
    initialize_retraining,
    reset_retraining_run_state,
    run_retraining_phase,
)
from .utils import _initialize_inverted_fields

def _require_supported_keras_version() -> None:
    major = int(str(getattr(keras, "__version__", "0")).split(".", 1)[0])
    if major < 3:
        raise RuntimeError(
            f"data_assimilation_SR requires Keras 3 or newer. "
            f"Detected keras == {keras.__version__}."
        )

def _build_halt(cfg) -> Halt:
    cfg_da = cfg.processes.data_assimilation_SR

    log_burst_crit = Criteria["log_burst"](
        metric=Metrics["cost"](),
        dtype=cfg.processes.iceflow.numerics.precision,
        patience=cfg_da.optimization.minimizer_patience,
        log_tol=1.0e-4,
        burst_log_tol=9.53e-2,
        patience_growth= 1.5,
        max_patience= 5 * cfg_da.optimization.minimizer_patience,
        min_iter= 0,
        cost_floor=1.0e-6,
    )

    grad_crit = Criteria["abs_tol"](
        metric=Metrics["grad_theta_norm"](),
        tol=1.0e-4,
        ord="id",
    )

    return Halt(
        crit_success=[log_burst_crit, grad_crit],
        crit_failure=[],
        freq=1,
        dtype=cfg.processes.iceflow.numerics.precision,
        success_mode="all",
    )


def data_assimilation_initialize(cfg, state) -> None:
    _require_supported_keras_version()

    cfg_da = cfg.processes.data_assimilation_SR
    dtype = normalize_precision(cfg.processes.iceflow.numerics.precision)
    retrain_iter = int(cfg_da.optimization.retrain_iter)

    _initialize_inverted_fields(cfg, state, dtype)

    da_map = MappingDataAssimilation(
        **InterfaceDataAssimilation.get_mapping_args(cfg, state)
    )
    cost_fn, objective = build_cost_and_objective(cfg, state, da_map)

    optimizer_args = InterfaceLBFGS.get_optimizer_args(cfg, cost_fn, da_map)
    optimizer_args["halt"] = _build_halt(cfg)
    optimizer = OptimizerLBFGSBoundsDA(**optimizer_args)

    # optimizer_args = InterfaceSPG.get_optimizer_args(cfg, cost_fn, da_map)
    # optimizer_args["halt"] = _build_halt(cfg)
    # optimizer = OptimizerSpectralProjectedGradient(**optimizer_args)

    retraining = None
    if retrain_iter > 0:
        retraining = initialize_retraining(cfg, state, da_map)

    da = DataAssimilationRuntime(
        map=da_map,
        opt=optimizer,
        cost_fn=cost_fn,
        objective=objective,
        out_freq=int(cfg_da.output.freq),
        retrain_iter=retrain_iter,
        retraining=retraining,
    )
    state.data_assimilation = da


def initialize(cfg, state) -> None:
    data_assimilation_initialize(cfg, state)


def update(cfg, state) -> None:
    da = state.data_assimilation
    reset_da_run_state(da)

    if da.retraining is not None:
        reset_retraining_run_state(da.retraining)

    da.retrain_iter_num = 0
    run_da_phase(cfg, state, da)

    for retrain_iter_num in range(1, da.retrain_iter + 1):
        da.retrain_iter_num = retrain_iter_num
        run_retraining_phase(cfg, state, da)
        run_da_phase(cfg, state, da)


def finalize(cfg, state) -> None:
    pass
