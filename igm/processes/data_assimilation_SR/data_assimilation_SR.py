#!/usr/bin/env python3
# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from __future__ import annotations

from igm.processes.iceflow.unified.halt import Halt
from igm.processes.iceflow.unified.halt.criteria import Criteria
from igm.processes.iceflow.unified.halt.metrics import Metrics
from igm.processes.iceflow.unified.mappings.data_assimilation import MappingDataAssimilation
from igm.processes.iceflow.unified.mappings.interfaces.data_assimilation import InterfaceDataAssimilation
from igm.processes.iceflow.unified.optimizers.interfaces import InterfaceLBFGS
from igm.processes.iceflow.unified.optimizers.lbfgs_DA import OptimizerLBFGSBoundsDA
from igm.utils.math.precision import normalize_precision

from .phase_runner import (
    DataAssimilationRuntime,
    StaticBatchSampler,
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


def _build_halt(cfg) -> Halt:
    cfg_da = cfg.processes.data_assimilation_SR
    patience_metric = Metrics["cost"]()
    patience_halt_crit = Criteria["patience"](
        metric=patience_metric,
        dtype=cfg.processes.iceflow.numerics.precision,
        tol=1e-2,
        patience=cfg_da.optimization.minimizer_patience,
    )
    return Halt(
        crit_success=[patience_halt_crit],
        crit_failure=[],
        freq=1,
        dtype=cfg.processes.iceflow.numerics.precision,
    )


def data_assimilation_initialize(cfg, state) -> None:
    cfg_da = cfg.processes.data_assimilation_SR
    dtype = normalize_precision(cfg.processes.iceflow.numerics.precision)

    _initialize_inverted_fields(cfg, state, dtype)

    da_map = MappingDataAssimilation(
        **InterfaceDataAssimilation.get_mapping_args(cfg, state)
    )
    cost_fn, objective = build_cost_and_objective(cfg, state, da_map)

    optimizer_args = InterfaceLBFGS.get_optimizer_args(cfg, cost_fn, da_map)
    optimizer_args["halt"] = _build_halt(cfg)
    optimizer = OptimizerLBFGSBoundsDA(**optimizer_args)
    optimizer.sampler = StaticBatchSampler()

    da = DataAssimilationRuntime(
        map=da_map,
        opt=optimizer,
        cost_fn=cost_fn,
        objective=objective,
        out_freq=int(cfg_da.output.freq),
        retrain_iter=int(cfg_da.optimization.retrain_iter),
        retraining=initialize_retraining(cfg, state, da_map),
    )
    state.data_assimilation = da


def initialize(cfg, state) -> None:
    data_assimilation_initialize(cfg, state)


def update(cfg, state) -> None:
    da = state.data_assimilation
    reset_da_run_state(da)
    reset_retraining_run_state(da.retraining)

    run_da_phase(cfg, state, da)
    for _ in range(da.retrain_iter):
        run_retraining_phase(cfg, state, da)
        run_da_phase(cfg, state, da)


def finalize(cfg, state) -> None:
    pass
