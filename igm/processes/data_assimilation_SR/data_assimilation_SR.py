#!/usr/bin/env python3
# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from __future__ import annotations

import numpy as np
import tensorflow as tf

from igm.processes.iceflow.utils.data_preprocessing import fieldin_state_to_X
from igm.processes.iceflow.unified.evaluator import evaluate_iceflow

from .outputs.output_ncdf import update_ncdf_optimize
from igm.processes.iceflow.unified.optimizers import Optimizers, InterfaceOptimizers

from .utils import initial_thickness
from igm.utils.math.precision import normalize_precision

from igm.processes.iceflow.unified.mappings.data_assimilation import MappingDataAssimilation
from igm.processes.iceflow.unified.mappings.interfaces.data_assimilation import InterfaceDataAssimilation
from igm.processes.iceflow.unified.optimizers.lbfgs_DA import OptimizerLBFGSBoundsDA
from igm.processes.iceflow.unified.optimizers.interfaces import InterfaceLBFGS

from igm.processes.iceflow.unified.halt import Halt
from igm.processes.iceflow.unified.halt.criteria import Criteria
from igm.processes.iceflow.unified.halt.metrics import Metrics


from igm.processes.iceflow.data_preparation.batch_builder import TrainingBatchBuilder
from igm.processes.data_assimilation_SR.objective import build_objective_from_cfg
from igm.processes.pretraining.cost_tmp import get_cost_fn


class DataAssimilation:
    def __init__(self):
        self.map = None
        self.opt = None
        self.cost_fn = None
        self.objective = None
        self.maxiter = 0
        self.out_freq = 0
        self.retrain_iter = 1
        self.result = None
        self.result_stage1 = None
        self.result_stage2 = None
        self.results_da = []
        self._ncdf_next_iter = 0

def get_cost_and_obj(cfg, state, da_map):
    objective = build_objective_from_cfg(cfg, state, da_map)

    def cost_function(U, V, inputs):
        total, misfit, reg, _ = objective(U, V, inputs)
        return total, misfit, reg

    return cost_function, objective

def _evaluate_cost_terms_current_theta(da: DataAssimilation, inputs):
    U, V = da.map.get_UV(inputs)
    inputs_used = da.map.inputs if hasattr(da.map, "inputs") else inputs
    total, data, reg = da.cost_fn(U, V, inputs_used)
    return total, data, reg

def _sync_state_costs_and_outputs(cfg, state, da: DataAssimilation, iteration: int, inputs) -> None:
    da.map.update_state_fields(state)

    total, data, reg = _evaluate_cost_terms_current_theta(da, inputs)
    state.da_cost_total = float(total.numpy())
    state.da_cost_data = float(data.numpy())
    state.da_cost_reg = float(reg.numpy())

    evaluate_iceflow(cfg, state)
    update_ncdf_optimize(cfg, state, int(iteration))

def _configure_da_step_callback(cfg, state, da: DataAssimilation, iter_offset: int) -> None:
    def _step_callback(it_tf):
        # Shift by +1 so we do not reuse the phase-start iteration index that was already written before minimize().
        it = int(it_tf.numpy()) + 1
        out_it = iter_offset + it

        da.map.update_state_fields(state)

        X = fieldin_state_to_X(cfg, state)
        inputs = state.iceflow.patching.generate_patches(X)

        total, data, reg = _evaluate_cost_terms_current_theta(da, inputs)
        state.da_cost_total = float(total.numpy())
        state.da_cost_data = float(data.numpy())
        state.da_cost_reg = float(reg.numpy())

        evaluate_iceflow(cfg, state)
        update_ncdf_optimize(cfg, state, out_it)

    da.map.set_step_callback(_step_callback, out_freq=da.out_freq)

def _run_da_phase(cfg, state, da: DataAssimilation, store_attr: str | None = None):
    iter_offset = int(da._ncdf_next_iter)
    _configure_da_step_callback(cfg, state, da, iter_offset)

    da.map.update_state_fields(state)
    X = fieldin_state_to_X(cfg, state)
    inputs = state.iceflow.patching.generate_patches(X)

    _sync_state_costs_and_outputs(cfg, state, da, iter_offset, inputs)

    costs = da.opt.minimize(inputs)

    if store_attr is not None:
        setattr(da, store_attr, costs)
    da.result = costs
    da.results_da.append(costs)

    # Number of accepted optimizer iterations in this phase.
    n_steps = int(tf.shape(costs)[0].numpy()) if tf.rank(costs) > 0 else 0
    final_iter = iter_offset + max(1, n_steps)

    # Keep state synchronized at the optimizer endpoint.
    da.map.update_state_fields(state)
    evaluate_iceflow(cfg, state)

    # Avoid duplicating the final output if the periodic callback already wrote it.
    callback_already_wrote_final = (
        da.out_freq > 0 and n_steps > 0 and (n_steps % int(da.out_freq) == 0)
    )
    if not callback_already_wrote_final:
        X = fieldin_state_to_X(cfg, state)
        inputs = state.iceflow.patching.generate_patches(X)
        _sync_state_costs_and_outputs(cfg, state, da, final_iter, inputs)

    da._ncdf_next_iter = final_iter + 1
    return costs

def _retrain_network_with_current_thickness(cfg, state, da: DataAssimilation):
    # sanity check: DA mapping and base mapping share the same network object.
    if hasattr(da.map, "network") and hasattr(state.iceflow.mapping, "network"):
        assert da.map.network is state.iceflow.mapping.network

    X = fieldin_state_to_X(cfg, state)
    inputs = state.iceflow.patching.generate_patches(X)

    optimizer_args = InterfaceOptimizers["adam"].get_optimizer_args(
        cfg=cfg,
        cost_fn=get_cost_fn(cfg, state),
        map=state.iceflow.mapping,
    )
    opt_train = Optimizers["adam"](**optimizer_args)
    opt_train.sampler = state.iceflow.optimizer.sampler

    opt_train.update_parameters(500, 1e-5, 0.98, 50)
    opt_train.minimize(inputs)

def data_assimilation_initialize(cfg, state):
    cfg_da = cfg.processes.data_assimilation_SR
    dtype = normalize_precision(cfg.processes.iceflow.numerics.precision)

    da = DataAssimilation()

    # Initial thickness and parameter guesses.
    thk0 = initial_thickness(
        s=state.usurf,
        u=state.uvelsurfobs,
        v=state.vvelsurfobs,
        mask=state.icemask,
        dx=state.dX[0, 0],
        dy=state.dX[0, 0],
    )
    slidingco0 = np.zeros_like(thk0) + 0.21
    arrhenius0 = np.zeros_like(thk0) + 78.0

    state.uvelsurfobs = tf.cast(state.uvelsurfobs, dtype=dtype)
    state.vvelsurfobs = tf.cast(state.vvelsurfobs, dtype=dtype)
    state.thk = tf.convert_to_tensor(thk0, dtype=dtype)
    state.slidingco = tf.convert_to_tensor(slidingco0, dtype=dtype)
    state.arrhenius = tf.convert_to_tensor(arrhenius0, dtype=dtype)
    state.usurf = tf.cast(state.usurf, dtype=dtype)
    state.dX = tf.cast(state.dX, dtype=dtype)

    mapping_args = InterfaceDataAssimilation.get_mapping_args(cfg, state)
    da.map = MappingDataAssimilation(**mapping_args)

    da.cost_fn, da.objective = get_cost_and_obj(cfg, state, da.map)

    patience_metric = Metrics["cost"]()
    patience_halt_crit = Criteria["patience"](
        metric=patience_metric,
        dtype=cfg.processes.iceflow.numerics.precision,
        tol=1e-2,
        patience=cfg_da.optimization.minimizer_patience,
    )
    halt = Halt(
        crit_success=[patience_halt_crit],
        crit_failure=[],
        freq=1,
        dtype=cfg.processes.iceflow.numerics.precision,
    )

    optimizer_args = InterfaceLBFGS.get_optimizer_args(cfg, da.cost_fn, da.map)
    optimizer_args["halt"] = halt
    da.opt = OptimizerLBFGSBoundsDA(**optimizer_args)

    num_patches = state.iceflow.patching.num_patches
    patch_H, patch_W, patch_C = state.iceflow.patching.patch_shape
    sampler = TrainingBatchBuilder(
        preparation_params=state.iceflow.preparation_params,
        fieldin_names=state.iceflow.preparation_params.fieldin_names,
        patch_shape=(patch_H, patch_W, patch_C),
        num_patches=num_patches,
    )
    da.opt.sampler = sampler

    da.maxiter = int(cfg_da.optimization.nbitmax)
    da.out_freq = int(cfg_da.output.freq)
    da.retrain_iter = int(cfg_da.optimization.retrain_iter)

    state.data_assimilation = da

def initialize(cfg, state):
    data_assimilation_initialize(cfg, state)

def update(cfg, state):
    da = state.data_assimilation

    da.results_da = []
    da.result = None
    da.result_stage1 = None
    da.result_stage2 = None
    da._ncdf_next_iter = 0

    # Always start with one DA phase.
    _run_da_phase(cfg, state, da, store_attr="result_stage1")

    # Then alternate RT -> DA retrain_iter number of times.
    for k in range(da.retrain_iter):
        _retrain_network_with_current_thickness(cfg, state, da)
        store_attr = "result_stage2" if k == 0 else None
        _run_da_phase(cfg, state, da, store_attr=store_attr)

def finalize(cfg, state):
    pass
