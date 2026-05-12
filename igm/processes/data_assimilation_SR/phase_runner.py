#!/usr/bin/env python3
# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import tensorflow as tf

from igm.processes.iceflow.unified.evaluator import evaluate_iceflow
from igm.processes.iceflow.unified.solver.solver import get_solver_inputs_from_state

from .objective import build_objective_from_cfg
from .outputs.output_ncdf import update_ncdf_optimize


@dataclass
class DACostHistory:
    total: tf.Tensor
    data: tf.Tensor
    reg: tf.Tensor


@dataclass
class DataAssimilationRuntime:
    map: Any
    opt: Any
    cost_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor, tf.Tensor]]
    objective: Any
    out_freq: int
    retrain_iter: int
    retraining: Any = None
    frozen_cost_history: Optional[DACostHistory] = None
    next_snapshot_iter: int = 0
    retrain_iter_num: int = 0


def build_cost_and_objective(cfg, state, da_map):
    objective = build_objective_from_cfg(cfg, state, da_map)

    def cost_function(U, V, inputs):
        total, misfit, reg, _ = objective(U, V, inputs)
        return total, misfit, reg

    return cost_function, objective


def reset_da_run_state(da: DataAssimilationRuntime) -> None:
    da.frozen_cost_history = None
    da.next_snapshot_iter = 0


def build_current_inputs(cfg, state) -> tf.Tensor:
    return get_solver_inputs_from_state(cfg, state)


def evaluate_cost_terms_current_theta(
    da: DataAssimilationRuntime,
    inputs: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    U, V = da.map.get_UV(inputs)
    total, data, reg = da.cost_fn(U, V, da.map.inputs)
    return total, data, reg


def _empty_cost_history(da: DataAssimilationRuntime) -> DACostHistory:
    dtype = getattr(getattr(da.opt, "last_total", None), "dtype", tf.float32)
    empty = tf.zeros([0], dtype=dtype)
    return DACostHistory(total=empty, data=empty, reg=empty)


def _get_frozen_cost_history(da: DataAssimilationRuntime) -> DACostHistory:
    return da.frozen_cost_history if da.frozen_cost_history is not None else _empty_cost_history(da)


def _optimizer_cost_history(da: DataAssimilationRuntime) -> DACostHistory:
    total = getattr(da.opt, "accepted_cost_total_hist", None)
    data = getattr(da.opt, "accepted_cost_data_hist", None)
    reg = getattr(da.opt, "accepted_cost_reg_hist", None)

    if total is None or data is None or reg is None:
        return _empty_cost_history(da)

    return DACostHistory(
        total=tf.identity(total),
        data=tf.identity(data),
        reg=tf.identity(reg),
    )


def _merge_cost_histories(a: DACostHistory, b: DACostHistory) -> DACostHistory:
    if tf.size(a.total).numpy() == 0:
        return DACostHistory(
            total=tf.identity(b.total),
            data=tf.identity(b.data),
            reg=tf.identity(b.reg),
        )

    if tf.size(b.total).numpy() == 0:
        return DACostHistory(
            total=tf.identity(a.total),
            data=tf.identity(a.data),
            reg=tf.identity(a.reg),
        )

    return DACostHistory(
        total=tf.concat([a.total, b.total], axis=0),
        data=tf.concat([a.data, b.data], axis=0),
        reg=tf.concat([a.reg, b.reg], axis=0),
    )


def _current_cost_history(
    da: DataAssimilationRuntime,
    *,
    include_live_phase: bool,
) -> DACostHistory:
    frozen = _get_frozen_cost_history(da)
    if not include_live_phase:
        return frozen
    return _merge_cost_histories(frozen, _optimizer_cost_history(da))


def _set_state_da_costs(state, total, data, reg) -> None:
    state.da_cost_total = float(tf.convert_to_tensor(total).numpy())
    state.da_cost_data = float(tf.convert_to_tensor(data).numpy())
    state.da_cost_reg = float(tf.convert_to_tensor(reg).numpy())


def _set_state_da_cost_history(state, history: DACostHistory) -> None:
    total = np.asarray(history.total.numpy())
    data = np.asarray(history.data.numpy())
    reg = np.asarray(history.reg.numpy())

    n_hist = int(total.shape[0])
    state.da_cost_hist_iter = np.arange(1, n_hist + 1, dtype=np.int32)
    state.da_cost_total_hist = total
    state.da_cost_data_hist = data
    state.da_cost_reg_hist = reg


def _push_state_and_maybe_write_ncdf(
    cfg,
    state,
    da: DataAssimilationRuntime,
    iteration: int,
    *,
    write_ncdf: bool,
) -> None:
    da.map.update_state_fields(state)
    evaluate_iceflow(cfg, state)
    state.retrain_iter_num = da.retrain_iter_num
    if write_ncdf:
        update_ncdf_optimize(cfg, state, int(iteration))


def _write_current_theta_snapshot(
    cfg,
    state,
    da: DataAssimilationRuntime,
    iteration: int,
    *,
    history: DACostHistory,
    inputs: Optional[tf.Tensor] = None,
    write_ncdf: bool = True,
) -> tf.Tensor:
    da.map.update_state_fields(state)

    if inputs is None:
        inputs = build_current_inputs(cfg, state)

    total, data, reg = evaluate_cost_terms_current_theta(da, inputs)
    _set_state_da_costs(state, total, data, reg)
    _set_state_da_cost_history(state, history)
    _push_state_and_maybe_write_ncdf(
        cfg,
        state,
        da,
        iteration,
        write_ncdf=write_ncdf,
    )
    return inputs


def _write_optimizer_snapshot(
    cfg,
    state,
    da: DataAssimilationRuntime,
    iteration: int,
    *,
    include_live_phase: bool,
) -> None:
    _set_state_da_costs(
        state,
        da.opt.last_total.read_value(),
        da.opt.last_data.read_value(),
        da.opt.last_reg.read_value(),
    )
    _set_state_da_cost_history(
        state,
        _current_cost_history(da, include_live_phase=include_live_phase),
    )
    _push_state_and_maybe_write_ncdf(
        cfg,
        state,
        da,
        iteration,
        write_ncdf=True,
    )


def sync_outputs_only(cfg, state, da: DataAssimilationRuntime) -> None:
    da.map.update_state_fields(state)
    evaluate_iceflow(cfg, state)


def configure_da_step_callback(
    cfg,
    state,
    da: DataAssimilationRuntime,
    snapshot_offset: int,
) -> None:
    def _step_callback(accepted_iter: int) -> None:
        snapshot_iter = snapshot_offset + int(accepted_iter)
        _write_optimizer_snapshot(
            cfg,
            state,
            da,
            snapshot_iter,
            include_live_phase=True,
        )

    da.map.set_step_callback(_step_callback, out_freq=da.out_freq)


def disable_da_step_callback(da: DataAssimilationRuntime) -> None:
    if hasattr(da.map, "clear_step_callback"):
        da.map.clear_step_callback()
    else:
        da.map.set_step_callback(None, out_freq=0)


def _num_accepted_steps(costs: tf.Tensor) -> int:
    rank = costs.shape.rank
    if rank == 0:
        return 0
    if rank is not None and rank > 0 and costs.shape[0] is not None:
        return int(costs.shape[0])
    return int(tf.shape(costs)[0].numpy())


def run_da_phase(cfg, state, da: DataAssimilationRuntime) -> tf.Tensor:
    snapshot_offset = int(da.next_snapshot_iter)
    frozen_before_phase = _get_frozen_cost_history(da)

    configure_da_step_callback(cfg, state, da, snapshot_offset)
    try:
        inputs = _write_current_theta_snapshot(
            cfg,
            state,
            da,
            snapshot_offset,
            history=frozen_before_phase,
            write_ncdf=True,
        )
        costs = da.opt.minimize(inputs)
    finally:
        disable_da_step_callback(da)

    accepted_history = _optimizer_cost_history(da)
    da.frozen_cost_history = _merge_cost_histories(frozen_before_phase, accepted_history)

    n_steps = _num_accepted_steps(costs)
    final_snapshot_iter = snapshot_offset + max(1, n_steps)
    final_written_by_callback = (
        da.out_freq > 0
        and n_steps > 0
        and (n_steps % int(da.out_freq) == 0)
    )

    if n_steps == 0:
        _write_current_theta_snapshot(
            cfg,
            state,
            da,
            final_snapshot_iter,
            history=da.frozen_cost_history,
            write_ncdf=True,
        )
    elif final_written_by_callback:
        sync_outputs_only(cfg, state, da)
    else:
        _write_optimizer_snapshot(
            cfg,
            state,
            da,
            final_snapshot_iter,
            include_live_phase=False,
        )

    da.next_snapshot_iter = final_snapshot_iter + 1
    return costs
