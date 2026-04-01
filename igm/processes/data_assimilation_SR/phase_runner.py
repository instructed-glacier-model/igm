#!/usr/bin/env python3
# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import tensorflow as tf

from igm.processes.iceflow.utils.data_preprocessing import fieldin_state_to_X
from igm.processes.iceflow.unified.evaluator import evaluate_iceflow

from .outputs.output_ncdf import update_ncdf_optimize
from .objective import build_objective_from_cfg


class StaticBatchSampler:
    dynamic_augmentation = False

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        # L-BFGS expects shape [M, B, H, W, C] with M=1
        return tf.expand_dims(inputs, axis=0)


@dataclass
class DataAssimilationRuntime:
    map: Any
    opt: Any
    cost_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor, tf.Tensor]]
    objective: Any
    out_freq: int
    retrain_iter: int
    retraining: Any = None
    result: Optional[tf.Tensor] = None
    result_stage1: Optional[tf.Tensor] = None
    result_stage2: Optional[tf.Tensor] = None
    results_da: list[tf.Tensor] = field(default_factory=list)
    _ncdf_next_iter: int = 0

    @property
    def shared_mapping(self):
        return self.retraining.shared_mapping

    @property
    def shared_network(self):
        return self.retraining.shared_network

    @property
    def cost_fn_train(self):
        return self.retraining.training_physics_cost_fn

    @property
    def retrain_optimizer(self):
        return self.retraining.optimizer

    @property
    def replay_enabled(self) -> bool:
        return self.retraining.replay.enabled

    @property
    def replay_train_it(self):
        return self.retraining.replay.train_it

    @property
    def replay_val_buffer(self):
        return self.retraining.replay.val_buffer

    @property
    def replay_metadata(self):
        return self.retraining.replay.metadata

    @property
    def replay_data_loss_fn(self):
        return self.retraining.replay.data_loss_fn

    @property
    def retrain_phase(self) -> int:
        return self.retraining.phase

    @retrain_phase.setter
    def retrain_phase(self, value: int) -> None:
        self.retraining.phase = int(value)

    @property
    def retrain_history(self):
        return self.retraining.history

    @property
    def last_retrain_summary(self):
        return self.retraining.last_summary

    @last_retrain_summary.setter
    def last_retrain_summary(self, value) -> None:
        self.retraining.last_summary = value

    @property
    def retrain_steps(self) -> int:
        return self.retraining.settings.steps

    @property
    def retrain_lr(self) -> float:
        return self.retraining.settings.lr

    @property
    def retrain_log_freq(self) -> int:
        return self.retraining.settings.log_freq

    @property
    def retrain_anchor_weight(self) -> float:
        return self.retraining.settings.anchor_weight

    @property
    def retrain_replay_data_weight(self) -> float:
        return self.retraining.settings.replay_data_weight

    @property
    def retrain_replay_phys_weight(self) -> float:
        return self.retraining.settings.replay_phys_weight

    @property
    def retrain_replay_batch_size(self) -> int:
        return self.retraining.settings.replay.batch_size

    @property
    def retrain_replay_val_batches(self) -> int:
        return self.retraining.settings.replay.val_batches

    @property
    def retrain_local_phys_scale(self):
        return self.retraining.local_phys_scale

    @property
    def retrain_replay_data_scale(self):
        return self.retraining.replay_data_scale

    @property
    def retrain_replay_phys_scale(self):
        return self.retraining.replay_phys_scale

    @property
    def pretrained_network_weights(self):
        return self.retraining.pretrained_network_weights


def build_cost_and_objective(cfg, state, da_map):
    objective = build_objective_from_cfg(cfg, state, da_map)

    def cost_function(U, V, inputs):
        total, misfit, reg, _ = objective(U, V, inputs)
        return total, misfit, reg

    return cost_function, objective


def reset_da_run_state(da: DataAssimilationRuntime) -> None:
    da.results_da.clear()
    da.result = None
    da.result_stage1 = None
    da.result_stage2 = None
    da._ncdf_next_iter = 0


def build_current_inputs(cfg, state) -> tf.Tensor:
    X = fieldin_state_to_X(cfg, state)
    return state.iceflow.patching.generate_patches(X)


def evaluate_cost_terms_current_theta(da: DataAssimilationRuntime, inputs: tf.Tensor):
    U, V = da.map.get_UV(inputs)
    inputs_used = da.map.inputs if hasattr(da.map, "inputs") else inputs
    total, data, reg = da.cost_fn(U, V, inputs_used)
    return total, data, reg


def snapshot_state(
    cfg,
    state,
    da: DataAssimilationRuntime,
    iteration: int,
    *,
    write_ncdf: bool,
    inputs: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    da.map.update_state_fields(state)

    if inputs is None:
        inputs = build_current_inputs(cfg, state)

    total, data, reg = evaluate_cost_terms_current_theta(da, inputs)
    state.da_cost_total = float(total.numpy())
    state.da_cost_data = float(data.numpy())
    state.da_cost_reg = float(reg.numpy())

    evaluate_iceflow(cfg, state)
    if write_ncdf:
        update_ncdf_optimize(cfg, state, int(iteration))

    return inputs


def sync_outputs_only(cfg, state, da: DataAssimilationRuntime) -> None:
    da.map.update_state_fields(state)
    evaluate_iceflow(cfg, state)


def configure_da_step_callback(cfg, state, da: DataAssimilationRuntime, iter_offset: int) -> None:
    def _step_callback(it_tf):
        out_it = iter_offset + int(it_tf.numpy()) + 1
        snapshot_state(cfg, state, da, out_it, write_ncdf=True)

    da.map.set_step_callback(_step_callback, out_freq=da.out_freq)


def run_da_phase(cfg, state, da: DataAssimilationRuntime, store_attr: str | None = None):
    iter_offset = int(da._ncdf_next_iter)
    configure_da_step_callback(cfg, state, da, iter_offset)

    inputs = snapshot_state(cfg, state, da, iter_offset, write_ncdf=True)
    costs = da.opt.minimize(inputs)

    if store_attr is not None:
        setattr(da, store_attr, costs)
    da.result = costs
    da.results_da.append(costs)

    n_steps = int(tf.shape(costs)[0].numpy()) if tf.rank(costs) > 0 else 0
    final_iter = iter_offset + max(1, n_steps)
    callback_already_wrote_final = (
        da.out_freq > 0 and n_steps > 0 and (n_steps % int(da.out_freq) == 0)
    )

    if callback_already_wrote_final:
        sync_outputs_only(cfg, state, da)
    else:
        snapshot_state(cfg, state, da, final_iter, write_ncdf=True)

    da._ncdf_next_iter = final_iter + 1
    return costs
