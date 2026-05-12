#!/usr/bin/env python3
# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import tensorflow as tf

from .phase_runner import build_current_inputs
from .utils import _safe_loss_scale
from igm.processes.pretraining.cost_tmp import get_cost_fn
from igm.processes.pretraining.training_utils import (
    _anchor_loss,
    build_tfrecord_datasets_for_nz,
    build_velocity_data_loss,
)


@dataclass(frozen=True)
class ReplaySettings:
    data_dir: Path
    batch_size: int
    val_batches: int
    shuffle_buffer: int
    split_seed: int
    data_loss_type: str
    data_loss_huber_delta: float


@dataclass(frozen=True)
class RetrainingSettings:
    steps: int
    lr: float
    log_freq: int
    anchor_weight: float
    replay_data_weight: float
    replay_phys_weight: float
    replay: ReplaySettings


@dataclass
class ReplayRuntime:
    enabled: bool
    data_loss_fn: Callable
    train_it: Any = None
    val_buffer: list[tuple[tf.Tensor, tf.Tensor]] = field(default_factory=list)
    metadata: Any = None


@dataclass
class RetrainingRuntime:
    settings: RetrainingSettings
    shared_mapping: Any
    shared_network: tf.keras.Model
    training_physics_cost_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]
    optimizer: tf.keras.optimizers.Optimizer
    trainer: Any
    pretrained_network_weights: list[tf.Variable]
    replay: ReplayRuntime
    local_phys_scale: tf.Variable
    replay_data_scale: tf.Variable
    replay_phys_scale: tf.Variable
    phase: int = 1
    history: list[dict[str, Any]] = field(default_factory=list)
    last_summary: Optional[dict[str, Any]] = None


class RetrainStepRunner:
    def __init__(
        self,
        *,
        shared_mapping,
        shared_network,
        training_physics_cost_fn,
        replay_data_loss_fn,
        optimizer,
        pretrained_network_weights,
        local_phys_scale,
        replay_data_scale,
        replay_phys_scale,
        anchor_weight: float,
        replay_data_weight: float,
        replay_phys_weight: float,
    ) -> None:
        self.shared_mapping = shared_mapping
        self.training_physics_cost_fn = training_physics_cost_fn
        self.replay_data_loss_fn = replay_data_loss_fn
        self.optimizer = optimizer
        self.shared_vars = tuple(shared_network.trainable_variables)
        self.pretrained_network_weights = tuple(pretrained_network_weights)
        self.local_phys_scale = local_phys_scale
        self.replay_data_scale = replay_data_scale
        self.replay_phys_scale = replay_phys_scale

        dtype = self.shared_vars[0].dtype
        self.dtype = dtype
        self.w_anchor = tf.constant(float(anchor_weight), dtype=dtype)
        self.w_replay_data = tf.constant(float(replay_data_weight), dtype=dtype)
        self.w_replay_phys = tf.constant(float(replay_phys_weight), dtype=dtype)
        self.replay_phys_enabled = float(replay_phys_weight) > 0.0

    @tf.function(reduce_retracing=True, jit_compile=False)
    def train_step_local_only(self, local_x: tf.Tensor):
        with tf.GradientTape() as tape:
            U_local, V_local = self.shared_mapping.get_UV(local_x)
            local_phys = tf.cast(
                self.training_physics_cost_fn(U_local, V_local, local_x),
                self.dtype,
            )
            total = local_phys / self.local_phys_scale
            anchor = tf.cast(
                _anchor_loss(self.shared_vars, self.pretrained_network_weights),
                self.dtype,
            )
            total = total + self.w_anchor * anchor

        grads = tape.gradient(total, self.shared_vars)
        grads = [
            tf.zeros_like(v) if g is None else g
            for g, v in zip(grads, self.shared_vars)
        ]
        self.optimizer.apply_gradients(zip(grads, self.shared_vars))

        zero = tf.zeros((), dtype=self.dtype)
        return total, local_phys, zero, zero, anchor

    @tf.function(reduce_retracing=True, jit_compile=False)
    def train_step_with_replay(
        self,
        local_x: tf.Tensor,
        replay_x: tf.Tensor,
        replay_y: tf.Tensor,
    ):
        with tf.GradientTape() as tape:
            U_local, V_local = self.shared_mapping.get_UV(local_x)
            local_phys = tf.cast(
                self.training_physics_cost_fn(U_local, V_local, local_x),
                self.dtype,
            )
            total = local_phys / self.local_phys_scale

            U_rep, V_rep = self.shared_mapping.get_UV(replay_x)
            replay_data = tf.cast(
                self.replay_data_loss_fn(U_rep, V_rep, replay_y),
                self.dtype,
            )
            total = total + self.w_replay_data * (
                replay_data / self.replay_data_scale
            )

            replay_phys = tf.zeros((), dtype=self.dtype)
            if self.replay_phys_enabled:
                replay_phys = tf.cast(
                    self.training_physics_cost_fn(U_rep, V_rep, replay_x),
                    self.dtype,
                )
                total = total + self.w_replay_phys * (
                    replay_phys / self.replay_phys_scale
                )

            anchor = tf.cast(
                _anchor_loss(self.shared_vars, self.pretrained_network_weights),
                self.dtype,
            )
            total = total + self.w_anchor * anchor

        grads = tape.gradient(total, self.shared_vars)
        grads = [
            tf.zeros_like(v) if g is None else g
            for g, v in zip(grads, self.shared_vars)
        ]
        self.optimizer.apply_gradients(zip(grads, self.shared_vars))
        return total, local_phys, replay_data, replay_phys, anchor


def _read_retraining_settings(cfg) -> RetrainingSettings:
    cfg_opt = cfg.processes.data_assimilation_SR.optimization
    return RetrainingSettings(
        steps=int(cfg_opt.retrain_steps),
        lr=float(cfg_opt.retrain_lr),
        log_freq=int(cfg_opt.retrain_log_freq),
        anchor_weight=float(cfg_opt.retrain_anchor_weight),
        replay_data_weight=float(cfg_opt.retrain_replay_data_weight),
        replay_phys_weight=float(cfg_opt.retrain_replay_phys_weight),
        replay=ReplaySettings(
            data_dir=Path(cfg_opt.replay_data_dir),
            batch_size=int(cfg_opt.replay_batch_size),
            val_batches=int(cfg_opt.replay_val_batches),
            shuffle_buffer=int(cfg_opt.replay_shuffle_buffer),
            split_seed=int(cfg_opt.replay_split_seed),
            data_loss_type=str(cfg_opt.replay_data_loss_type),
            data_loss_huber_delta=float(cfg_opt.replay_data_loss_huber_delta),
        ),
    )


def _build_pretrained_anchor_snapshot(network: tf.keras.Model) -> list[tf.Variable]:
    return [
        tf.Variable(tf.identity(v), trainable=False, name=f"pretrained_anchor_{i}")
        for i, v in enumerate(network.trainable_variables)
    ]


def _build_replay_runtime(cfg, settings: ReplaySettings, enable_replay: bool) -> ReplayRuntime:
    data_loss_fn = build_velocity_data_loss(
        loss_type=settings.data_loss_type,
        huber_delta=settings.data_loss_huber_delta,
    )

    if not enable_replay:
        return ReplayRuntime(enabled=False, data_loss_fn=data_loss_fn)

    Nz = int(cfg.processes.iceflow.numerics.Nz)
    datasets = build_tfrecord_datasets_for_nz(
        settings.data_dir,
        nz=Nz,
        inputs=tuple(str(x) for x in cfg.processes.iceflow.unified.inputs),
        batch_size=settings.batch_size,
        compression="GZIP",
        shuffle_buffer=settings.shuffle_buffer,
        split_seed=settings.split_seed,
    )

    replay = ReplayRuntime(
        enabled=True,
        data_loss_fn=data_loss_fn,
        train_it=iter(datasets.train_ds),
        metadata=datasets.metadata,
    )

    val_it = iter(datasets.val_ds.repeat())
    for _ in range(settings.val_batches):
        x_b, y_b = next(val_it)
        replay.val_buffer.append((tf.identity(x_b), tf.identity(y_b)))

    print(
        "[replay] enabled "
        f"batch_size={settings.batch_size} "
        f"val_batches={settings.val_batches} "
        f"data_dir={settings.data_dir}"
    )
    return replay


def initialize_retraining(cfg, state, da_map) -> RetrainingRuntime:
    settings = _read_retraining_settings(cfg)
    shared_mapping = state.iceflow.mapping
    shared_network = shared_mapping.network

    if not hasattr(da_map, "network") or da_map.network is not shared_network:
        raise RuntimeError(
            "Data assimilation mapping must share the exact network instance with "
            "state.iceflow.mapping for retraining to remain coupled to inversion."
        )

    optimizer = tf.keras.optimizers.Adam(learning_rate=settings.lr)
    if hasattr(optimizer, "build"):
        optimizer.build(shared_network.trainable_variables)

    dtype = shared_network.trainable_variables[0].dtype
    local_phys_scale = tf.Variable(
        1.0,
        trainable=False,
        dtype=dtype,
        name="retrain_local_phys_scale",
    )
    replay_data_scale = tf.Variable(
        1.0,
        trainable=False,
        dtype=dtype,
        name="retrain_replay_data_scale",
    )
    replay_phys_scale = tf.Variable(
        1.0,
        trainable=False,
        dtype=dtype,
        name="retrain_replay_phys_scale",
    )

    replay_requested = (
        settings.replay_data_weight > 0.0 or settings.replay_phys_weight > 0.0
    )
    replay = _build_replay_runtime(cfg, settings.replay, replay_requested)

    training_physics_cost_fn = get_cost_fn(cfg, state)
    pretrained_network_weights = _build_pretrained_anchor_snapshot(da_map.network)
    trainer = RetrainStepRunner(
        shared_mapping=shared_mapping,
        shared_network=shared_network,
        training_physics_cost_fn=training_physics_cost_fn,
        replay_data_loss_fn=replay.data_loss_fn,
        optimizer=optimizer,
        pretrained_network_weights=pretrained_network_weights,
        local_phys_scale=local_phys_scale,
        replay_data_scale=replay_data_scale,
        replay_phys_scale=replay_phys_scale,
        anchor_weight=settings.anchor_weight,
        replay_data_weight=settings.replay_data_weight,
        replay_phys_weight=settings.replay_phys_weight,
    )

    return RetrainingRuntime(
        settings=settings,
        shared_mapping=shared_mapping,
        shared_network=shared_network,
        training_physics_cost_fn=training_physics_cost_fn,
        optimizer=optimizer,
        trainer=trainer,
        pretrained_network_weights=pretrained_network_weights,
        replay=replay,
        local_phys_scale=local_phys_scale,
        replay_data_scale=replay_data_scale,
        replay_phys_scale=replay_phys_scale,
    )


def reset_retraining_run_state(retr: RetrainingRuntime) -> None:
    retr.phase = 1
    retr.history.clear()
    retr.last_summary = None


def _current_inputs_from_da_state(cfg, state, da) -> tf.Tensor:
    da.map.update_state_fields(state)
    return build_current_inputs(cfg, state)


def _evaluate_local_physics_loss(retr: RetrainingRuntime, current_inputs: tf.Tensor) -> tf.Tensor:
    U, V = retr.shared_mapping.get_UV(current_inputs)
    return tf.cast(retr.training_physics_cost_fn(U, V, current_inputs), U.dtype)


def _evaluate_replay_validation(retr: RetrainingRuntime):
    dtype = retr.shared_network.trainable_variables[0].dtype
    if not retr.replay.enabled or not retr.replay.val_buffer:
        z = tf.constant(0.0, dtype=dtype)
        return z, z

    data_vals = []
    phys_vals = []
    for x_b, y_b in retr.replay.val_buffer:
        U, V = retr.shared_mapping.get_UV(x_b)
        data_vals.append(tf.cast(retr.replay.data_loss_fn(U, V, y_b), dtype))
        phys_vals.append(tf.cast(retr.training_physics_cost_fn(U, V, x_b), dtype))

    n = tf.cast(len(data_vals), dtype)
    return tf.add_n(data_vals) / n, tf.add_n(phys_vals) / n


def run_retraining_phase(cfg, state, da) -> None:
    retr: RetrainingRuntime = da.retraining
    current_inputs = _current_inputs_from_da_state(cfg, state, da)

    dtype = retr.shared_network.trainable_variables[0].dtype
    local_phys_before = tf.cast(_evaluate_local_physics_loss(retr, current_inputs), dtype)
    replay_val_data_before, replay_val_phys_before = _evaluate_replay_validation(retr)

    retr.local_phys_scale.assign(_safe_loss_scale(local_phys_before, dtype))
    retr.replay_data_scale.assign(_safe_loss_scale(replay_val_data_before, dtype))
    retr.replay_phys_scale.assign(_safe_loss_scale(replay_val_phys_before, dtype))

    history = {
        "phase": int(retr.phase),
        "local_phys_before": float(local_phys_before.numpy()),
        "replay_val_data_before": float(replay_val_data_before.numpy()),
        "replay_val_phys_before": float(replay_val_phys_before.numpy()),
        "local_phys_scale": float(retr.local_phys_scale.numpy()),
        "replay_val_data_scale": float(retr.replay_data_scale.numpy()),
        "replay_val_phys_scale": float(retr.replay_phys_scale.numpy()),
        "steps": [],
    }

    print(
        f"[retrain {retr.phase}] start "
        f"local_phys={history['local_phys_before']:.6e} "
        f"replay_val_data={history['replay_val_data_before']:.6e} "
        f"replay_val_phys={history['replay_val_phys_before']:.6e}"
    )

    use_replay = retr.replay.enabled
    step_fn = (
        retr.trainer.train_step_with_replay
        if use_replay
        else retr.trainer.train_step_local_only
    )

    for step in range(1, retr.settings.steps + 1):
        if use_replay:
            replay_x, replay_y = next(retr.replay.train_it)
            total, local_phys, replay_data, replay_phys, anchor = step_fn(
                current_inputs,
                replay_x,
                replay_y,
            )
        else:
            total, local_phys, replay_data, replay_phys, anchor = step_fn(current_inputs)

        if (
            step == 1
            or step == retr.settings.steps
            or (
                retr.settings.log_freq > 0
                and step % retr.settings.log_freq == 0
            )
        ):
            replay_val_data, replay_val_phys = _evaluate_replay_validation(retr)
            rec = {
                "step": int(step),
                "total": float(total.numpy()),
                "local_phys": float(local_phys.numpy()),
                "replay_train_data": float(replay_data.numpy()),
                "replay_train_phys": float(replay_phys.numpy()),
                "anchor": float(anchor.numpy()),
                "replay_val_data": float(replay_val_data.numpy()),
                "replay_val_phys": float(replay_val_phys.numpy()),
            }
            history["steps"].append(rec)
            print(
                f"[retrain {retr.phase}] step {step:4d}/{retr.settings.steps} "
                f"total={rec['total']:.6e} "
                f"local_phys={rec['local_phys']:.6e} "
                f"replay_train_data={rec['replay_train_data']:.6e} "
                f"anchor={rec['anchor']:.6e} "
                f"replay_val_data={rec['replay_val_data']:.6e}"
            )

    current_inputs = _current_inputs_from_da_state(cfg, state, da)
    local_phys_after = tf.cast(_evaluate_local_physics_loss(retr, current_inputs), dtype)
    replay_val_data_after, replay_val_phys_after = _evaluate_replay_validation(retr)

    history["local_phys_after"] = float(local_phys_after.numpy())
    history["replay_val_data_after"] = float(replay_val_data_after.numpy())
    history["replay_val_phys_after"] = float(replay_val_phys_after.numpy())

    retr.history.append(history)
    retr.last_summary = history
    state.retrain_local_phys = history["local_phys_after"]
    state.retrain_replay_val_data = history["replay_val_data_after"]
    state.retrain_replay_val_phys = history["replay_val_phys_after"]

    print(
        f"[retrain {retr.phase}] end   "
        f"local_phys={history['local_phys_after']:.6e} "
        f"replay_val_data={history['replay_val_data_after']:.6e} "
        f"replay_val_phys={history['replay_val_phys_after']:.6e}"
    )

    retr.phase += 1
