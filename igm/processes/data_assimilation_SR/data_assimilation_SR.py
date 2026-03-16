#!/usr/bin/env python3
# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from __future__ import annotations

from pathlib import Path
import random

import tensorflow as tf

from igm.processes.iceflow.utils.data_preprocessing import fieldin_state_to_X
from igm.processes.iceflow.unified.evaluator import evaluate_iceflow

from .outputs.output_ncdf import update_ncdf_optimize
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
from igm.processes.pretraining.io_tfrecords import load_metadata, list_shards, make_datasets


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

        # shared network / retraining
        self.shared_mapping = None
        self.shared_network = None
        self.cost_fn_train = None
        self._train_sampler = None
        self.retrain_optimizer = None
        self._retrain_train_step = None

        # retraining hyperparameters
        self.retrain_steps = 500
        self.retrain_lr = 1e-4
        self.retrain_log_freq = 50
        self.retrain_anchor_weight = 1e-4
        self.retrain_replay_data_weight = 5e-2
        self.retrain_replay_phys_weight = 0.0
        self.retrain_replay_batch_size = 8
        self.retrain_replay_val_batches = 4

        # pretrained anchor snapshot
        self.pretrained_network_weights = []

        # replay
        self.replay_enabled = False
        self.replay_train_it = None
        self.replay_val_buffer = []
        self.replay_metadata = None
        self.replay_data_loss_fn = None

        # diagnostics
        self.retrain_phase = 0
        self.retrain_history = []
        self.last_retrain_summary = None


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

def _get_inverted_field_names(cfg) -> set[str]:
    return {
        str(item["name"])
        for item in getattr(cfg.processes.data_assimilation_SR, "variables", [])
    }

def _initialize_inverted_fields(cfg, state, dtype) -> set[str]:
    """
    Only initialize fields that are actually inverted for.
    Non-inverted fields are left unchanged.
    """
    inverted = _get_inverted_field_names(cfg)

    # These are not inversion variables, just cast once for consistency.
    state.uvelsurfobs = tf.cast(state.uvelsurfobs, dtype=dtype)
    state.vvelsurfobs = tf.cast(state.vvelsurfobs, dtype=dtype)
    state.usurf = tf.cast(state.usurf, dtype=dtype)
    state.dX = tf.cast(state.dX, dtype=dtype)

    if "thk" in inverted:
        thk0 = initial_thickness(
            s=state.usurf,
            u=state.uvelsurfobs,
            v=state.vvelsurfobs,
            mask=state.icemask,
            dx=state.dX[0, 0],
            dy=state.dX[0, 0],
        )
        state.thk = tf.convert_to_tensor(thk0, dtype=dtype)

    if "slidingco" in inverted:
        state.slidingco = (
            tf.ones_like(state.usurf, dtype=dtype)
            * tf.cast(cfg.processes.iceflow.physics.init_slidingco, dtype)
        )

    if "arrhenius" in inverted:
        state.arrhenius = (
            tf.ones_like(state.usurf, dtype=dtype)
            * tf.cast(cfg.processes.iceflow.physics.init_arrhenius, dtype)
        )

    return inverted

def _configure_da_step_callback(cfg, state, da: DataAssimilation, iter_offset: int) -> None:
    def _step_callback(it_tf):
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

    n_steps = int(tf.shape(costs)[0].numpy()) if tf.rank(costs) > 0 else 0
    final_iter = iter_offset + max(1, n_steps)

    da.map.update_state_fields(state)
    evaluate_iceflow(cfg, state)

    callback_already_wrote_final = da.out_freq > 0 and n_steps > 0 and (n_steps % int(da.out_freq) == 0)
    if not callback_already_wrote_final:
        X = fieldin_state_to_X(cfg, state)
        inputs = state.iceflow.patching.generate_patches(X)
        _sync_state_costs_and_outputs(cfg, state, da, final_iter, inputs)

    da._ncdf_next_iter = final_iter + 1
    return costs


def _build_replay_data_loss_fn(cfg):

    delta = float(50.0)
    huber = tf.keras.losses.Huber(delta=delta, reduction=tf.keras.losses.Reduction.NONE)

    @tf.function(reduce_retracing=True, jit_compile=False)
    def replay_data_loss(U, V, y_batch):
        Ut = tf.cast(y_batch[..., 0], U.dtype)
        Vt = tf.cast(y_batch[..., 1], V.dtype)
        return tf.reduce_mean(huber(Ut, U) + huber(Vt, V))


    return replay_data_loss


def _validate_replay_dataset(cfg, replay_root: Path):
    meta = load_metadata(replay_root)
    Nz = int(cfg.processes.iceflow.numerics.Nz)

    if "example_shapes_by_nz" not in meta or str(Nz) not in meta["example_shapes_by_nz"]:
        raise ValueError(f"Replay metadata at {replay_root} has no entry for Nz={Nz}.")

    shapes = meta["example_shapes_by_nz"][str(Nz)]
    H, W, Cx = shapes["x"]
    inputs = tuple(str(x) for x in cfg.processes.iceflow.unified.inputs)

    if Cx != len(inputs):
        raise ValueError(
            f"Replay TFRecords have C={Cx} channels, but cfg.processes.iceflow.unified.inputs has {len(inputs)} entries: {inputs}."
        )

    if Cx != 3 or inputs != ("thk", "usurf", "slidingco"):
        raise ValueError(
            "Replay retraining currently reuses igm.processes.pretraining.io_tfrecords.parse_example(), "
            "which assumes exactly 3 channels in the order ('thk', 'usurf', 'slidingco'). "
            f"Got C={Cx}, inputs={inputs}."
        )

    return meta, H, W


def _setup_replay(cfg, da: DataAssimilation) -> None:
    cfg_opt = cfg.processes.data_assimilation_SR.optimization

    replay_root = Path("/home/srosier/work2/tfrecords/strict_v2/")
    meta, H, W = _validate_replay_dataset(cfg, replay_root)
    Nz = int(cfg.processes.iceflow.numerics.Nz)

    replay_batch_size = int(4)
    replay_val_batches = int(4)
    replay_shuffle_buffer = int(2048)
    split_seed = int(0)

    train_files = list_shards(replay_root, Nz, split="train")
    val_files = list_shards(replay_root, Nz, split="val")

    rng = random.Random(split_seed)
    rng.shuffle(train_files)

    train_ds, val_ds = make_datasets(
        train_files=train_files,
        val_files=val_files,
        H=H,
        W=W,
        Nz=Nz,
        compression="GZIP",
        batch_size=replay_batch_size,
        shuffle_buffer=replay_shuffle_buffer,
    )

    da.replay_train_it = iter(train_ds)
    da.replay_val_buffer = []
    val_it = iter(val_ds.repeat())
    for _ in range(replay_val_batches):
        x_b, y_b = next(val_it)
        da.replay_val_buffer.append((tf.identity(x_b), tf.identity(y_b)))

    da.replay_enabled = True
    da.replay_metadata = meta
    da.retrain_replay_batch_size = replay_batch_size
    da.retrain_replay_val_batches = replay_val_batches
    da.replay_data_loss_fn = _build_replay_data_loss_fn(cfg)

    print(
        "[replay] enabled "
        f"batch_size={replay_batch_size} "
        f"val_batches={replay_val_batches} "
        f"data_dir={replay_root}"
    )


def _sample_local_batch(da: DataAssimilation, current_inputs: tf.Tensor) -> tf.Tensor:
    sampled = da._train_sampler(current_inputs)
    if sampled.shape.rank == 5:
        return sampled[0, :, :, :, :]
    if sampled.shape.rank == 4:
        return sampled
    raise ValueError(f"Unexpected sampler output rank {sampled.shape.rank}; expected 4 or 5.")


def _anchor_loss(current_vars, ref_vars):
    if not current_vars:
        return tf.constant(0.0, dtype=tf.float32)
    dtype = current_vars[0].dtype
    vals = [tf.reduce_mean(tf.square(tf.cast(v, dtype) - tf.cast(v0, dtype))) for v, v0 in zip(current_vars, ref_vars)]
    return tf.add_n(vals) / tf.cast(len(vals), dtype)


def _evaluate_local_physics_loss(da: DataAssimilation, current_inputs: tf.Tensor) -> tf.Tensor:
    U, V = da.shared_mapping.get_UV(current_inputs)
    return tf.cast(da.cost_fn_train(U, V, current_inputs), U.dtype)


def _evaluate_replay_validation(da: DataAssimilation):
    dtype = da.shared_network.trainable_variables[0].dtype
    if not da.replay_enabled or not da.replay_val_buffer:
        z = tf.constant(0.0, dtype=dtype)
        return z, z

    data_vals = []
    phys_vals = []
    for x_b, y_b in da.replay_val_buffer:
        U, V = da.shared_mapping.get_UV(x_b)
        data_vals.append(tf.cast(da.replay_data_loss_fn(U, V, y_b), dtype))
        phys_vals.append(tf.cast(da.cost_fn_train(U, V, x_b), dtype))

    n = tf.cast(len(data_vals), dtype)
    return tf.add_n(data_vals) / n, tf.add_n(phys_vals) / n


def _make_retrain_train_step(da: DataAssimilation):
    shared_mapping = da.shared_mapping
    shared_vars = da.shared_network.trainable_variables
    dtype = shared_vars[0].dtype
    w_anchor = tf.constant(float(da.retrain_anchor_weight), dtype=dtype)
    w_replay_data = tf.constant(float(da.retrain_replay_data_weight), dtype=dtype)
    w_replay_phys = tf.constant(float(da.retrain_replay_phys_weight), dtype=dtype)
    replay_enabled = bool(da.replay_enabled)
    replay_phys_enabled = float(da.retrain_replay_phys_weight) > 0.0

    @tf.function(reduce_retracing=True, jit_compile=False)
    def train_step(local_x, replay_x, replay_y):
        with tf.GradientTape() as tape:
            U_local, V_local = shared_mapping.get_UV(local_x)
            local_phys = tf.cast(da.cost_fn_train(U_local, V_local, local_x), dtype)

            total = local_phys
            replay_data = tf.zeros((), dtype=dtype)
            replay_phys = tf.zeros((), dtype=dtype)

            if replay_enabled:
                U_rep, V_rep = shared_mapping.get_UV(replay_x)
                replay_data = tf.cast(da.replay_data_loss_fn(U_rep, V_rep, replay_y), dtype)
                total = total + w_replay_data * replay_data

                if replay_phys_enabled:
                    replay_phys = tf.cast(da.cost_fn_train(U_rep, V_rep, replay_x), dtype)
                    total = total + w_replay_phys * replay_phys

            anchor = tf.cast(_anchor_loss(shared_vars, da.pretrained_network_weights), dtype)
            total = total + w_anchor * anchor

        grads = tape.gradient(total, shared_vars)
        grads = [tf.zeros_like(v) if g is None else g for g, v in zip(grads, shared_vars)]
        da.retrain_optimizer.apply_gradients(zip(grads, shared_vars))
        return total, local_phys, replay_data, replay_phys, anchor

    return train_step


def _retrain_network_with_current_thickness(cfg, state, da: DataAssimilation):
    if hasattr(da.map, "network") and hasattr(state.iceflow.mapping, "network"):
        assert da.map.network is state.iceflow.mapping.network

    da.map.update_state_fields(state)
    X = fieldin_state_to_X(cfg, state)
    inputs_current = state.iceflow.patching.generate_patches(X)

    local_phys_before = tf.cast(_evaluate_local_physics_loss(da, inputs_current), da.shared_network.trainable_variables[0].dtype)
    replay_val_data_before, replay_val_phys_before = _evaluate_replay_validation(da)

    history = {
        "phase": int(da.retrain_phase),
        "local_phys_before": float(local_phys_before.numpy()),
        "replay_val_data_before": float(replay_val_data_before.numpy()),
        "replay_val_phys_before": float(replay_val_phys_before.numpy()),
        "steps": [],
    }

    print(
        f"[retrain {da.retrain_phase}] start "
        f"local_phys={history['local_phys_before']:.6e} "
        f"replay_val_data={history['replay_val_data_before']:.6e} "
        f"replay_val_phys={history['replay_val_phys_before']:.6e}"
    )

    for step in range(1, da.retrain_steps + 1):
        local_batch = _sample_local_batch(da, inputs_current)

        if da.replay_enabled:
            replay_x, replay_y = next(da.replay_train_it)
        else:
            replay_x = tf.zeros([1, 1, 1, 3], dtype=local_batch.dtype)
            replay_y = tf.zeros([1, 1, 1, 2, 2], dtype=local_batch.dtype)

        total, local_phys, replay_data, replay_phys, anchor = da._retrain_train_step(local_batch, replay_x, replay_y)

        if step == 1 or step == da.retrain_steps or (da.retrain_log_freq > 0 and step % da.retrain_log_freq == 0):
            replay_val_data, replay_val_phys = _evaluate_replay_validation(da)
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
                f"[retrain {da.retrain_phase}] step {step:4d}/{da.retrain_steps} "
                f"total={rec['total']:.6e} "
                f"local_phys={rec['local_phys']:.6e} "
                f"replay_train_data={rec['replay_train_data']:.6e} "
                f"anchor={rec['anchor']:.6e} "
                f"replay_val_data={rec['replay_val_data']:.6e}"
            )

    da.map.update_state_fields(state)
    X = fieldin_state_to_X(cfg, state)
    inputs_current = state.iceflow.patching.generate_patches(X)

    local_phys_after = tf.cast(_evaluate_local_physics_loss(da, inputs_current), da.shared_network.trainable_variables[0].dtype)
    replay_val_data_after, replay_val_phys_after = _evaluate_replay_validation(da)

    history["local_phys_after"] = float(local_phys_after.numpy())
    history["replay_val_data_after"] = float(replay_val_data_after.numpy())
    history["replay_val_phys_after"] = float(replay_val_phys_after.numpy())

    da.retrain_history.append(history)
    da.last_retrain_summary = history
    state.retrain_local_phys = history["local_phys_after"]
    state.retrain_replay_val_data = history["replay_val_data_after"]
    state.retrain_replay_val_phys = history["replay_val_phys_after"]

    print(
        f"[retrain {da.retrain_phase}] end   "
        f"local_phys={history['local_phys_after']:.6e} "
        f"replay_val_data={history['replay_val_data_after']:.6e} "
        f"replay_val_phys={history['replay_val_phys_after']:.6e}"
    )

    da.retrain_phase += 1


def data_assimilation_initialize(cfg, state):
    cfg_da = cfg.processes.data_assimilation_SR
    cfg_opt = cfg.processes.data_assimilation_SR.optimization
    dtype = normalize_precision(cfg.processes.iceflow.numerics.precision)

    da = DataAssimilation()

    _initialize_inverted_fields(cfg, state, dtype)

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

    da.shared_mapping = state.iceflow.mapping
    da.shared_network = state.iceflow.mapping.network
    da.cost_fn_train = get_cost_fn(cfg, state)
    da._train_sampler = state.iceflow.optimizer.sampler

    da.pretrained_network_weights = [
        tf.Variable(v.read_value(), trainable=False, name=f"pretrained_anchor_{i}")
        for i, v in enumerate(da.shared_network.trainable_variables)
    ]

    da.retrain_steps = int(500)
    da.retrain_lr = cfg_opt.retrain_lr
    da.retrain_log_freq = int(50)
    da.retrain_anchor_weight = float(cfg_opt.retrain_anchor_weight)
    da.retrain_replay_data_weight = float(cfg_opt.retrain_replay_data_weight)
    da.retrain_replay_phys_weight = float(cfg_opt.retrain_replay_phys_weight)

    da.retrain_optimizer = tf.keras.optimizers.Adam(learning_rate=da.retrain_lr)
    if hasattr(da.retrain_optimizer, "build"):
        da.retrain_optimizer.build(da.shared_network.trainable_variables)

    replay_requested = (
        da.retrain_replay_data_weight > 0.0
        or da.retrain_replay_phys_weight > 0.0
    )

    if replay_requested:
        _setup_replay(cfg, da)
    else:
        da.replay_enabled = False
        da.replay_data_loss_fn = _build_replay_data_loss_fn(cfg)

    da._retrain_train_step = _make_retrain_train_step(da)

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
    da.retrain_phase = 1
    da.retrain_history = []
    da.last_retrain_summary = None

    _run_da_phase(cfg, state, da, store_attr="result_stage1")

    for k in range(da.retrain_iter):
        _retrain_network_with_current_thickness(cfg, state, da)
        store_attr = "result_stage2" if k == 0 else None
        _run_da_phase(cfg, state, da, store_attr=store_attr)


def finalize(cfg, state):
    pass