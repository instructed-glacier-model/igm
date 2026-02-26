#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, Any, Callable, Optional
import random
import yaml
import time

import numpy as np
import tensorflow as tf

from igm.processes.iceflow.unified.mappings import Mappings, InterfaceMappings
from igm.processes.iceflow.emulate.utils.artifacts import save_emulator_artifact
from igm.processes.pretraining.cost_tmp import get_cost_fn
from igm.utils.math.precision import normalize_precision

# Best-guess local imports; adjust paths as needed in your repo
from .io_tfrecords import load_metadata, list_shards, make_datasets
from .history import load_history_yaml, save_history_yaml
from .plots import save_loss_plot, save_speed_compare

from igm.processes.iceflow.emulate.utils.artifacts_schema_v3 import build_manifest_v3
from igm.processes.iceflow.emulate.utils.normalizations import FixedChannelStandardization


def update(cfg, state):
    pass


def finalize(cfg, state):
    pass


# ----------------------------
# Dataclasses to reduce argument sprawl
# ----------------------------

@dataclass
class MetricsBundle:
    train_total: tf.keras.metrics.Metric
    train_data: tf.keras.metrics.Metric
    train_phys: tf.keras.metrics.Metric
    train_lam: tf.keras.metrics.Metric
    val_total: tf.keras.metrics.Metric
    val_data: tf.keras.metrics.Metric
    val_phys: tf.keras.metrics.Metric


@dataclass
class HistoryBundle:
    train_total: list
    val_total: list
    train_data: list
    val_data: list
    train_phys: list
    val_phys: list
    lambda_phys: list


@dataclass
class LoopContext:
    start_epoch: int
    n_epochs: int
    train_ds: tf.data.Dataset
    val_ds: tf.data.Dataset
    val_vis_it: Any
    train_step: Callable[[tf.Tensor, tf.Tensor], None]
    val_step: Callable[[tf.Tensor, tf.Tensor], None]
    mapping: Any
    Nz: int
    fig_dir: Path
    ckpt_mgr: Optional[tf.train.CheckpointManager]
    out_dir: Path
    make_plots: bool
    save_model: bool


def _prepare_run_dirs(out_dir: Path, resume: bool) -> Tuple[Path, Path]:
    ckpt_dir = out_dir / "checkpoints"
    fig_dir = out_dir / "figures"

    if resume:
        if not out_dir.exists():
            raise FileNotFoundError(
                f"resume=True but experiment directory does not exist: {out_dir}"
            )
        if not ckpt_dir.exists():
            raise FileNotFoundError(
                f"resume=True but checkpoints directory missing: {ckpt_dir}"
            )
        # history.yaml existence checked later by load_history_yaml()
    else:
        # Prevent silently overwriting an existing run
        if ckpt_dir.exists() and any(ckpt_dir.glob("ckpt-*")):
            raise FileExistsError(
                f"Experiment already has checkpoints at {ckpt_dir} but resume=False. "
                "Set cfg.processes.pretraining.resume=true or use a new experiment_name."
            )
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        fig_dir.mkdir(parents=True, exist_ok=True)

    return ckpt_dir, fig_dir


def _validate_pretraining_setup(inputs: Tuple[str, ...], Cx: int, cfg_physics, state) -> None:
    if Cx != len(inputs):
        raise ValueError(
            f"TFRecord x has C={Cx} channels (from metadata), but cfg.processes.iceflow.unified.inputs "
            f"has {len(inputs)} entries: {inputs}. These must match in count and order."
        )

    if Cx == 3 and inputs != ("thk", "usurf", "slidingco"):
        raise ValueError(
            f"TFRecord x has 3 channels and parse_example() assumes ('thk','usurf','slidingco') "
            f"in that order, but cfg inputs are {inputs}."
        )

    if not hasattr(state, "iceflow") or not hasattr(state.iceflow, "discr_v") or state.iceflow.discr_v is None:
        raise RuntimeError(
            "state.iceflow.discr_v is missing, but the physics cost requires it. "
            "Ensure the iceflow vertical discretization is initialized before pretraining."
        )


def _reset_metrics(metrics: MetricsBundle) -> None:
    metrics.train_total.reset_state()
    metrics.train_data.reset_state()
    metrics.train_phys.reset_state()
    metrics.train_lam.reset_state()
    metrics.val_total.reset_state()
    metrics.val_data.reset_state()
    metrics.val_phys.reset_state()


def _init_empty_histories() -> HistoryBundle:
    return HistoryBundle(
        train_total=[],
        val_total=[],
        train_data=[],
        val_data=[],
        train_phys=[],
        val_phys=[],
        lambda_phys=[],
    )


def _append_epoch(history: HistoryBundle, metrics: MetricsBundle) -> Tuple[float, float, float, float]:
    """
    Append epoch results to history (in-place) and return the most recent scalar values:
    (train_total, train_data, train_phys, lambda_phys_epoch_mean).
    """
    tt = float(metrics.train_total.result().numpy())
    td = float(metrics.train_data.result().numpy())
    tp = float(metrics.train_phys.result().numpy())
    lam = float(metrics.train_lam.result().numpy())

    vt = float(metrics.val_total.result().numpy())
    vd = float(metrics.val_data.result().numpy())
    vp = float(metrics.val_phys.result().numpy())

    history.train_total.append(tt)
    history.train_data.append(td)
    history.train_phys.append(tp)

    history.val_total.append(vt)
    history.val_data.append(vd)
    history.val_phys.append(vp)

    history.lambda_phys.append(lam)

    return tt, td, tp, lam


def _run_training_loop(ctx: LoopContext, metrics: MetricsBundle, history: HistoryBundle) -> None:
    """
    Non-TF orchestration: epoch loop, metric resets, history appends, printing,
    optional plotting, optional checkpointing, and optional history.yaml persistence.
    """
    TRAIN_STEPS = 1000
    VAL_STEPS   = 20

    train_it = iter(ctx.train_ds)  # infinite because of .repeat()

    for epoch in range(ctx.start_epoch, ctx.n_epochs):

        _reset_metrics(metrics)
        # --- train ---
        for _ in range(TRAIN_STEPS):
            x_b, y_b = next(train_it)
            ctx.train_step(x_b, y_b)

        # --- validate (new iterator each epoch => new shuffle order) ---
        val_it = iter(ctx.val_ds)
        for _ in range(VAL_STEPS):
            x_b, y_b = next(val_it)
            ctx.val_step(x_b, y_b)

        # --- append histories ---
        tt, td, tp, lam = _append_epoch(history, metrics)

        print(
            f"[epoch {epoch+1}/{ctx.n_epochs}] "
            f"train_total={tt:.6e} "
            f"train_data={td:.6e} "
            f"train_phys={tp:.6e} "
            f"lambda_phys={lam:.3e} "
            f"val_total={history.val_total[-1]:.6e}"
        )

        # --- plots + comparisons (optional) ---
        if ctx.make_plots:
            save_loss_plot(
                history.train_total, history.val_total,
                history.train_data,  history.val_data,
                history.train_phys,  history.val_phys,
                history.lambda_phys,
                ctx.fig_dir / "loss_curve.png",
            )

            x_vis, y_vis = next(ctx.val_vis_it)
            save_speed_compare(
                ctx.mapping,
                x_vis,
                y_vis,
                ctx.Nz,
                ctx.fig_dir / f"speed_compare_epoch{epoch+1:04d}.png",
            )

        # --- persistence (optional) ---
        if ctx.save_model:
            if ctx.ckpt_mgr is not None:
                ctx.ckpt_mgr.save()
            save_history_yaml(
                out_dir=ctx.out_dir,
                epoch=epoch + 1,
                train_total_hist=history.train_total,
                val_total_hist=history.val_total,
                train_data_hist=history.train_data,
                val_data_hist=history.val_data,
                train_phys_hist=history.train_phys,
                val_phys_hist=history.val_phys,
                lambda_hist=history.lambda_phys,
            )


def initialize(cfg, state):
    tf.config.optimizer.set_jit(False)

    # ----------------------------
    # A) Config / paths
    # ----------------------------
    cfg_pretraining = cfg.processes.pretraining
    cfg_iceflow     = cfg.processes.iceflow
    cfg_physics     = cfg.processes.iceflow.physics
    Nz = cfg_iceflow.numerics.Nz

    make_plots = bool(getattr(cfg_pretraining, "make_plots", True))
    save_model = bool(getattr(cfg_pretraining, "save_model", True))

    tfrecord_root = Path(cfg_pretraining.data_dir)

    out_dir = Path(cfg_pretraining.out_dir) / cfg_pretraining.experiment_name

    # If save_model=False, we force resume off (your requirement: no resume/manifest/checkpoints).
    resume = bool(getattr(cfg_pretraining, "resume", False)) if save_model else False

    # Only create the run directory if we will actually write something (plots or model artifacts).
    if make_plots or save_model:
        out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # B) Read metadata + validate invariants
    # ----------------------------
    meta = load_metadata(tfrecord_root)
    shapes = meta["example_shapes_by_nz"][str(Nz)]
    H, W, Cx = shapes["x"]

    inputs = tuple(cfg_iceflow.unified.inputs)
    _validate_pretraining_setup(inputs=inputs, Cx=Cx, cfg_physics=cfg_physics, state=state)

    # ----------------------------
    # C) Directories / resume checks (only if save_model; avoids sweep collisions)
    # ----------------------------
    if save_model:
        ckpt_dir, fig_dir = _prepare_run_dirs(out_dir=out_dir, resume=resume)
    else:
        ckpt_dir = out_dir / "checkpoints"
        fig_dir = out_dir / "figures"
        if make_plots and (make_plots or save_model):
            fig_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # D) Datasets
    # ----------------------------
    train_files = list_shards(tfrecord_root, Nz, split="train")
    val_files   = list_shards(tfrecord_root, Nz, split="val")

    rng = random.Random(getattr(cfg_pretraining, "split_seed", 0))
    rng.shuffle(train_files)  # shuffle train only; val can stay deterministic

    train_ds, val_ds = make_datasets(
        train_files=train_files,
        val_files=val_files,
        H=H, W=W, Nz=Nz,
        compression="GZIP",
        batch_size=cfg_pretraining.batch_size,
    )

    # ----------------------------
    # E) Mapping / optimizer / loss pieces
    # ----------------------------
    mapping_args = InterfaceMappings["network"].get_mapping_args(cfg, state)
    mapping = Mappings["network"](**mapping_args)
    state.iceflow.mapping = mapping

    # ----------------------------
    # E2) Compute normalization stats ONCE (Keras), then attach FixedChannelStandardization (forward pass)
    #     Manifest interactions are optional (save_model flag).
    # ----------------------------
    manifest_path = out_dir / "manifest.yaml"
    desired_dtype = normalize_precision(cfg.processes.iceflow.numerics.precision)

    if not resume:
        tmp = tf.keras.layers.Normalization(axis=-1, dtype=tf.float64)
        tmp.adapt(train_ds.map(lambda x, y: x, num_parallel_calls=tf.data.AUTOTUNE).take(2000))

        # Force one batch materialization so data pipeline is “ready”
        _ = next(iter(train_ds))

        mean_1d = tmp.mean.numpy().reshape(-1).astype(np.float64)
        var_1d  = tmp.variance.numpy().reshape(-1).astype(np.float64)
        eps = 1e-7

        print(f"[norm-stats] computed once: mean={mean_1d} var={var_1d}")

        state.iceflow_model.input_normalizer = FixedChannelStandardization(
            mean_1d=mean_1d,
            var_1d=var_1d,
            epsilon=eps,
            dtype=desired_dtype,
            name="input_norm",
        )
        _ = state.iceflow_model.input_normalizer(tf.zeros((1, H, W, Cx), dtype=desired_dtype))

        # Write schema v3 manifest immediately (optional)
        if save_model:
            if not manifest_path.exists():
                dummy_x = tf.zeros((1, H, W, Cx), dtype=desired_dtype)
                y0 = state.iceflow_model(dummy_x, training=False)
                nb_outputs = int(y0.shape[-1])

                manifest_v3 = build_manifest_v3(
                    cfg=cfg,
                    model=state.iceflow_model,
                    inputs=list(inputs),
                    nb_outputs=nb_outputs,
                )
                manifest_path.write_text(yaml.safe_dump(manifest_v3.to_dict(), sort_keys=False))
                print(f"[manifest] wrote schema v3 {manifest_path}")
            else:
                print(f"[manifest] exists, not overwriting: {manifest_path}")

    else:
        # This block is only reachable if save_model=True (resume forced off otherwise).
        print("[norm] resume=True: will attach FixedChannelStandardization from manifest after checkpoint restore")

    opt = tf.keras.optimizers.Adam(learning_rate=cfg_pretraining.learning_rate)
    physics_cost_fn = get_cost_fn(cfg, state)

    lt = cfg_pretraining.loss_type.lower()
    if lt not in ("mse", "huber"):
        raise ValueError(f"loss_type must be 'mse' or 'huber', got {lt!r}")

    if lt == "huber":
        delta = float(getattr(cfg_pretraining, "huber_delta", 50.0))
        huber = tf.keras.losses.Huber(delta=delta, reduction=tf.keras.losses.Reduction.NONE)

    def compute_losses(x_batch: tf.Tensor, y_batch: tf.Tensor, in_warmup: tf.Tensor):
        U, V = mapping.get_UV(x_batch)
        Ut, Vt = y_batch[..., 0], y_batch[..., 1]

        if lt == "huber":
            data_loss = tf.reduce_mean(huber(Ut, U) + huber(Vt, V))
        else:
            data_loss = tf.reduce_mean(tf.square(U - Ut) + tf.square(V - Vt))

        phys_loss = tf.cond(
            in_warmup,
            lambda: tf.zeros((), dtype=data_loss.dtype),
            lambda: tf.cast(physics_cost_fn(U, V, x_batch), data_loss.dtype),
        )
        return data_loss, phys_loss

    def safe_global_norm(grads):
        gs = [g for g in grads if g is not None]
        return tf.linalg.global_norm(gs) if gs else tf.constant(0.0, tf.float32)

    EMA          = tf.constant(0.99, tf.float32)
    UPDATE_EVERY = tf.constant(100, tf.int64)
    LAM_MIN      = tf.constant(1e-3, tf.float32)
    LAM_MAX      = tf.constant(1e3, tf.float32)
    EPS          = tf.constant(1e-6, tf.float32)
    WARMUP_STEPS = tf.constant(100000, tf.int64)

    step = tf.Variable(0, trainable=False, dtype=tf.int64, name="step")
    lambda_phys = tf.Variable(0.1, trainable=False, dtype=tf.float32, name="lambda_phys")

    # ----------------------------
    # F) Metrics
    # ----------------------------
    metrics = MetricsBundle(
        train_total=tf.keras.metrics.Mean(name="train_total"),
        train_data=tf.keras.metrics.Mean(name="train_data"),
        train_phys=tf.keras.metrics.Mean(name="train_phys"),
        train_lam=tf.keras.metrics.Mean(name="lambda_phys"),
        val_total=tf.keras.metrics.Mean(name="val_total"),
        val_data=tf.keras.metrics.Mean(name="val_data"),
        val_phys=tf.keras.metrics.Mean(name="val_phys"),
    )

    # ----------------------------
    # G) Train/val steps
    # ----------------------------
    @tf.function(reduce_retracing=True, jit_compile=False)
    def train_step(x_batch: tf.Tensor, y_batch: tf.Tensor):
        vars_ = state.iceflow_model.trainable_variables
        step.assign_add(1)

        in_warmup = step <= WARMUP_STEPS
        do_update = tf.equal(step % UPDATE_EVERY, 0)

        tf.debugging.assert_all_finite(x_batch, "train: x_batch has NaN/Inf")
        tf.debugging.assert_all_finite(y_batch, "train: y_batch has NaN/Inf")

        def update_branch():
            with tf.GradientTape(persistent=True) as tape:
                data_loss, phys_loss = compute_losses(x_batch, y_batch, in_warmup)

            g_data = tape.gradient(data_loss, vars_)
            g_phys = tape.gradient(phys_loss, vars_)
            del tape

            norm_data = safe_global_norm(g_data)
            norm_phys = safe_global_norm(g_phys)

            lam_hat = norm_data / (norm_phys + EPS)

            MAX_UP   = tf.constant(2.0, tf.float32)
            MAX_DOWN = tf.constant(2.0, tf.float32)
            lam_hat = tf.clip_by_value(lam_hat, lambda_phys / MAX_DOWN, lambda_phys * MAX_UP)

            lam_new = EMA * lambda_phys + (1.0 - EMA) * tf.stop_gradient(lam_hat)
            lam_new = tf.clip_by_value(lam_new, LAM_MIN, LAM_MAX)
            lambda_phys.assign(lam_new)

            grads = []
            for gd, gp in zip(g_data, g_phys):
                if gd is None and gp is None:
                    grads.append(None)
                elif gd is None:
                    grads.append(lam_new * gp)
                elif gp is None:
                    grads.append(gd)
                else:
                    grads.append(gd + lam_new * gp)

            opt.apply_gradients([(g, v) for g, v in zip(grads, vars_) if g is not None])
            total_loss = data_loss + tf.cast(lam_new, data_loss.dtype) * tf.cast(phys_loss, data_loss.dtype)
            return data_loss, phys_loss, total_loss, lam_new

        def normal_branch():
            with tf.GradientTape() as tape:
                data_loss, phys_loss = compute_losses(x_batch, y_batch, in_warmup)
                total_loss = tf.cond(
                    in_warmup,
                    lambda: data_loss,
                    lambda: data_loss + tf.cast(lambda_phys, data_loss.dtype) * tf.cast(phys_loss, data_loss.dtype),
                )
                lam = tf.where(in_warmup, tf.constant(0.0, tf.float32), lambda_phys)

            grads = tape.gradient(total_loss, vars_)
            opt.apply_gradients([(g, v) for g, v in zip(grads, vars_) if g is not None])
            return data_loss, phys_loss, total_loss, lam

        data_loss, phys_loss, total_loss, lam = tf.cond(
            in_warmup,
            normal_branch,
            lambda: tf.cond(do_update, update_branch, normal_branch),
        )

        metrics.train_data.update_state(data_loss)
        metrics.train_phys.update_state(phys_loss)
        metrics.train_total.update_state(total_loss)
        metrics.train_lam.update_state(lam)

    @tf.function(reduce_retracing=True, jit_compile=False)
    def val_step(x_batch: tf.Tensor, y_batch: tf.Tensor):
        data_loss, phys_loss = compute_losses(x_batch, y_batch, in_warmup=tf.constant(False))
        total_loss = data_loss + tf.cast(lambda_phys, data_loss.dtype) * tf.cast(phys_loss, data_loss.dtype)
        metrics.val_data.update_state(data_loss)
        metrics.val_phys.update_state(phys_loss)
        metrics.val_total.update_state(total_loss)

    # ----------------------------
    # H) Checkpointing + optional resume restore (optional via save_model)
    # ----------------------------
    ckpt_mgr: Optional[tf.train.CheckpointManager] = None
    if save_model:
        ckpt = tf.train.Checkpoint(
            step=step,
            optimizer=opt,
            model=state.iceflow_model,
            lambda_phys=lambda_phys,
        )
        ckpt_mgr = tf.train.CheckpointManager(ckpt, str(ckpt_dir), max_to_keep=3)

        if resume:
            latest = ckpt_mgr.latest_checkpoint
            if not latest:
                raise FileNotFoundError(f"resume=True but no checkpoints found in {ckpt_dir}")

            desired_dtype = normalize_precision(cfg.processes.iceflow.numerics.precision)
            dummy_x = tf.zeros((1, H, W, Cx), dtype=desired_dtype)
            _ = state.iceflow_model(dummy_x, training=False)
            if hasattr(opt, "build"):
                opt.build(state.iceflow_model.trainable_variables)

            status = ckpt.restore(latest)
            status.assert_existing_objects_matched()

            if not manifest_path.exists():
                raise FileNotFoundError(
                    f"resume=True but {manifest_path} not found. "
                    "You chose manifest as source of truth; it must exist."
                )

            raw = yaml.safe_load(manifest_path.read_text())
            p = raw["normalization"]["params"]
            mean_1d = np.asarray(p["mean_1d"], dtype=np.float64)
            var_1d  = np.asarray(p["var_1d"],  dtype=np.float64)
            eps = float(p.get("epsilon", p.get("variance_epsilon", 1e-7)))

            desired_dtype = normalize_precision(cfg.processes.iceflow.numerics.precision)
            norm2 = FixedChannelStandardization(
                mean_1d=mean_1d,
                var_1d=var_1d,
                epsilon=eps,
                dtype=desired_dtype,
                name="input_norm",
            )
            _ = norm2(tf.zeros((1, H, W, Cx), dtype=desired_dtype))

            state.iceflow_model.input_normalizer = norm2
            print("[norm] reapplied fixed normalizer from manifest")
            print("[norm] reapplied from manifest mean=", mean_1d)
            print("[norm] reapplied from manifest var= ", var_1d)

            (
                start_epoch,
                train_total_hist, val_total_hist,
                train_data_hist,  val_data_hist,
                train_phys_hist,  val_phys_hist,
                lambda_hist,
            ) = load_history_yaml(out_dir)

            history = HistoryBundle(
                train_total=train_total_hist,
                val_total=val_total_hist,
                train_data=train_data_hist,
                val_data=val_data_hist,
                train_phys=train_phys_hist,
                val_phys=val_phys_hist,
                lambda_phys=lambda_hist,
            )
        else:
            start_epoch = 0
            history = _init_empty_histories()
    else:
        start_epoch = 0
        history = _init_empty_histories()

    if start_epoch > int(cfg_pretraining.epochs):
        raise ValueError(
            f"history.yaml says epoch={start_epoch} but cfg_pretraining.epochs={cfg_pretraining.epochs}."
        )

    # ----------------------------
    # I) Visual sampling iterator (optional)
    # ----------------------------
    if make_plots:
        val_vis_ds = (
            val_ds.unbatch()
            .shuffle(4096, reshuffle_each_iteration=True)
            .batch(cfg_pretraining.batch_size, drop_remainder=True)
        )
        val_vis_it = iter(val_vis_ds.repeat())
        fig_dir.mkdir(parents=True, exist_ok=True)
    else:
        val_vis_it = None  # never used when make_plots=False

    # ----------------------------
    # J) Training loop
    # ----------------------------
    ctx = LoopContext(
        start_epoch=start_epoch,
        n_epochs=cfg_pretraining.epochs,
        train_ds=train_ds,
        val_ds=val_ds,
        val_vis_it=val_vis_it,
        train_step=train_step,
        val_step=val_step,
        mapping=mapping,
        Nz=Nz,
        fig_dir=fig_dir,
        ckpt_mgr=ckpt_mgr,
        out_dir=out_dir,
        make_plots=make_plots,
        save_model=save_model,
    )

    _run_training_loop(ctx=ctx, metrics=metrics, history=history)

    # ----------------------------
    # K) Export + score (export optional)
    # ----------------------------
    if save_model:
        save_emulator_artifact(
            artifact_dir=out_dir,
            cfg=cfg,
            model=state.iceflow_model,
            inputs=list(inputs),
        )
        print(f"[export] saved emulator artifact to {out_dir}")

    k = min(5, len(history.val_total))
    state.score = float(np.mean(history.val_total[-k:]))