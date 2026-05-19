#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Any, Callable, Optional


import numpy as np
import tensorflow as tf

from igm.processes.iceflow.unified.mappings import Mappings, InterfaceMappings
from igm.processes.iceflow.emulate.utils.artifacts import (
    load_emulator_artifact,
    save_emulator_artifact,
    wrap_emulator_artifact,
)
from igm.processes.pretraining.cost_tmp import get_cost_fn
from igm.utils.math.precision import normalize_precision

# Best-guess local imports; adjust paths as needed in your repo
from .history import load_history_yaml, save_history_yaml
from .plots import save_loss_plot, save_speed_compare
from .training_utils import build_tfrecord_datasets_for_nz, build_velocity_data_loss


def update(cfg, state):
    pass


def finalize(cfg, state):
    pass

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
    accum_steps: int 


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
    
    TRAIN_UPDATES = 1000 # optimizer updates per epoch
    VAL_STEPS     = 50

    # We must feed micro-batches to train_step; one optimizer update happens every ctx.accum_steps micro-batches
    TRAIN_MICRO_STEPS = TRAIN_UPDATES * int(ctx.accum_steps)

    train_it = iter(ctx.train_ds)  # infinite because of .repeat()

    for epoch in range(ctx.start_epoch, ctx.n_epochs):

        _reset_metrics(metrics)

        # --- train ---
        for _ in range(TRAIN_MICRO_STEPS):
            x_b, y_b = next(train_it)
            ctx.train_step(x_b, y_b)

        # --- validate ---
        val_it = iter(ctx.val_ds)
        for _ in range(VAL_STEPS):
            x_b, y_b = next(val_it)
            ctx.val_step(x_b, y_b)

        tt, td, tp, lam = _append_epoch(history, metrics)

        print(
            f"[epoch {epoch+1}/{ctx.n_epochs}] "
            f"train_total={tt:.6e} "
            f"train_data={td:.6e} "
            f"train_phys={tp:.6e} "
            f"lambda_phys={lam:.3e} "
            f"val_total={history.val_total[-1]:.6e}"
        )

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
    Nz = cfg_iceflow.numerics.Nz

    make_plots = bool(getattr(cfg_pretraining, "make_plots", True))
    save_model = bool(getattr(cfg_pretraining, "save_model", True))

    tfrecord_root = Path(cfg_pretraining.data_dir)

    out_dir = Path(cfg_pretraining.out_dir) / cfg_pretraining.experiment_name

    # If save_model=False, we force resume off to avoid collisions and confusion
    resume = bool(getattr(cfg_pretraining, "resume", False)) if save_model else False

    # Only create the run directory if we will actually write something (plots or model artifacts).
    if make_plots or save_model:
        out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # B) Read metadata + validate invariants
    # ----------------------------
    inputs = tuple(cfg_iceflow.unified.inputs)

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
    # D) Datasets + gradient accumulation sizes
    # ----------------------------
    effective_bs = int(cfg_pretraining.batch_size)
    micro_bs = int(getattr(cfg_pretraining, "micro_batch_size", effective_bs))

    if micro_bs <= 0 or effective_bs <= 0:
        raise ValueError(f"batch_size and micro_batch_size must be > 0, got batch_size={effective_bs}, micro_batch_size={micro_bs}")

    if micro_bs > effective_bs:
        raise ValueError(f"micro_batch_size ({micro_bs}) cannot exceed batch_size ({effective_bs})")

    if effective_bs % micro_bs != 0:
        raise ValueError(
            f"batch_size ({effective_bs}) must be divisible by micro_batch_size ({micro_bs}) "
            f"for clean accumulation, but {effective_bs} % {micro_bs} != 0"
        )

    accum_steps_py = effective_bs // micro_bs
    print(f"[grad-accum] effective_bs={effective_bs} micro_bs={micro_bs} accum_steps={accum_steps_py}")

    datasets = build_tfrecord_datasets_for_nz(
        tfrecord_root,
        nz=int(Nz),
        inputs=inputs,
        batch_size=micro_bs,
        compression="GZIP",
        split_seed=int(getattr(cfg_pretraining, "split_seed", 0)),
    )
    H, W, Cx = datasets.H, datasets.W, datasets.Cx
    train_ds = datasets.train_ds
    val_ds = datasets.val_ds

    # ----------------------------
    # E) Create model + gather mapping args
    # ----------------------------
    # get_mapping_args() constructs the non-old-format model from cfg when
    # pretrained=False. For resume runs we immediately replace that temporary
    # cfg-built skeleton with the saved Keras artifact before any restore occurs.
    mapping_args = InterfaceMappings["network"].get_mapping_args(cfg, state)

    # ----------------------------
    # E2) Attach final normalization / build final skeleton
    # ----------------------------
    desired_dtype = normalize_precision(cfg.processes.iceflow.numerics.precision)

    dummy_x = tf.zeros((1, H, W, Cx), dtype=desired_dtype)

    if resume:
        state.iceflow_model = load_emulator_artifact(artifact_dir=out_dir)
        mapping_args["network"] = state.iceflow_model
        print("[resume] loaded emulator.keras; checkpoint will restore weights/optimizer state")
    else:
        state.iceflow_model = wrap_emulator_artifact(state.iceflow_model)
        state.iceflow_model.input_normalizer.adapt(
            train_ds.map(lambda x, y: x, num_parallel_calls=tf.data.AUTOTUNE).take(2000)
        )
        print(
            f"[norm-stats] adapted: "
            f"mean={np.asarray(state.iceflow_model.input_normalizer.mean).reshape(-1)} "
            f"var={np.asarray(state.iceflow_model.input_normalizer.variance).reshape(-1)}"
        )

        state.iceflow_model.build(dummy_x.shape)
        _ = state.iceflow_model(dummy_x, training=False)
        mapping_args["network"] = state.iceflow_model

        if save_model:
            artifact_path = save_emulator_artifact(
                artifact_dir=out_dir,
                model=state.iceflow_model,
            )
            print(f"[artifact] wrote {artifact_path}")

    # ----------------------------
    # E3) Only now instantiate the Mapping
    # ----------------------------
    mapping = Mappings["network"](**mapping_args)
    state.iceflow.mapping = mapping

    opt = tf.keras.optimizers.Adam(learning_rate=cfg_pretraining.learning_rate)

    # If resuming and restoring optimizer state, ensure slots exist before restore:
    if save_model and resume and hasattr(opt, "build"):
        opt.build(state.iceflow_model.trainable_variables)

    physics_cost_fn = get_cost_fn(cfg, state)

    data_loss_fn = build_velocity_data_loss(
        loss_type=cfg_pretraining.loss_type,
        huber_delta=float(getattr(cfg_pretraining, "huber_delta", 50.0)),
    )

    def compute_losses(x_batch: tf.Tensor, y_batch: tf.Tensor, in_warmup: tf.Tensor):
        U, V = mapping.get_UV(x_batch)
        data_loss = tf.cast(data_loss_fn(U, V, y_batch), U.dtype)

        phys_loss = tf.cond(
            in_warmup,
            lambda: tf.zeros((), dtype=data_loss.dtype),
            lambda: tf.cast(physics_cost_fn(U, V, x_batch), data_loss.dtype),
        )
        return data_loss, phys_loss

    EMA          = tf.constant(0.99, tf.float32)
    UPDATE_EVERY = tf.constant(100, tf.int64)
    LAM_MIN      = tf.constant(1e-3, tf.float32)
    LAM_MAX      = tf.constant(float(getattr(cfg_pretraining, "lam_max", 100.0)), tf.float32)
    EPS          = tf.constant(1e-6, tf.float32)
    WARMUP_STEPS = tf.constant(100000, tf.int64)
    ACCUM_STEPS = tf.constant(accum_steps_py, tf.int64)
    ACCUM_STEPS_F = tf.cast(ACCUM_STEPS, tf.float32)

    accum_count = tf.Variable(0, trainable=False, dtype=tf.int64, name="accum_count")

    # Two gradient buffers:
    #   - accum_g: holds either total grads (normal cycles) OR data grads (lambda-update cycles)
    #   - accum_g_phys: holds phys grads only during lambda-update cycles (else remains zero)
    vars_ = state.iceflow_model.trainable_variables
    accum_g      = [tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False, name=f"accum_g_{i}")      for i, v in enumerate(vars_)]
    accum_g_phys = [tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False, name=f"accum_gp_{i}")     for i, v in enumerate(vars_)]

    # Loss accumulators (so metrics reflect the effective batch)
    accum_data_loss = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="accum_data_loss")
    accum_phys_loss = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="accum_phys_loss")

    def _zero_accumulators():
        for ag in accum_g:
            ag.assign(tf.zeros_like(ag))
        for ap in accum_g_phys:
            ap.assign(tf.zeros_like(ap))
        accum_data_loss.assign(0.0)
        accum_phys_loss.assign(0.0)
        accum_count.assign(0)

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
    def _global_norm_vars(var_list):
        # var_list are tf.Variables (tensors), never None here
        return tf.linalg.global_norm([tf.convert_to_tensor(v) for v in var_list])

    @tf.function(reduce_retracing=True, jit_compile=False)
    def train_step(x_batch: tf.Tensor, y_batch: tf.Tensor):
        # step counts OPTIMIZER UPDATES (effective batches)
        next_step = step + 1
        cycle_in_warmup = next_step <= WARMUP_STEPS
        cycle_do_update = tf.logical_and(
            tf.logical_not(cycle_in_warmup),
            tf.equal(next_step % UPDATE_EVERY, 0),
        )

        tf.debugging.assert_all_finite(x_batch, "train: x_batch has NaN/Inf")
        tf.debugging.assert_all_finite(y_batch, "train: y_batch has NaN/Inf")

        # there are two accum methods, one for warmup+normal cycles where we just accumulate total grads, and one for lambda-update cycles where we accumulate data and phys grads separately for the lambda update
        # this saves compute by not doing extra forward/backward passes during warmup and normal cycles, in these cases the total gradient can be calculated without a persistent tape because we don't need separate data vs phys grads for the lambda update
        # however for the lambda update cycles we do need separate data vs phys grads, because the weighting of the two is based on their relative magnitudes, so they have to be calculated independently and stored separately in accum_g and accum_g_phys

        def _accum_warmup_or_normal():
            with tf.GradientTape() as tape:
                dl, pl = compute_losses(x_batch, y_batch, in_warmup=cycle_in_warmup)
                total = tf.cond(
                    cycle_in_warmup,
                    lambda: dl,
                    lambda: dl + tf.cast(lambda_phys, dl.dtype) * tf.cast(pl, dl.dtype),
                )
            grads = tape.gradient(total, vars_)

            for ag, g in zip(accum_g, grads):
                if g is not None:
                    ag.assign_add(tf.cast(g, tf.float32))

            accum_data_loss.assign_add(tf.cast(dl, tf.float32))
            accum_phys_loss.assign_add(tf.cast(pl, tf.float32))
            return tf.constant(0)

        def _accum_for_lambda_update():
            # only called when cycle_do_update=True (so not warmup)
            with tf.GradientTape(persistent=True) as tape:
                dl, pl = compute_losses(x_batch, y_batch, in_warmup=tf.constant(False))
            g_data = tape.gradient(dl, vars_)
            g_phys = tape.gradient(pl, vars_)
            del tape

            for ag, gd in zip(accum_g, g_data):
                if gd is not None:
                    ag.assign_add(tf.cast(gd, tf.float32))

            for ap, gp in zip(accum_g_phys, g_phys):
                if gp is not None:
                    ap.assign_add(tf.cast(gp, tf.float32))

            accum_data_loss.assign_add(tf.cast(dl, tf.float32))
            accum_phys_loss.assign_add(tf.cast(pl, tf.float32))
            return tf.constant(0)

        tf.cond(
            cycle_do_update,
            _accum_for_lambda_update,
            _accum_warmup_or_normal,
        )

        accum_count.assign_add(1)

        def _apply_if_ready():
            def _apply_warmup():
                grads_avg = [ag / ACCUM_STEPS_F for ag in accum_g]
                opt.apply_gradients([(tf.cast(g, v.dtype), v) for g, v in zip(grads_avg, vars_)])

                d_avg = accum_data_loss / ACCUM_STEPS_F
                p_avg = accum_phys_loss / ACCUM_STEPS_F
                t_avg = d_avg

                metrics.train_data.update_state(d_avg, sample_weight=tf.cast(ACCUM_STEPS, tf.float32))
                metrics.train_phys.update_state(p_avg, sample_weight=tf.cast(ACCUM_STEPS, tf.float32))
                metrics.train_total.update_state(t_avg, sample_weight=tf.cast(ACCUM_STEPS, tf.float32))
                metrics.train_lam.update_state(0.0, sample_weight=tf.cast(ACCUM_STEPS, tf.float32))

                step.assign_add(1)
                _zero_accumulators()
                return tf.constant(0)

            def _apply_normal_no_lambda_update():
                grads_avg = [ag / ACCUM_STEPS_F for ag in accum_g]
                opt.apply_gradients([(tf.cast(g, v.dtype), v) for g, v in zip(grads_avg, vars_)])

                d_avg = accum_data_loss / ACCUM_STEPS_F
                p_avg = accum_phys_loss / ACCUM_STEPS_F
                lam_used = lambda_phys
                t_avg = d_avg + tf.cast(lam_used, tf.float32) * p_avg

                metrics.train_data.update_state(d_avg, sample_weight=tf.cast(ACCUM_STEPS, tf.float32))
                metrics.train_phys.update_state(p_avg, sample_weight=tf.cast(ACCUM_STEPS, tf.float32))
                metrics.train_total.update_state(t_avg, sample_weight=tf.cast(ACCUM_STEPS, tf.float32))
                metrics.train_lam.update_state(lam_used, sample_weight=tf.cast(ACCUM_STEPS, tf.float32))

                step.assign_add(1)
                _zero_accumulators()
                return tf.constant(0)

            def _apply_with_lambda_update():
                norm_data = _global_norm_vars(accum_g)
                norm_phys = _global_norm_vars(accum_g_phys)

                lam_hat = norm_data / (norm_phys + EPS)
                MAX_UP   = tf.constant(2.0, tf.float32)
                MAX_DOWN = tf.constant(2.0, tf.float32)
                lam_hat = tf.clip_by_value(lam_hat, lambda_phys / MAX_DOWN, lambda_phys * MAX_UP)

                lam_new = EMA * lambda_phys + (1.0 - EMA) * tf.stop_gradient(lam_hat)
                lam_new = tf.clip_by_value(lam_new, LAM_MIN, LAM_MAX)
                lambda_phys.assign(lam_new)

                grads_avg = [(ag + lam_new * ap) / ACCUM_STEPS_F for ag, ap in zip(accum_g, accum_g_phys)]
                opt.apply_gradients([(tf.cast(g, v.dtype), v) for g, v in zip(grads_avg, vars_)])

                d_avg = accum_data_loss / ACCUM_STEPS_F
                p_avg = accum_phys_loss / ACCUM_STEPS_F
                t_avg = d_avg + tf.cast(lam_new, tf.float32) * p_avg

                metrics.train_data.update_state(d_avg, sample_weight=tf.cast(ACCUM_STEPS, tf.float32))
                metrics.train_phys.update_state(p_avg, sample_weight=tf.cast(ACCUM_STEPS, tf.float32))
                metrics.train_total.update_state(t_avg, sample_weight=tf.cast(ACCUM_STEPS, tf.float32))
                metrics.train_lam.update_state(lam_new, sample_weight=tf.cast(ACCUM_STEPS, tf.float32))

                step.assign_add(1)
                _zero_accumulators()
                return tf.constant(0)

            return tf.cond(
                cycle_in_warmup,
                _apply_warmup,
                lambda: tf.cond(cycle_do_update, _apply_with_lambda_update, _apply_normal_no_lambda_update),
            )

        tf.cond(tf.equal(accum_count, ACCUM_STEPS), _apply_if_ready, lambda: tf.constant(0))

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

            status = ckpt.restore(latest)
            status.assert_existing_objects_matched()

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
            .batch(micro_bs, drop_remainder=True)
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
        accum_steps=accum_steps_py,
    )

    _run_training_loop(ctx=ctx, metrics=metrics, history=history)

    # ----------------------------
    # K) Export + score (export optional)
    # ----------------------------
    if save_model:
        artifact_path = save_emulator_artifact(
            artifact_dir=out_dir,
            model=state.iceflow_model,
        )
        print(f"[export] saved emulator artifact to {artifact_path}")

    k = min(5, len(history.val_total))
    state.score = float(np.mean(history.val_total[-k:]))
