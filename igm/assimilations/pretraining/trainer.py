#!/usr/bin/env python3
# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import tensorflow as tf

from .history import History, load_history_yaml, save_history_yaml
from .plots import save_loss_plot, save_speed_compare
from .training_utils import build_velocity_data_loss


class Trainer:
    """
    Single-emulator pretraining loop with adaptive physics weighting and
    gradient accumulation.

    """

    def __init__(
        self,
        cfg: Any,
        model: tf.keras.Model,
        mapping: Any,
        physics_cost_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        *,
        out_dir: Path,
        ckpt_dir: Path,
        fig_dir: Path,
        n_epochs: int,
        accum_steps: int,
        Nz: int,
        save_model: bool,
        make_plots: bool,
    ):
        self.cfg = cfg
        self.model = model
        self.mapping = mapping
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.out_dir = out_dir
        self.ckpt_dir = ckpt_dir
        self.fig_dir = fig_dir
        self.n_epochs = int(n_epochs)
        self.accum_steps = int(accum_steps)
        self.Nz = int(Nz)
        self.save_model = bool(save_model)
        self.make_plots = bool(make_plots)

        cfg_p = cfg.assimilations.pretraining

        # ------------------------------------------------------------------
        # Hyperparameters
        # ------------------------------------------------------------------
        self.steps_per_epoch = int(cfg_p.steps_per_epoch)
        self.val_steps       = int(cfg_p.val_steps)

        self._EMA           = tf.constant(float(cfg_p.lambda_ema),        tf.float32)
        self._LAM_MIN       = tf.constant(float(cfg_p.lambda_min),        tf.float32)
        self._LAM_MAX       = tf.constant(float(cfg_p.lambda_max),        tf.float32)
        self._MAX_CHANGE    = tf.constant(float(cfg_p.lambda_max_change), tf.float32)
        self._RHO           = tf.constant(float(cfg_p.lambda_subordinacy), tf.float32)

        # Which gradient statistic drives the lambda target:
        #   "norm" -> global L2 norm ratio (data/phys gradients made equal magnitude)
        #   "std"  -> per-parameter gradient standard deviation ratio (inverse-Dirichlet,
        #             Maddu et al. 2022); more robust to heavy-tailed physics gradients.
        self._WEIGHT_STAT = str(cfg_p.lambda_weight_stat).lower()
        if self._WEIGHT_STAT not in ("norm", "std"):
            raise ValueError(
                f"lambda_weight_stat must be 'norm' or 'std', got {cfg_p.lambda_weight_stat!r}"
            )
        self._EPS           = tf.constant(1e-6, tf.float32)
        self._UPDATE_EVERY  = tf.constant(int(cfg_p.lambda_update_every), tf.int64)
        self._WARMUP_STEPS  = tf.constant(int(cfg_p.warmup_steps),        tf.int64)
        self._ACCUM_STEPS   = tf.constant(self.accum_steps, tf.int64)
        self._ACCUM_STEPS_F = tf.cast(self._ACCUM_STEPS, tf.float32)

        # ------------------------------------------------------------------
        # Optimizer / training state
        # ------------------------------------------------------------------
        # Architecture-agnostic safety net against physics-loss-driven blow-ups:
        # a global-norm clip on the (post-accumulation) applied gradient bounds
        # the per-step weight update without distorting gradient *direction*, so
        # it protects unconstrained backbones (e.g. a vanilla CNN) while leaving
        # well-behaved training untouched when the clip never binds. A value of
        # <= 0 disables clipping. See _apply for the finite-guard that drops
        # non-finite updates (clipnorm alone does not sanitise NaN/Inf).
        self._GRAD_CLIP = float(cfg_p.grad_clip_norm)
        clipnorm = self._GRAD_CLIP if self._GRAD_CLIP > 0.0 else None
        self.opt = tf.keras.optimizers.Adam(
            learning_rate=cfg_p.learning_rate, clipnorm=clipnorm
        )
        self.step        = tf.Variable(0, trainable=False, dtype=tf.int64,   name="step")
        self.lambda_phys = tf.Variable(0.1, trainable=False, dtype=tf.float32, name="lambda_phys")
        # Cumulative count of optimizer updates skipped due to non-finite grads.
        self.skipped_updates = tf.Variable(0, trainable=False, dtype=tf.int64, name="skipped_updates")

        # ------------------------------------------------------------------
        # Gradient + loss accumulators
        # ------------------------------------------------------------------
        self._vars = model.trainable_variables
        self._accum_g      = [tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False, name=f"accum_g_{i}")
                              for i, v in enumerate(self._vars)]
        self._accum_g_phys = [tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False, name=f"accum_gp_{i}")
                              for i, v in enumerate(self._vars)]
        self._accum_data_loss = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="accum_data_loss")
        self._accum_phys_loss = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="accum_phys_loss")
        self._accum_count     = tf.Variable(0,   trainable=False, dtype=tf.int64,   name="accum_count")

        # ------------------------------------------------------------------
        # Metrics
        # ------------------------------------------------------------------
        self.m_train_total = tf.keras.metrics.Mean(name="train_total")
        self.m_train_data  = tf.keras.metrics.Mean(name="train_data")
        self.m_train_phys  = tf.keras.metrics.Mean(name="train_phys")
        self.m_train_lam   = tf.keras.metrics.Mean(name="lambda_phys")
        self.m_train_gnorm = tf.keras.metrics.Mean(name="train_gnorm")
        self.m_val_total   = tf.keras.metrics.Mean(name="val_total")
        self.m_val_data    = tf.keras.metrics.Mean(name="val_data")
        self.m_val_phys    = tf.keras.metrics.Mean(name="val_phys")
        self._metrics_all = [
            self.m_train_total, self.m_train_data, self.m_train_phys, self.m_train_lam,
            self.m_train_gnorm,
            self.m_val_total,   self.m_val_data,   self.m_val_phys,
        ]

        # ------------------------------------------------------------------
        # Loss functions
        # ------------------------------------------------------------------
        self._data_loss_fn = build_velocity_data_loss(
            loss_type=cfg_p.loss_type,
            huber_delta=float(cfg_p.huber_delta),
        )
        self._physics_cost_fn = physics_cost_fn

        # ------------------------------------------------------------------
        # History + checkpoint
        # ------------------------------------------------------------------
        self.history = History()

        self.ckpt = tf.train.Checkpoint(
            step=self.step,
            optimizer=self.opt,
            model=self.model,
            lambda_phys=self.lambda_phys,
        )
        self.ckpt_mgr: Optional[tf.train.CheckpointManager] = (
            tf.train.CheckpointManager(self.ckpt, str(ckpt_dir), max_to_keep=3)
            if save_model else None
        )

        # Visual sampling iterator for per-epoch speed-compare plots.
        if make_plots:
            micro_bs = int(cfg_p.micro_batch_size)
            val_vis_ds = (
                val_ds.unbatch()
                .shuffle(4096, reshuffle_each_iteration=True)
                .batch(micro_bs, drop_remainder=True)
            )
            self._val_vis_it = iter(val_vis_ds.repeat())
        else:
            self._val_vis_it = None

        self._build_train_step()
        self._build_val_step()

    def restore_from_checkpoint(self) -> int:
        """Restore weights, optimizer, step, lambda_phys, and history. Returns start_epoch."""
        if self.ckpt_mgr is None:
            raise RuntimeError("restore_from_checkpoint requires save_model=True")
        latest = self.ckpt_mgr.latest_checkpoint
        if not latest:
            raise FileNotFoundError(f"No checkpoints found in {self.ckpt_dir}")

        if hasattr(self.opt, "build"):
            self.opt.build(self.model.trainable_variables)

        status = self.ckpt.restore(latest)
        status.assert_existing_objects_matched()

        start_epoch, self.history = load_history_yaml(self.out_dir)
        return start_epoch

    def run(self, start_epoch: int = 0) -> None:
        if start_epoch > self.n_epochs:
            raise ValueError(
                f"start_epoch={start_epoch} exceeds cfg.assimilations.pretraining.epochs={self.n_epochs}"
            )
        train_it = iter(self.train_ds)

        for epoch in range(start_epoch, self.n_epochs):
            for m in self._metrics_all:
                m.reset_state()

            self._train_epoch(train_it)
            self._validate()
            self._record_and_log(epoch)
            self._maybe_plot(epoch)
            self._maybe_save_step(epoch)

    def final_score(self) -> float:
        k = min(5, len(self.history.val_total))
        if k == 0:
            return float("nan")
        return float(np.mean(self.history.val_total[-k:]))

    # ----------------------------------------------------------------------
    # Loss + accumulator helpers
    # ----------------------------------------------------------------------
    def _compute_losses(self, x_batch, y_batch, in_warmup):
        U, V = self.mapping.get_UV(x_batch)
        data_loss = tf.cast(self._data_loss_fn(U, V, y_batch), U.dtype)
        phys_loss = tf.cond(
            in_warmup,
            lambda: tf.zeros((), dtype=data_loss.dtype),
            lambda: tf.cast(self._physics_cost_fn(U, V, x_batch), data_loss.dtype),
        )
        return data_loss, phys_loss

    def _zero_accumulators(self):
        for ag in self._accum_g:
            ag.assign(tf.zeros_like(ag))
        for ap in self._accum_g_phys:
            ap.assign(tf.zeros_like(ap))
        self._accum_data_loss.assign(0.0)
        self._accum_phys_loss.assign(0.0)
        self._accum_count.assign(0)

    def _grad_stat(self, buf):
        """Scalar gradient-size statistic for a list of accumulated grad buffers."""
        if self._WEIGHT_STAT == "std":
            # Inverse-Dirichlet: std of the flattened gradient vector over all params.
            flat = tf.concat([tf.reshape(tf.convert_to_tensor(v), [-1]) for v in buf], axis=0)
            return tf.math.reduce_std(flat)
        return tf.linalg.global_norm([tf.convert_to_tensor(v) for v in buf])

    def _compute_new_lambda(self):
        """Gradient-balanced lambda update, EMA-smoothed and clipped both ways.

        Target: lambda * stat(g_phys) = rho * stat(g_data), i.e. the physics
        gradient contribution is held at a fraction rho of the data gradient
        (rho=1 -> equal magnitude; rho<1 -> physics subordinate). stat(.) is
        either the L2 norm or the per-parameter std (see lambda_weight_stat).
        """
        stat_data = self._grad_stat(self._accum_g)
        stat_phys = self._grad_stat(self._accum_g_phys)

        lam_hat = self._RHO * stat_data / (stat_phys + self._EPS)
        lam_hat = tf.clip_by_value(
            lam_hat,
            self.lambda_phys / self._MAX_CHANGE,
            self.lambda_phys * self._MAX_CHANGE,
        )
        lam_new = self._EMA * self.lambda_phys + (1.0 - self._EMA) * tf.stop_gradient(lam_hat)
        lam_new = tf.clip_by_value(lam_new, self._LAM_MIN, self._LAM_MAX)
        # Guard against a blown-up batch poisoning lambda: if the gradient norms
        # are non-finite (e.g. an exploding physics loss), keep the previous
        # value rather than latching NaN/Inf into lambda for the rest of training.
        lam_new = tf.where(tf.math.is_finite(lam_new), lam_new, self.lambda_phys)
        self.lambda_phys.assign(lam_new)
        return lam_new

    def _build_train_step(self):
        accum_g, accum_g_phys = self._accum_g, self._accum_g_phys
        accum_data_loss, accum_phys_loss = self._accum_data_loss, self._accum_phys_loss
        accum_count = self._accum_count
        step = self.step
        lambda_phys = self.lambda_phys
        vars_ = self._vars
        opt = self.opt
        WARMUP_STEPS = self._WARMUP_STEPS
        UPDATE_EVERY = self._UPDATE_EVERY
        ACCUM_STEPS = self._ACCUM_STEPS
        ACCUM_STEPS_F = self._ACCUM_STEPS_F
        compute_losses = self._compute_losses
        compute_new_lambda = self._compute_new_lambda
        zero_accumulators = self._zero_accumulators
        m_train_total = self.m_train_total
        m_train_data  = self.m_train_data
        m_train_phys  = self.m_train_phys
        m_train_lam   = self.m_train_lam
        m_train_gnorm = self.m_train_gnorm
        skipped_updates = self.skipped_updates

        def _add_into(buffer, grads):
            for ag, g in zip(buffer, grads):
                if g is not None:
                    ag.assign_add(tf.cast(g, tf.float32))

        @tf.function(reduce_retracing=True, jit_compile=False)
        def train_step(x_batch, y_batch):
            next_step = step + 1
            cycle_in_warmup = next_step <= WARMUP_STEPS
            cycle_do_update = tf.logical_and(
                tf.logical_not(cycle_in_warmup),
                tf.equal(next_step % UPDATE_EVERY, 0),
            )

            tf.debugging.assert_all_finite(x_batch, "train: x_batch has NaN/Inf")
            tf.debugging.assert_all_finite(y_batch, "train: y_batch has NaN/Inf")

            # Persistent tape is reserved for lambda-updates where we
            # need data and physics gradients separately. Warmup + normal
            # cycles use the cheaper single-pass tape on a precombined total.
            def _accum_persistent():
                with tf.GradientTape(persistent=True) as tape:
                    dl, pl = compute_losses(x_batch, y_batch, in_warmup=tf.constant(False))
                _add_into(accum_g,      tape.gradient(dl, vars_))
                _add_into(accum_g_phys, tape.gradient(pl, vars_))
                return dl, pl

            def _accum_cheap():
                with tf.GradientTape() as tape:
                    dl, pl = compute_losses(x_batch, y_batch, in_warmup=cycle_in_warmup)
                    total = tf.cond(
                        cycle_in_warmup,
                        lambda: dl,
                        lambda: dl + tf.cast(lambda_phys, dl.dtype) * tf.cast(pl, dl.dtype),
                    )
                _add_into(accum_g, tape.gradient(total, vars_))
                return dl, pl

            dl, pl = tf.cond(cycle_do_update, _accum_persistent, _accum_cheap)
            accum_data_loss.assign_add(tf.cast(dl, tf.float32))
            accum_phys_loss.assign_add(tf.cast(pl, tf.float32))
            accum_count.assign_add(1)

            # --- Apply when accumulation is full ---
            def _apply():
                # Effective lambda for this update:
                #   warmup    -> 0  (physics ignored)
                #   normal    -> current lambda_phys
                #   update    -> freshly computed lambda_new (also assigned)
                lam_used = tf.cond(
                    cycle_in_warmup,
                    lambda: tf.constant(0.0, tf.float32),
                    lambda: tf.cond(
                        cycle_do_update,
                        compute_new_lambda,
                        lambda: tf.identity(lambda_phys),
                    ),
                )

                grads_avg = [
                    (ag + lam_used * ap) / ACCUM_STEPS_F
                    for ag, ap in zip(accum_g, accum_g_phys)
                ]

                # Pre-clip global norm: also our non-finite detector, since
                # global_norm propagates NaN/Inf. The optimizer applies the
                # actual clipnorm; here we only gate apply on finiteness so a
                # blown-up batch cannot corrupt the weights in a single step.
                gnorm = tf.linalg.global_norm(grads_avg)
                all_finite = tf.math.is_finite(gnorm)

                d_avg = accum_data_loss / ACCUM_STEPS_F
                p_avg = accum_phys_loss / ACCUM_STEPS_F
                t_avg = d_avg + lam_used * p_avg
                w = tf.cast(ACCUM_STEPS, tf.float32)

                def _do_apply():
                    opt.apply_gradients([
                        (tf.cast(g, v.dtype), v) for g, v in zip(grads_avg, vars_)
                    ])
                    m_train_gnorm.update_state(gnorm, sample_weight=w)
                    return tf.constant(0)

                def _skip_apply():
                    skipped_updates.assign_add(1)
                    return tf.constant(0)

                tf.cond(all_finite, _do_apply, _skip_apply)

                # Only record loss metrics for finite steps to keep epoch
                # averages meaningful when a blow-up batch is dropped.
                def _record_losses():
                    m_train_data.update_state(d_avg, sample_weight=w)
                    m_train_phys.update_state(p_avg, sample_weight=w)
                    m_train_total.update_state(t_avg, sample_weight=w)
                    m_train_lam.update_state(lam_used, sample_weight=w)
                    return tf.constant(0)

                tf.cond(all_finite, _record_losses, lambda: tf.constant(0))

                step.assign_add(1)
                zero_accumulators()
                return tf.constant(0)

            tf.cond(tf.equal(accum_count, ACCUM_STEPS), _apply, lambda: tf.constant(0))

        self._train_step = train_step

    def _build_val_step(self):
        compute_losses = self._compute_losses
        lambda_phys = self.lambda_phys
        m_val_data  = self.m_val_data
        m_val_phys  = self.m_val_phys
        m_val_total = self.m_val_total

        @tf.function(reduce_retracing=True, jit_compile=False)
        def val_step(x_batch, y_batch):
            data_loss, phys_loss = compute_losses(x_batch, y_batch, in_warmup=tf.constant(False))
            total_loss = data_loss + tf.cast(lambda_phys, data_loss.dtype) * tf.cast(phys_loss, data_loss.dtype)
            m_val_data.update_state(data_loss)
            m_val_phys.update_state(phys_loss)
            m_val_total.update_state(total_loss)

        self._val_step = val_step

    # ----------------------------------------------------------------------
    # Epoch helpers
    # ----------------------------------------------------------------------
    def _train_epoch(self, train_it):
        micro_steps = self.steps_per_epoch * self.accum_steps
        for _ in range(micro_steps):
            x_b, y_b = next(train_it)
            self._train_step(x_b, y_b)

    def _validate(self):
        val_it = iter(self.val_ds)
        for _ in range(self.val_steps):
            x_b, y_b = next(val_it)
            self._val_step(x_b, y_b)

    def _record_and_log(self, epoch: int) -> None:
        tt = float(self.m_train_total.result().numpy())
        td = float(self.m_train_data.result().numpy())
        tp = float(self.m_train_phys.result().numpy())
        lam = float(self.m_train_lam.result().numpy())
        vt = float(self.m_val_total.result().numpy())
        vd = float(self.m_val_data.result().numpy())
        vp = float(self.m_val_phys.result().numpy())
        self.history.append_epoch(
            train_total=tt, val_total=vt,
            train_data=td,  val_data=vd,
            train_phys=tp,  val_phys=vp,
            lambda_phys=lam,
        )
        gn = float(self.m_train_gnorm.result().numpy())
        n_skip = int(self.skipped_updates.numpy())
        print(
            f"[epoch {epoch + 1}/{self.n_epochs}] "
            f"train_total={tt:.6e} train_data={td:.6e} train_phys={tp:.6e} "
            f"lambda_phys={lam:.3e} val_total={vt:.6e} "
            f"grad_norm={gn:.3e} skipped={n_skip}"
        )

    def _maybe_plot(self, epoch: int) -> None:
        if not self.make_plots:
            return
        save_loss_plot(
            self.history.train_total, self.history.val_total,
            self.history.train_data,  self.history.val_data,
            self.history.train_phys,  self.history.val_phys,
            self.history.lambda_phys,
            self.fig_dir / "loss_curve.png",
        )
        x_vis, y_vis = next(self._val_vis_it)
        save_speed_compare(
            self.mapping, x_vis, y_vis, self.Nz,
            self.fig_dir / f"speed_compare_epoch{epoch + 1:04d}.png",
        )

    def _maybe_save_step(self, epoch: int) -> None:
        if not self.save_model:
            return
        if self.ckpt_mgr is not None:
            self.ckpt_mgr.save()
        save_history_yaml(self.out_dir, epoch=epoch + 1, history=self.history)
