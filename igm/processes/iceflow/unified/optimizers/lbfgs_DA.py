#!/usr/bin/env python3
# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from __future__ import annotations

import collections

import tensorflow as tf
from typing import Tuple, Optional
from ..halt import HaltStatus

from .lbfgs_bounds import OptimizerLBFGSBounds
from .line_searches import ValueAndGradient
from .da_progress_optimizer import _DAProgressOptimizer


LineSearchResult = collections.namedtuple(
    "LineSearchResult",
    ["alpha", "converged", "failed", "func_evals", "iterations", "left", "right"],
)


class OptimizerLBFGSBoundsDA(OptimizerLBFGSBounds):
    """
    Bounded L-BFGS for data assimilation with scale-aware rho spike clamping.
    """

    def __init__(
        self,
        *args,
        rho_spike_factor: float = 20.0,
        rho_warmup: int = 5,
        rho_ema_beta: float = 0.99,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        dtype = getattr(self.map, "precision", tf.float32)

        self.last_total = tf.Variable(0.0, trainable=False, dtype=dtype)
        self.last_data = tf.Variable(0.0, trainable=False, dtype=dtype)
        self.last_reg = tf.Variable(0.0, trainable=False, dtype=dtype)

        self.rho_mean = tf.Variable(0.0, trainable=False, dtype=dtype)
        self.rho_count = tf.Variable(0, trainable=False, dtype=tf.int64)

        self.rho_spike_factor = tf.constant(rho_spike_factor, dtype=dtype)
        self.rho_warmup = tf.constant(rho_warmup, dtype=tf.int64)
        self.rho_ema_beta = tf.constant(rho_ema_beta, dtype=dtype)

        # Cache this once to avoid repeated Python hasattr() checks in the hot path.
        self._line_search_supports_result = bool(hasattr(self.line_search, "search_result"))

        # swap display for DA-specific one, preserving enabled/freq
        self.display = _DAProgressOptimizer(
            enabled=self.display.enabled,
            freq=self.display.freq,
        )

        empty = tf.zeros([0], dtype=dtype)
        self.accepted_cost_total_hist = empty
        self.accepted_cost_data_hist = empty
        self.accepted_cost_reg_hist = empty

    def minimize(self, inputs: tf.Tensor) -> tf.Tensor:
        self.rho_mean.assign(tf.cast(0.0, self.rho_mean.dtype))
        self.rho_count.assign(0)
        empty = tf.zeros([0], dtype=self.last_total.dtype)
        self.accepted_cost_total_hist = empty
        self.accepted_cost_data_hist = empty
        self.accepted_cost_reg_hist = empty
        return super().minimize(inputs)

    def _publish_cost_history(
        self,
        cost_total_hist: tf.TensorArray,
        cost_data_hist: tf.TensorArray,
        cost_reg_hist: tf.TensorArray,
        n_keep: int,
    ) -> None:
        self.accepted_cost_total_hist = cost_total_hist.stack()[:n_keep]
        self.accepted_cost_data_hist = cost_data_hist.stack()[:n_keep]
        self.accepted_cost_reg_hist = cost_reg_hist.stack()[:n_keep]

    @tf.function(reduce_retracing=True)
    def _dot(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        acc = tf.tensordot(tf.cast(a, tf.float64), tf.cast(b, tf.float64), axes=1)
        return tf.cast(acc, self.precision)

    @tf.function(reduce_retracing=True)
    def _rho_cap(self) -> tf.Tensor:
        inf = tf.constant(float("inf"), dtype=self.rho_mean.dtype)

        def cap():
            mean = tf.maximum(self.rho_mean, tf.cast(self.eps, self.rho_mean.dtype))
            return tf.cast(self.rho_spike_factor, mean.dtype) * mean

        return tf.cond(self.rho_count >= self.rho_warmup, cap, lambda: inf)

    @tf.function(reduce_retracing=True)
    def _update_memory(
        self,
        s_flat_mem: tf.Tensor,
        y_flat_mem: tf.Tensor,
        idx_memory: tf.Tensor,
        s: tf.Tensor,
        y: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        dot_ys = self._dot(y, s)
        finite = tf.math.is_finite(dot_ys)
        accept = finite & (dot_ys > self.eps)

        def _update_stats():
            rho = 1.0 / (dot_ys + self.eps)
            beta = tf.cast(self.rho_ema_beta, self.rho_mean.dtype)
            rho_cast = tf.cast(rho, self.rho_mean.dtype)

            def init():
                self.rho_mean.assign(rho_cast)
                self.rho_count.assign_add(1)
                return 0

            def ema():
                self.rho_mean.assign(beta * self.rho_mean + (1.0 - beta) * rho_cast)
                self.rho_count.assign_add(1)
                return 0

            return tf.cond(self.rho_count <= 0, init, ema)

        tf.cond(
            accept,
            lambda: tf.cast(_update_stats(), tf.int32),
            lambda: tf.constant(0, tf.int32),
        )

        def update():
            def append():
                return (
                    tf.tensor_scatter_nd_update(s_flat_mem, [[idx_memory]], [s]),
                    tf.tensor_scatter_nd_update(y_flat_mem, [[idx_memory]], [y]),
                    idx_memory + 1,
                )

            def shift():
                return (
                    tf.concat([s_flat_mem[1:], [s]], axis=0),
                    tf.concat([y_flat_mem[1:], [y]], axis=0),
                    idx_memory,
                )

            return tf.cond(idx_memory < self.memory, append, shift)

        return tf.cond(accept, update, lambda: (s_flat_mem, y_flat_mem, idx_memory))

    @tf.function(reduce_retracing=True)
    def _compute_direction(
        self,
        grad: tf.Tensor,
        s_list: tf.Tensor,
        y_list: tf.Tensor,
        num_elems: tf.Tensor,
        tau: tf.Tensor,
    ) -> tf.Tensor:
        if tf.equal(num_elems, 0):
            return -grad

        rho_cap = tf.cast(self._rho_cap(), grad.dtype)

        q = grad
        alpha_list = tf.TensorArray(dtype=grad.dtype, size=num_elems, dynamic_size=False)

        for i in tf.range(num_elems - 1, -1, -1):
            s_i = s_list[i]
            y_i = y_list[i]

            rho = 1.0 / (self._dot(y_i, s_i) + self.eps)
            rho = tf.minimum(tf.cast(rho, grad.dtype), rho_cap)

            alpha_i = rho * self._dot(s_i, q)
            alpha_list = alpha_list.write(i, tf.cast(alpha_i, q.dtype))
            q = q - tf.cast(alpha_i, q.dtype) * y_i

        last_y = y_list[num_elems - 1]
        last_s = s_list[num_elems - 1]
        gamma = self._dot(last_y, last_s) / (self._dot(last_y, last_y) + self.eps)

        gamma = tf.where(tf.math.is_finite(gamma), gamma, tf.constant(1.0, gamma.dtype))
        gamma = tf.clip_by_value(gamma, self.gamma_min, self.gamma_max)
        gamma = tf.cast(gamma, q.dtype)

        r = tau * gamma * q

        for i in tf.range(num_elems):
            s_i = s_list[i]
            y_i = y_list[i]

            rho = 1.0 / (self._dot(y_i, s_i) + self.eps)
            rho = tf.minimum(tf.cast(rho, grad.dtype), rho_cap)

            beta = rho * self._dot(y_i, r)
            alpha_i = alpha_list.read(i)
            r = r + s_i * (tf.cast(alpha_i, r.dtype) - tf.cast(beta, r.dtype))

        return -r

    @tf.function(reduce_retracing=True, jit_compile=False)
    def _search_grad(
        self,
        grad_base_flat: tf.Tensor,
        mask_base: Optional[tf.Tensor],
    ) -> tf.Tensor:
        """
        Gradient representation used for:
        - the L-BFGS search direction
        - curvature pairs

        In DA this is now just the true gradient, optionally restricted to the
        active subspace.
        """
        g = grad_base_flat

        if mask_base is not None:
            g = tf.where(mask_base, g, tf.zeros_like(g))

        return g

    @tf.function(reduce_retracing=True)
    def _usable_line_search_point(self, vg: ValueAndGradient) -> tf.Tensor:
        zero = tf.zeros_like(vg.x)
        return (
            tf.math.is_finite(vg.x)
            & tf.math.is_finite(vg.f)
            & tf.math.is_finite(vg.df)
            & (vg.x > zero)
        )

    @tf.function(reduce_retracing=True)
    def _select_line_search_alpha(self, ls_result) -> Tuple[tf.Tensor, tf.Tensor]:
        dtype = ls_result.alpha.dtype
        zero = tf.constant(0.0, dtype=dtype)

        alpha_raw = tf.cast(ls_result.alpha, dtype)
        alpha_raw_valid = tf.math.is_finite(alpha_raw) & (alpha_raw >= zero)

        left_valid = self._usable_line_search_point(ls_result.left)
        right_valid = self._usable_line_search_point(ls_result.right)

        left_x = tf.cast(ls_result.left.x, dtype)
        right_x = tf.cast(ls_result.right.x, dtype)
        left_f = tf.cast(ls_result.left.f, dtype)
        right_f = tf.cast(ls_result.right.f, dtype)

        best_endpoint_alpha = tf.where(
            left_valid & right_valid,
            tf.where(left_f <= right_f, left_x, right_x),
            tf.where(left_valid, left_x, tf.where(right_valid, right_x, zero)),
        )

        use_fallback = tf.cast(ls_result.failed, tf.bool) | tf.logical_not(alpha_raw_valid)
        alpha = tf.where(use_fallback, best_endpoint_alpha, alpha_raw)
        alpha = tf.where(tf.math.is_finite(alpha), alpha, zero)

        return alpha, use_fallback

    @tf.function(reduce_retracing=True)
    def _get_grad_trial(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, list[tf.Tensor]]:
        # the point of a separate method here is to avoid writing the wrong costs to the
        # display when doing line search evaluations, which can be confusing when the line
        # search evaluates points with much higher cost than the current iterate
        theta = self.map.get_theta()

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            for t in theta:
                tape.watch(t)

            U, V = self.map.get_UV(inputs)
            total, _, _ = self.cost_fn(U, V, self.map.inputs)

        grad_theta = tape.gradient(total, theta)
        grad_theta = [tf.zeros_like(t) if g is None else g for g, t in zip(grad_theta, theta)]
        return total, grad_theta

    @tf.function(reduce_retracing=True, jit_compile=False)
    def _line_search_result(
        self,
        theta_flat: tf.Tensor,
        p_flat: tf.Tensor,
        input: tf.Tensor,
    ) -> LineSearchResult:
        L, U = self.map.get_box_bounds_flat()
        amax = self._alpha_max(theta_flat, p_flat, L, U)

        def eval_fn(alpha: tf.Tensor) -> ValueAndGradient:
            alpha_eff = tf.minimum(alpha, amax)

            theta_backup = self.map.copy_theta(self.map.get_theta())
            theta_alpha, _ = self._apply_step(theta_flat, alpha_eff, p_flat)

            self.map.set_theta(self.map.unflatten_theta(theta_alpha))

            f, grad_theta = self._get_grad_trial(input)
            grad_flat = self.map.flatten_theta(grad_theta)

            mask = self._get_mask(theta_alpha, grad_flat, L, U)
            p_masked = tf.where(mask, p_flat, tf.zeros_like(p_flat))
            df = self._dot(grad_flat, p_masked)

            self.map.set_theta(theta_backup)
            return ValueAndGradient(x=alpha_eff, f=f, df=tf.cast(df, grad_flat.dtype))

        if self._line_search_supports_result:
            return self.line_search.search_result(theta_flat, p_flat, eval_fn)

        alpha = self.line_search.search(theta_flat, p_flat, eval_fn)
        alpha_valid = tf.math.is_finite(alpha) & (alpha >= tf.zeros_like(alpha))
        alpha_safe = tf.where(alpha_valid, alpha, tf.zeros_like(alpha))
        vg = eval_fn(alpha_safe)

        return LineSearchResult(
            alpha=tf.cast(vg.x, theta_flat.dtype),
            converged=tf.cast(alpha_valid, tf.bool),
            failed=tf.logical_not(tf.cast(alpha_valid, tf.bool)),
            func_evals=tf.constant(-1, dtype=tf.int32),
            iterations=tf.constant(-1, dtype=tf.int32),
            left=vg,
            right=vg,
        )

    @tf.function(reduce_retracing=True, jit_compile=False)
    def _line_search_with_fallback(
        self,
        theta_flat: tf.Tensor,
        p_flat: tf.Tensor,
        input: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        ls_result = self._line_search_result(theta_flat, p_flat, input)
        alpha, used_fallback = self._select_line_search_alpha(ls_result)
        return tf.cast(alpha, theta_flat.dtype), used_fallback

    @tf.function(reduce_retracing=True, jit_compile=False)
    def _line_search(self, theta_flat: tf.Tensor, p_flat: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        alpha, _ = self._line_search_with_fallback(theta_flat, p_flat, input)
        return alpha

    def minimize_impl(self, inputs: tf.Tensor) -> tf.Tensor:
        first_batch = self.sampler(inputs)  # [M, B, H, W, C]
        n_batches = first_batch.shape[0]
        if n_batches != 1:
            raise NotImplementedError("❌ L-BFGS requires a single batch.")

        if getattr(self.sampler, "dynamic_augmentation", False):
            static_batches = None
            dynamic_augmentation = True
        else:
            static_batches = first_batch
            dynamic_augmentation = False

        input = first_batch[0, :, :, :, :]

        # State variables
        theta_flat = self.map.flatten_theta(self.map.get_theta())
        cost, grad_u, grad_theta = self._get_grad(input)
        grad_theta_flat = self.map.flatten_theta(grad_theta)
        U, V = self.map.get_UV(input)
        self._init_step_state(U, V, theta_flat)

        # Memory variables
        w_dim = tf.shape(theta_flat)[0]
        idx_memory = tf.constant(0, dtype=tf.int32)
        s_flat_mem = tf.zeros([self.memory, w_dim], dtype=theta_flat.dtype)
        y_flat_mem = tf.zeros([self.memory, w_dim], dtype=theta_flat.dtype)

        # Accessory variables
        halt_status = tf.constant(HaltStatus.CONTINUE.value, dtype=tf.int32)
        iter_last = tf.constant(-1, dtype=tf.int32)
        costs = tf.TensorArray(dtype=cost.dtype, size=0, dynamic_size=True)
        cost_total_hist = tf.TensorArray(dtype=cost.dtype, size=0, dynamic_size=True)
        cost_data_hist = tf.TensorArray(dtype=cost.dtype, size=0, dynamic_size=True)
        cost_reg_hist = tf.TensorArray(dtype=cost.dtype, size=0, dynamic_size=True)

        for iter in tf.range(self.iter_max):
            # Sample fresh augmented batch for this iteration
            if dynamic_augmentation:
                next_batch = self.sampler(inputs)
            else:
                next_batch = static_batches

            input = next_batch[0, :, :, :, :]

            theta_prev = theta_flat

            # Tempering
            tau = self._compute_tau(iter)

            # choose base point / gradient for the step (bounded subclass overrides this)
            theta_base, grad_base_flat, mask_base = self._step_base_point(
                theta_flat, grad_theta_flat, input
            )

            # Gradient used by the inverse-Hessian model.
            grad_search_prev = self._search_grad(grad_base_flat, mask_base)

            # Restrict memory to the active subspace if needed.
            s_list = s_flat_mem[:idx_memory]
            y_list = y_flat_mem[:idx_memory]
            s_list, y_list = self._mask_memory_for_subspace(s_list, y_list, mask_base)

            # Search direction is built from the quasi-Newton model gradient.
            p_flat = self._compute_direction(
                grad_search_prev,
                s_list,
                y_list,
                idx_memory,
                tau,
            )

            # Force descent uses TRUE base gradient
            p_flat, mask = self._force_descent(p_flat, grad_base_flat, theta_base)

            # Line search uses TRUE gradients internally.
            alpha, ls_used_fallback = self._line_search_with_fallback(theta_base, p_flat, input)
            alpha = tf.cond(
                ls_used_fallback,
                lambda: tf.maximum(alpha, tf.zeros_like(alpha)),
                lambda: tf.maximum(alpha, tf.cast(self.alpha_min, alpha.dtype)),
            )
            alpha = self._clip_alpha(alpha, theta_base, p_flat)

            theta_flat, theta_trial = self._apply_step(theta_base, alpha, p_flat)

            # New weights, cost, and TRUE grads
            self.map.set_theta(self.map.unflatten_theta(theta_flat))
            cost, grad_u, grad_theta = self._get_grad(input)
            grad_theta_flat = self.map.flatten_theta(grad_theta)

            # Curvature pair is built from the same search-gradient representation.
            _, grad_base_new, mask_base_new = self._step_base_point(
                theta_flat, grad_theta_flat, input
            )
            grad_search_new = self._search_grad(grad_base_new, mask_base_new)

            s = theta_flat - theta_prev
            y = grad_search_new - grad_search_prev

            s, y = self._constrain_pair(
                s, y, theta_prev, theta_trial, mask, theta_flat, grad_theta_flat
            )

            # Update memory (DA override handles rho spike clamping etc.)
            s_flat_mem, y_flat_mem, idx_memory = self._update_memory(
                s_flat_mem, y_flat_mem, idx_memory, s, y
            )

            costs = costs.write(iter, cost)
            cost_total_hist = cost_total_hist.write(iter, self.last_total.read_value())
            cost_data_hist = cost_data_hist.write(iter, self.last_data.read_value())
            cost_reg_hist = cost_reg_hist.write(iter, self.last_reg.read_value())

            iter_py = int(iter.numpy())
            accepted_iter = iter_py + 1

            if self.map._da_out_freq > 0 and accepted_iter % self.map._da_out_freq == 0:
                self._publish_cost_history(
                    cost_total_hist,
                    cost_data_hist,
                    cost_reg_hist,
                    accepted_iter,
                )

            self.map.maybe_run_step_callback(iter_py)

            U, V = self.map.get_UV(input)
            grad_u_norm, grad_theta_norm = self._get_grad_norm(grad_u, grad_theta)
            self._update_step_state(
                iter, U, V, theta_flat, cost, grad_u_norm, grad_theta_norm
            )
            halt_status = self._check_stopping()
            self._update_display()

            if self.debug_mode and iter % self.debug_freq == 0:
                self._update_debug_state(iter, cost, grad_u, grad_theta)
                self._debug_display()

            iter_last = iter

            if tf.not_equal(halt_status, HaltStatus.CONTINUE.value):
                break

        self._finalize_display(halt_status)

        n_keep = max(0, int(iter_last.numpy()) + 1)
        self._publish_cost_history(
            cost_total_hist,
            cost_data_hist,
            cost_reg_hist,
            n_keep,
        )

        return costs.stack()[:n_keep]

    def _update_display(self) -> None:
        if not getattr(self.display, "enabled", False):
            return

        if not bool(self.display.should_update(self.step_state.iter).numpy()):
            return

        if self.halt_state.criterion_values and self.halt_state.criterion_satisfied:
            values = [float(v.numpy()) for v in self.halt_state.criterion_values]
            satisfied = [bool(v.numpy()) for v in self.halt_state.criterion_satisfied]
        else:
            values = None
            satisfied = None

        self.display.update(
            int(self.step_state.iter.numpy()),
            float(self.last_total.read_value().numpy()),
            float(self.last_data.read_value().numpy()),
            float(self.last_reg.read_value().numpy()),
            values,
            satisfied,
        )

    @tf.function(reduce_retracing=True)
    def _get_grad(
        self, inputs: tf.Tensor
    ) -> Tuple[tf.Tensor, list[tf.Tensor], list[tf.Tensor]]:
        theta = self.map.get_theta()

        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            for t in theta:
                tape.watch(t)

            U, V = self.map.get_UV(inputs)
            total, data, reg = self.cost_fn(U, V, self.map.inputs)

        grad_u = tape.gradient(total, [U, V])
        grad_theta = tape.gradient(total, theta)
        del tape

        grad_theta = [tf.zeros_like(t) if g is None else g for g, t in zip(grad_theta, theta)]

        self.last_total.assign(tf.stop_gradient(total))
        self.last_data.assign(tf.stop_gradient(data))
        self.last_reg.assign(tf.stop_gradient(reg))

        return total, grad_u, grad_theta
