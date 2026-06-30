#!/usr/bin/env python3
# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from __future__ import annotations

import collections

import tensorflow as tf
from typing import Tuple, Optional, Sequence
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
    Bounded L-BFGS for data assimilation.

    Curvature pairs are accepted/used based on a relative (cosine-like) test
    `y·s > curv_eps_rel * |y| * |s|` instead of an absolute threshold (which
    rejects all pairs once steps get small) or an EMA-based rho cap (which is
    contaminated by spikes and systematically clamps legitimate pairs late in
    the optimization when y·s shrinks).
    """

    def __init__(
        self,
        *args,
        curv_eps_rel: float = 1e-8,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        dtype = getattr(self.map, "precision", tf.float32)

        self.last_total = tf.Variable(0.0, trainable=False, dtype=dtype)
        self.last_data = tf.Variable(0.0, trainable=False, dtype=dtype)
        self.last_reg = tf.Variable(0.0, trainable=False, dtype=dtype)

        # Relative curvature threshold for (s, y) pair acceptance and use.
        self.curv_eps_rel = tf.constant(curv_eps_rel, dtype=dtype)

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
        empty = tf.zeros([0], dtype=self.last_total.dtype)
        self.accepted_cost_total_hist = empty
        self.accepted_cost_data_hist = empty
        self.accepted_cost_reg_hist = empty
        return super().minimize(inputs)

    @tf.function(reduce_retracing=True)
    def _compute_tau(self, iter: tf.Tensor) -> tf.Tensor:
        # No early-iteration damping of the initial Hessian scaling for
        # full-batch deterministic DA: it only shrinks the trial step and makes
        # the line search expand it back out at extra cost.
        del iter
        return tf.constant(1.0, dtype=self.precision)

    @staticmethod
    def _stack_history(values: Sequence[tf.Tensor], dtype: tf.DType) -> tf.Tensor:
        if not values:
            return tf.zeros([0], dtype=dtype)
        return tf.stack([tf.cast(v, dtype) for v in values], axis=0)

    def _publish_cost_history(
        self,
        cost_total_hist: Sequence[tf.Tensor],
        cost_data_hist: Sequence[tf.Tensor],
        cost_reg_hist: Sequence[tf.Tensor],
        n_keep: int,
    ) -> None:
        dtype = self.last_total.dtype
        self.accepted_cost_total_hist = self._stack_history(cost_total_hist[:n_keep], dtype)
        self.accepted_cost_data_hist = self._stack_history(cost_data_hist[:n_keep], dtype)
        self.accepted_cost_reg_hist = self._stack_history(cost_reg_hist[:n_keep], dtype)

    @tf.function(reduce_retracing=True)
    def _dot(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        acc = tf.tensordot(tf.cast(a, tf.float64), tf.cast(b, tf.float64), axes=1)
        return tf.cast(acc, self.precision)

    @tf.function(reduce_retracing=True)
    def _ordered_memory(
        self,
        mem: tf.Tensor,
        next_memory: tf.Tensor,
        num_memory: tf.Tensor,
    ) -> tf.Tensor:
        """Return L-BFGS memory in oldest-to-newest order."""
        return tf.cond(
            num_memory < self.memory,
            lambda: mem[:num_memory],
            lambda: tf.concat([mem[next_memory:], mem[:next_memory]], axis=0),
        )

    @tf.function(reduce_retracing=True)
    def _update_memory(
        self,
        s_flat_mem: tf.Tensor,
        y_flat_mem: tf.Tensor,
        next_memory: tf.Tensor,
        num_memory: tf.Tensor,
        s: tf.Tensor,
        y: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        dot_ys = self._dot(y, s)
        s_norm = tf.sqrt(self._dot(s, s))
        y_norm = tf.sqrt(self._dot(y, y))

        # Relative (cosine-like) curvature test. An absolute threshold rejects
        # all pairs once y·s becomes legitimately small (fine grids, small
        # steps near convergence) and degrades the optimizer to plain gradient
        # descent; this test instead rejects only near-orthogonal pairs that
        # would produce an untrustworthy, exploding rho.
        threshold = tf.cast(self.curv_eps_rel, dot_ys.dtype) * s_norm * y_norm
        accept = tf.math.is_finite(dot_ys) & (dot_ys > threshold)

        def update():
            slot = next_memory
            s_new = tf.tensor_scatter_nd_update(s_flat_mem, [[slot]], [s])
            y_new = tf.tensor_scatter_nd_update(y_flat_mem, [[slot]], [y])
            next_new = tf.math.floormod(slot + 1, tf.cast(self.memory, slot.dtype))
            num_new = tf.minimum(num_memory + 1, tf.cast(self.memory, num_memory.dtype))
            return s_new, y_new, next_new, num_new

        return tf.cond(
            accept,
            update,
            lambda: (s_flat_mem, y_flat_mem, next_memory, num_memory),
        )

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

        one = tf.constant(1.0, grad.dtype)
        zero = tf.constant(0.0, grad.dtype)
        eps_rel = tf.cast(self.curv_eps_rel, grad.dtype)

        q = grad
        alpha_list = tf.TensorArray(dtype=grad.dtype, size=num_elems, dynamic_size=False)
        rho_list = tf.TensorArray(dtype=grad.dtype, size=num_elems, dynamic_size=False)

        for i in tf.range(num_elems - 1, -1, -1):
            s_i = s_list[i]
            y_i = y_list[i]

            # Subspace masking can degrade stored pairs: a pair that passed the
            # curvature test when stored may have a tiny or negative y·s after
            # masking. Skip such pairs (rho = 0 removes their contribution from
            # both recursion loops) instead of clamping rho, which would break
            # the secant relation for otherwise-healthy pairs.
            dot_ys = tf.cast(self._dot(y_i, s_i), grad.dtype)
            norm_prod = tf.sqrt(tf.cast(self._dot(s_i, s_i), grad.dtype)) * tf.sqrt(
                tf.cast(self._dot(y_i, y_i), grad.dtype)
            )
            valid = tf.math.is_finite(dot_ys) & (dot_ys > eps_rel * norm_prod)
            rho = tf.where(valid, tf.math.divide_no_nan(one, dot_ys), zero)

            alpha_i = rho * tf.cast(self._dot(s_i, q), grad.dtype)
            alpha_list = alpha_list.write(i, alpha_i)
            rho_list = rho_list.write(i, rho)
            q = q - alpha_i * y_i

        last_y = y_list[num_elems - 1]
        last_s = s_list[num_elems - 1]
        dot_ys_last = tf.cast(self._dot(last_y, last_s), grad.dtype)
        dot_yy_last = tf.cast(self._dot(last_y, last_y), grad.dtype)

        gamma = tf.math.divide_no_nan(dot_ys_last, dot_yy_last)
        gamma_ok = tf.math.is_finite(gamma) & (gamma > zero)
        gamma = tf.where(gamma_ok, gamma, one)
        gamma = tf.clip_by_value(
            gamma, tf.cast(self.gamma_min, grad.dtype), tf.cast(self.gamma_max, grad.dtype)
        )

        r = tau * gamma * q

        for i in tf.range(num_elems):
            s_i = s_list[i]
            y_i = y_list[i]

            rho = rho_list.read(i)
            beta = rho * tf.cast(self._dot(y_i, r), grad.dtype)
            alpha_i = alpha_list.read(i)
            r = r + s_i * (alpha_i - beta)

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
    def _select_line_search_alpha(
        self, ls_result, f0: Optional[tf.Tensor] = None
    ) -> Tuple[tf.Tensor, tf.Tensor]:
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

        # A fallback endpoint must not increase the cost relative to alpha = 0;
        # without this guard a failed search happily accepts an uphill step
        # (the left endpoint at x = 0 is never "usable", so the right endpoint
        # used to win by default whatever its value).
        if f0 is not None:
            f0_cast = tf.cast(f0, dtype)
            left_valid = left_valid & (left_f <= f0_cast)
            right_valid = right_valid & (right_f <= f0_cast)

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
        f0: Optional[tf.Tensor] = None,
        grad0_flat: Optional[tf.Tensor] = None,
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

        # Value/slope at alpha = 0 are already known to the caller (cost and
        # gradient at the current iterate); hand them to the line search to
        # avoid one full re-evaluation per line search.
        if f0 is not None and grad0_flat is not None:
            dtype = theta_flat.dtype
            mask0 = self._get_mask(theta_flat, grad0_flat, L, U)
            p_masked0 = tf.where(mask0, p_flat, tf.zeros_like(p_flat))
            df0 = self._dot(grad0_flat, p_masked0)
            val_0 = ValueAndGradient(
                x=tf.constant(0.0, dtype=dtype),
                f=tf.cast(f0, dtype),
                df=tf.cast(df0, dtype),
            )
        else:
            val_0 = None

        if self._line_search_supports_result:
            return self.line_search.search_result(theta_flat, p_flat, eval_fn, val_0=val_0)

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
        f0: Optional[tf.Tensor] = None,
        grad0_flat: Optional[tf.Tensor] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        ls_result = self._line_search_result(theta_flat, p_flat, input, f0, grad0_flat)
        alpha, used_fallback = self._select_line_search_alpha(ls_result, f0)
        return tf.cast(alpha, theta_flat.dtype), used_fallback

    @tf.function(reduce_retracing=True, jit_compile=False)
    def _line_search(self, theta_flat: tf.Tensor, p_flat: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        alpha, _ = self._line_search_with_fallback(theta_flat, p_flat, input)
        return alpha

    def minimize_impl(self, inputs: tf.Tensor) -> tf.Tensor:
        # inputs: [N, H, W, C] — used as a single full batch (LBFGS is full-batch)
        input = inputs

        # Project the initial iterate into the box (the bounded parent's
        # minimize_impl is bypassed by this override, so do it explicitly).
        lb_flat, ub_flat = self.map.get_box_bounds_flat()
        theta_flat = self.map.flatten_theta(self.map.get_theta())
        theta_flat = self._project(theta_flat, lb_flat, ub_flat)
        self.map.set_theta(self.map.unflatten_theta(theta_flat))

        # State variables
        cost, grad_u, grad_theta = self._get_grad(input)
        grad_theta_flat = self.map.flatten_theta(grad_theta)
        U, V = self.map.get_UV(input)
        self._init_step_state(U, V, theta_flat)

        # Memory variables. Use a ring buffer to avoid shifting O(memory * n_params)
        # vectors on every accepted step.
        w_dim = tf.shape(theta_flat)[0]
        next_memory = tf.constant(0, dtype=tf.int32)
        num_memory = tf.constant(0, dtype=tf.int32)
        s_flat_mem = tf.zeros([self.memory, w_dim], dtype=theta_flat.dtype)
        y_flat_mem = tf.zeros([self.memory, w_dim], dtype=theta_flat.dtype)

        halt_status = tf.constant(HaltStatus.CONTINUE.value, dtype=tf.int32)
        iter_last = -1
        costs_hist: list[tf.Tensor] = []
        cost_total_hist: list[tf.Tensor] = []
        cost_data_hist: list[tf.Tensor] = []
        cost_reg_hist: list[tf.Tensor] = []

        iter_max = int(self.iter_max.numpy())
        debug_freq = int(tf.convert_to_tensor(self.debug_freq).numpy()) if self.debug_mode else 1
        for iter_py in range(iter_max):
            iter = tf.constant(iter_py, dtype=tf.int32)
            input = inputs

            theta_prev = theta_flat
            grad_prev_flat = grad_theta_flat

            # Tempering (identity for DA, see _compute_tau)
            tau = self._compute_tau(iter)

            # Predict the free set via the Cauchy probe; the step itself starts
            # from the current iterate (theta_base == theta_flat).
            theta_base, grad_base_flat, mask_base = self._step_base_point(
                theta_flat, grad_theta_flat, input
            )

            # Gradient used by the inverse-Hessian model.
            grad_search_prev = self._search_grad(grad_base_flat, mask_base)

            # Restrict memory to the active subspace if needed.
            s_list = self._ordered_memory(s_flat_mem, next_memory, num_memory)
            y_list = self._ordered_memory(y_flat_mem, next_memory, num_memory)
            s_list, y_list = self._mask_memory_for_subspace(s_list, y_list, mask_base)

            # Search direction is built from the quasi-Newton model gradient.
            p_flat = self._compute_direction(
                grad_search_prev,
                s_list,
                y_list,
                num_memory,
                tau,
            )

            # Force descent uses TRUE base gradient.
            p_flat, mask = self._force_descent(p_flat, grad_base_flat, theta_base)

            # Line search uses TRUE gradients internally; cost/gradient at
            # alpha = 0 are the ones at the current iterate, so pass them in.
            alpha, ls_used_fallback = self._line_search_with_fallback(
                theta_base, p_flat, input, cost, grad_theta_flat
            )
            alpha = tf.cond(
                ls_used_fallback,
                lambda: tf.maximum(alpha, tf.zeros_like(alpha)),
                lambda: tf.maximum(alpha, tf.cast(self.alpha_min, alpha.dtype)),
            )
            alpha = self._clip_alpha(alpha, theta_base, p_flat)

            theta_flat, theta_trial = self._apply_step(theta_base, alpha, p_flat)

            # New weights, cost, and TRUE grads.
            self.map.set_theta(self.map.unflatten_theta(theta_flat))
            cost, grad_u, grad_theta = self._get_grad(input)
            grad_theta_flat = self.map.flatten_theta(grad_theta)

            # Secant-consistent curvature pair: differences of the actual
            # iterates and of the true gradients evaluated at those same two
            # points. _constrain_pair filters components that crossed or sit on
            # a bound.
            s = theta_flat - theta_prev
            y = grad_theta_flat - grad_prev_flat

            s, y = self._constrain_pair(
                s, y, theta_prev, theta_trial, mask, theta_flat, grad_theta_flat
            )

            # Update memory (relative curvature test decides acceptance).
            s_flat_mem, y_flat_mem, next_memory, num_memory = self._update_memory(
                s_flat_mem, y_flat_mem, next_memory, num_memory, s, y
            )

            costs_hist.append(tf.identity(cost))
            cost_total_hist.append(tf.identity(self.last_total.read_value()))
            cost_data_hist.append(tf.identity(self.last_data.read_value()))
            cost_reg_hist.append(tf.identity(self.last_reg.read_value()))

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

            if self.debug_mode and debug_freq > 0 and iter_py % debug_freq == 0:
                self._update_debug_state(iter, cost, grad_u, grad_theta)
                self._debug_display()

            iter_last = iter_py
            if bool(tf.not_equal(halt_status, HaltStatus.CONTINUE.value).numpy()):
                break

        self._finalize_display(halt_status)

        n_keep = max(0, iter_last + 1)
        self._publish_cost_history(
            cost_total_hist,
            cost_data_hist,
            cost_reg_hist,
            n_keep,
        )

        return self._stack_history(costs_hist[:n_keep], cost.dtype)

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
