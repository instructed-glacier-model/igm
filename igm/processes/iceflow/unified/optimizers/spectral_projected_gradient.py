#!/usr/bin/env python3
# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Tuple

import tensorflow as tf

from .optimizer import Optimizer
from ..halt import Halt, HaltStatus
from ..mappings import Mapping
from igm.utils.math.norms import compute_norm

try:  # Optional: available in the DA optimizer package, but not required generally.
    from .da_progress_optimizer import _DAProgressOptimizer
except Exception:  # pragma: no cover - keeps the optimizer usable outside DA builds.
    _DAProgressOptimizer = None


class OptimizerSpectralProjectedGradient(Optimizer):
    """
    Spectral projected gradient optimizer for box-constrained inverse problems.

    The method is deliberately simpler than bounded L-BFGS:
      - one full gradient evaluation per accepted step;
      - trial points in the line search are value-only evaluations;
      - feasibility is enforced by projection onto flat theta-space bounds;
      - the step length is a scalar Barzilai--Borwein spectral estimate;
      - a nonmonotone Armijo test guards against unstable spectral steps.

    The mapping must provide get_box_bounds_flat(), flatten_theta(),
    unflatten_theta(), get_theta(), and set_theta().  This matches the current
    data-assimilation mapping API, where physical bounds are converted once into
    theta-space bounds by the mapping.

    This class also exposes last_total / last_data / last_reg and accepted cost
    histories so it can be used as a drop-in DA optimizer in the current phase
    runner.  If the supplied cost_fn returns a scalar, data and regularization
    histories are filled with zeros.  If it returns (total, data, reg), all three
    are recorded.
    """

    def __init__(
        self,
        cost_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor | tuple[tf.Tensor, ...]],
        map: Mapping,
        halt: Optional[Halt] = None,
        print_cost: bool = True,
        print_cost_freq: int = 1,
        precision: str = "float32",
        ord_grad_u: str = "l2_weighted",
        ord_grad_theta: str = "l2_weighted",
        iter_max: int = int(1e5),
        alpha0: float = 1.0,
        alpha_min: float = 1e-12,
        alpha_max: float = 1e12,
        armijo_c: float = 1e-4,
        backtrack_factor: float = 0.5,
        max_backtracks: int = 20,
        nonmonotone_window: int = 10,
        bb_variant: str = "alternating",
        step_tol: float = 0.0,
        use_da_display: bool = True,
        debug_mode: bool = False,
        debug_freq: int = 100,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            cost_fn=cost_fn,
            map=map,
            halt=halt,
            print_cost=print_cost,
            print_cost_freq=print_cost_freq,
            precision=precision,
            ord_grad_u=ord_grad_u,
            ord_grad_theta=ord_grad_theta,
            debug_mode=debug_mode,
            debug_freq=debug_freq,
        )
        self.name = "spectral_projected_gradient"
        # Keep construction tolerant of existing LBFGS/Adam interface dictionaries.
        # Unknown options are deliberately not forwarded to the base class.
        self.ignored_options = tuple(sorted(kwargs.keys()))

        if not hasattr(self.map, "get_box_bounds_flat"):
            raise ValueError(
                "❌ Mapping must provide get_box_bounds_flat() for "
                "spectral projected gradient optimization."
            )

        if not (0.0 < backtrack_factor < 1.0):
            raise ValueError("backtrack_factor must be in the open interval (0, 1).")
        if armijo_c <= 0.0 or armijo_c >= 1.0:
            raise ValueError("armijo_c should be in the open interval (0, 1).")
        if alpha_min <= 0.0 or alpha_max < alpha_min:
            raise ValueError("Require 0 < alpha_min <= alpha_max.")
        if bb_variant not in {"bb1", "bb2", "alternating"}:
            raise ValueError("bb_variant must be one of: 'bb1', 'bb2', 'alternating'.")

        self.iter_max = tf.Variable(int(iter_max), dtype=tf.int32, trainable=False)
        self.alpha0 = tf.Variable(float(alpha0), dtype=self.precision, trainable=False)
        self.alpha_min = tf.Variable(float(alpha_min), dtype=self.precision, trainable=False)
        self.alpha_max = tf.Variable(float(alpha_max), dtype=self.precision, trainable=False)

        self.armijo_c = tf.constant(float(armijo_c), dtype=self.precision)
        self.backtrack_factor = tf.constant(float(backtrack_factor), dtype=self.precision)
        self.max_backtracks = int(max_backtracks)
        self.nonmonotone_window = max(1, int(nonmonotone_window))
        self.bb_variant = bb_variant
        self.step_tol = tf.constant(float(step_tol), dtype=self.precision)

        self.eps = tf.constant(1e-12 if self.precision == tf.float32 else 1e-20, self.precision)

        self.last_total = tf.Variable(0.0, trainable=False, dtype=self.precision)
        self.last_data = tf.Variable(0.0, trainable=False, dtype=self.precision)
        self.last_reg = tf.Variable(0.0, trainable=False, dtype=self.precision)
        self.accepted_cost_total_hist = tf.zeros([0], dtype=self.precision)
        self.accepted_cost_data_hist = tf.zeros([0], dtype=self.precision)
        self.accepted_cost_reg_hist = tf.zeros([0], dtype=self.precision)

        self._uses_da_display = False
        if use_da_display and _DAProgressOptimizer is not None:
            self.display = _DAProgressOptimizer(
                enabled=self.display.enabled,
                freq=self.display.freq,
            )
            self._uses_da_display = True

    def update_parameters(
        self,
        iter_max: Optional[int] = None,
        alpha0: Optional[float] = None,
        alpha_min: Optional[float] = None,
        alpha_max: Optional[float] = None,
    ) -> None:
        """Update mutable run parameters without rebuilding the optimizer."""
        if iter_max is not None:
            self.iter_max.assign(int(iter_max))
        if alpha0 is not None:
            self.alpha0.assign(float(alpha0))
        if alpha_min is not None:
            self.alpha_min.assign(float(alpha_min))
        if alpha_max is not None:
            self.alpha_max.assign(float(alpha_max))

    def minimize(self, inputs: tf.Tensor) -> tf.Tensor:
        empty = tf.zeros([0], dtype=self.precision)
        self.accepted_cost_total_hist = empty
        self.accepted_cost_data_hist = empty
        self.accepted_cost_reg_hist = empty
        self.last_total.assign(tf.cast(0.0, self.precision))
        self.last_data.assign(tf.cast(0.0, self.precision))
        self.last_reg.assign(tf.cast(0.0, self.precision))
        return super().minimize(inputs)

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
        self.accepted_cost_total_hist = self._stack_history(cost_total_hist[:n_keep], self.precision)
        self.accepted_cost_data_hist = self._stack_history(cost_data_hist[:n_keep], self.precision)
        self.accepted_cost_reg_hist = self._stack_history(cost_reg_hist[:n_keep], self.precision)

    def _bounds(self) -> Tuple[tf.Tensor, tf.Tensor]:
        lower, upper = self.map.get_box_bounds_flat()
        return tf.cast(lower, self.precision), tf.cast(upper, self.precision)

    @tf.function(reduce_retracing=True)
    def _project(self, theta_flat: tf.Tensor, lower: tf.Tensor, upper: tf.Tensor) -> tf.Tensor:
        return tf.clip_by_value(theta_flat, lower, upper)

    @tf.function(reduce_retracing=True)
    def _dot(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        # Float64 accumulation is cheap relative to a forward model call and improves
        # BB and Armijo decisions for large theta vectors.
        val = tf.tensordot(tf.cast(a, tf.float64), tf.cast(b, tf.float64), axes=1)
        return tf.cast(val, self.precision)

    @tf.function(reduce_retracing=True)
    def _projected_gradient(
        self,
        theta_flat: tf.Tensor,
        grad_flat: tf.Tensor,
        lower: tf.Tensor,
        upper: tf.Tensor,
    ) -> tf.Tensor:
        # Stationarity mapping: zero iff theta is first-order stationary for the
        # box-constrained problem under a unit projected-gradient step.
        return theta_flat - self._project(theta_flat - grad_flat, lower, upper)

    def _split_cost_output(self, cost_output: Any) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        if isinstance(cost_output, (tuple, list)):
            if len(cost_output) >= 3:
                total, data, reg = cost_output[:3]
            elif len(cost_output) == 2:
                total, data = cost_output
                reg = tf.zeros_like(tf.cast(total, self.precision))
            elif len(cost_output) == 1:
                total = cost_output[0]
                data = tf.zeros_like(tf.cast(total, self.precision))
                reg = tf.zeros_like(tf.cast(total, self.precision))
            else:
                raise ValueError("cost_fn returned an empty tuple/list.")
        else:
            total = cost_output
            data = tf.zeros_like(tf.cast(total, self.precision))
            reg = tf.zeros_like(tf.cast(total, self.precision))

        total = tf.cast(total, self.precision)
        data = tf.cast(data, self.precision)
        reg = tf.cast(reg, self.precision)
        return total, data, reg

    def _cost_inputs(self, raw_inputs: tf.Tensor) -> tf.Tensor:
        # MappingDataAssimilation patches theta-dependent input channels inside
        # get_UV() and stores the synchronized tensor as map.inputs.  Plain
        # mappings either expose the same field or fall back to raw_inputs.
        return getattr(self.map, "inputs", raw_inputs)

    @tf.function(reduce_retracing=True, jit_compile=False)
    def _evaluate_value(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        U, V = self.map.get_UV(inputs)
        total, data, reg = self._split_cost_output(self.cost_fn(U, V, self._cost_inputs(inputs)))
        return total, data, reg

    @tf.function(reduce_retracing=True, jit_compile=False)
    def _evaluate_grad(
        self,
        inputs: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tuple[tf.Tensor, tf.Tensor], list[tf.Tensor], tf.Tensor, tf.Tensor]:
        theta = self.map.get_theta()

        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            for theta_i in theta:
                tape.watch(theta_i)
            U, V = self.map.get_UV(inputs)
            total, data, reg = self._split_cost_output(
                self.cost_fn(U, V, self._cost_inputs(inputs))
            )

        grad_u = tuple(tape.gradient(total, [U, V]))
        grad_theta = tape.gradient(total, theta)
        del tape

        grad_theta = [
            tf.zeros_like(theta_i) if grad_i is None else grad_i
            for grad_i, theta_i in zip(grad_theta, theta)
        ]

        self.last_total.assign(tf.stop_gradient(total))
        self.last_data.assign(tf.stop_gradient(data))
        self.last_reg.assign(tf.stop_gradient(reg))

        return total, data, reg, grad_u, grad_theta, U, V

    @tf.function(reduce_retracing=True, jit_compile=False)
    def _get_grad(
        self,
        inputs: tf.Tensor,
    ) -> Tuple[tf.Tensor, tuple[tf.Tensor, tf.Tensor], list[tf.Tensor]]:
        total, _, _, grad_u, grad_theta, _, _ = self._evaluate_grad(inputs)
        return total, grad_u, grad_theta

    def _get_grad_norm(
        self,
        grad_u: list[tf.Tensor],
        grad_theta: list[tf.Tensor],
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        grad_u_norm, _ = super()._get_grad_norm(grad_u, grad_theta)
        theta_flat = self.map.flatten_theta(self.map.get_theta())
        grad_flat = self.map.flatten_theta(grad_theta)
        lower, upper = self._bounds()
        pg = self._projected_gradient(theta_flat, grad_flat, lower, upper)
        pg_norm = compute_norm(pg, ord=self.ord_grad_theta)
        return grad_u_norm, pg_norm

    def _reference_cost(self, recent_total_costs: Sequence[tf.Tensor], current_cost: tf.Tensor) -> tf.Tensor:
        if not recent_total_costs:
            return tf.identity(current_cost)
        window = recent_total_costs[-self.nonmonotone_window :]
        return tf.reduce_max(self._stack_history(window, self.precision))

    def _spectral_alpha(
        self,
        iteration: int,
        theta_flat: tf.Tensor,
        theta_prev_flat: Optional[tf.Tensor],
        grad_flat: tf.Tensor,
        grad_prev_flat: Optional[tf.Tensor],
    ) -> tf.Tensor:
        if theta_prev_flat is None or grad_prev_flat is None:
            return tf.clip_by_value(self.alpha0.read_value(), self.alpha_min, self.alpha_max)

        s = theta_flat - theta_prev_flat
        y = grad_flat - grad_prev_flat
        sy = self._dot(s, y)
        ss = self._dot(s, s)
        yy = self._dot(y, y)

        finite = tf.math.is_finite(sy) & tf.math.is_finite(ss) & tf.math.is_finite(yy)
        positive_curvature = sy > self.eps

        use_bb1 = self.bb_variant == "bb1" or (
            self.bb_variant == "alternating" and iteration % 2 == 0
        )

        if use_bb1:
            alpha = ss / (sy + self.eps)
        else:
            alpha = sy / (yy + self.eps)

        alpha = tf.where(finite & positive_curvature, alpha, self.alpha0.read_value())
        alpha = tf.where(tf.math.is_finite(alpha), alpha, self.alpha0.read_value())
        return tf.clip_by_value(tf.cast(alpha, self.precision), self.alpha_min, self.alpha_max)

    def _projected_armijo_search(
        self,
        theta_flat: tf.Tensor,
        grad_flat: tf.Tensor,
        current_total: tf.Tensor,
        current_data: tf.Tensor,
        current_reg: tf.Tensor,
        reference_total: tf.Tensor,
        alpha_start: tf.Tensor,
        inputs: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, bool]:
        lower, upper = self._bounds()
        theta_backup = self.map.copy_theta(self.map.get_theta())

        alpha = tf.clip_by_value(tf.cast(alpha_start, self.precision), self.alpha_min, self.alpha_max)
        best_theta = tf.identity(theta_flat)
        best_total = tf.identity(current_total)
        best_data = tf.identity(current_data)
        best_reg = tf.identity(current_reg)
        best_alpha = tf.zeros((), dtype=self.precision)
        best_step_norm = tf.zeros((), dtype=self.precision)
        found_strict_decrease = False

        accepted = False
        accepted_payload: Optional[
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]
        ] = None

        for _ in range(self.max_backtracks):
            theta_trial = self._project(theta_flat - alpha * grad_flat, lower, upper)
            step = theta_trial - theta_flat
            step_norm = tf.reduce_max(tf.abs(step))

            if bool((step_norm <= self.step_tol).numpy()):
                break

            directional_derivative = self._dot(grad_flat, step)
            if (
                not bool(tf.math.is_finite(directional_derivative).numpy())
                or bool((directional_derivative >= -self.eps).numpy())
            ):
                alpha = alpha * self.backtrack_factor
                continue

            self.map.set_theta(self.map.unflatten_theta(theta_trial))
            trial_total, trial_data, trial_reg = self._evaluate_value(inputs)

            finite_trial = bool(
                (
                    tf.math.is_finite(trial_total)
                    & tf.math.is_finite(trial_data)
                    & tf.math.is_finite(trial_reg)
                ).numpy()
            )

            if finite_trial and bool((trial_total < best_total).numpy()):
                best_theta = tf.identity(theta_trial)
                best_total = tf.identity(trial_total)
                best_data = tf.identity(trial_data)
                best_reg = tf.identity(trial_reg)
                best_alpha = tf.identity(alpha)
                best_step_norm = tf.identity(step_norm)
                found_strict_decrease = True

            armijo_rhs = reference_total + self.armijo_c * directional_derivative
            if finite_trial and bool((trial_total <= armijo_rhs).numpy()):
                accepted = True
                accepted_payload = (
                    tf.identity(theta_trial),
                    tf.identity(trial_total),
                    tf.identity(trial_data),
                    tf.identity(trial_reg),
                    tf.identity(alpha),
                    tf.identity(step_norm),
                )
                break

            alpha = alpha * self.backtrack_factor
            if bool((alpha < self.alpha_min).numpy()):
                break

        self.map.set_theta(theta_backup)

        if accepted and accepted_payload is not None:
            theta_new, total_new, data_new, reg_new, alpha_used, step_norm = accepted_payload
            return theta_new, total_new, data_new, reg_new, alpha_used, step_norm, True

        # Conservative fallback: if Armijo fails but a strictly lower finite value was
        # found, accept that value-only descent point rather than declaring failure.
        # This is useful for noisy DA objectives and still preserves monotone descent.
        if found_strict_decrease:
            return best_theta, best_total, best_data, best_reg, best_alpha, best_step_norm, True

        return (
            tf.identity(theta_flat),
            tf.identity(current_total),
            tf.identity(current_data),
            tf.identity(current_reg),
            tf.zeros((), dtype=self.precision),
            tf.zeros((), dtype=self.precision),
            False,
        )

    def _run_step_callback(self, iter_py: int) -> None:
        # Local DA mappings may provide maybe_run_step_callback(); the base mapping
        # provides on_step_end().  Prefer the former when present to stay compatible
        # with the current DA phase runner without hard-coding that mapping type here.
        if hasattr(self.map, "maybe_run_step_callback"):
            self.map.maybe_run_step_callback(iter_py)
        else:
            self.map.on_step_end(tf.constant(iter_py, dtype=tf.int32))

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

        if self._uses_da_display:
            self.display.update(
                int(self.step_state.iter.numpy()),
                float(self.last_total.read_value().numpy()),
                float(self.last_data.read_value().numpy()),
                float(self.last_reg.read_value().numpy()),
                values,
                satisfied,
            )
        else:
            self.display.update(
                int(self.step_state.iter.numpy()),
                float(self.last_total.read_value().numpy()),
                values,
                satisfied,
            )

    def minimize_impl(self, inputs: tf.Tensor) -> tf.Tensor:
        inputs_full = inputs
        lower, upper = self._bounds()

        # Ensure the initial point is feasible before the first forward solve.
        theta_flat = self.map.flatten_theta(self.map.get_theta())
        theta_flat = self._project(tf.cast(theta_flat, self.precision), lower, upper)
        self.map.set_theta(self.map.unflatten_theta(theta_flat))

        total, data, reg, grad_u, grad_theta, U, V = self._evaluate_grad(inputs_full)
        grad_flat = self.map.flatten_theta(grad_theta)
        self._init_step_state(U, V, theta_flat)

        halt_status = tf.constant(HaltStatus.CONTINUE.value, dtype=tf.int32)
        iter_last = -1
        costs_hist: list[tf.Tensor] = []
        cost_total_hist: list[tf.Tensor] = []
        cost_data_hist: list[tf.Tensor] = []
        cost_reg_hist: list[tf.Tensor] = []

        theta_prev_flat: Optional[tf.Tensor] = None
        grad_prev_flat: Optional[tf.Tensor] = None

        iter_max = int(self.iter_max.numpy())
        debug_freq = int(tf.convert_to_tensor(self.debug_freq).numpy()) if self.debug_mode else 1

        for iter_py in range(iter_max):
            iter_tf = tf.constant(iter_py, dtype=tf.int32)
            alpha = self._spectral_alpha(
                iteration=iter_py,
                theta_flat=theta_flat,
                theta_prev_flat=theta_prev_flat,
                grad_flat=grad_flat,
                grad_prev_flat=grad_prev_flat,
            )
            reference_total = self._reference_cost(cost_total_hist, total)

            theta_new, _, _, _, _, step_norm, step_accepted = self._projected_armijo_search(
                theta_flat=theta_flat,
                grad_flat=grad_flat,
                current_total=total,
                current_data=data,
                current_reg=reg,
                reference_total=reference_total,
                alpha_start=alpha,
                inputs=inputs_full,
            )

            if not step_accepted:
                halt_status = tf.cond(
                    step_norm <= self.step_tol,
                    lambda: tf.constant(HaltStatus.COMPLETED.value, dtype=tf.int32),
                    lambda: tf.constant(HaltStatus.FAILURE.value, dtype=tf.int32),
                )
                break

            theta_old = theta_flat
            grad_old = grad_flat
            theta_flat = theta_new
            self.map.set_theta(self.map.unflatten_theta(theta_flat))

            # One full gradient evaluation per accepted step.  The value-only line
            # search deliberately avoids gradients at rejected trial points.
            total, data, reg, grad_u, grad_theta, U, V = self._evaluate_grad(inputs_full)
            grad_flat = self.map.flatten_theta(grad_theta)

            theta_prev_flat = theta_old
            grad_prev_flat = grad_old

            costs_hist.append(tf.identity(total))
            cost_total_hist.append(tf.identity(self.last_total.read_value()))
            cost_data_hist.append(tf.identity(self.last_data.read_value()))
            cost_reg_hist.append(tf.identity(self.last_reg.read_value()))

            accepted_iter = iter_py + 1
            if getattr(self.map, "_da_out_freq", 0) > 0 and accepted_iter % int(self.map._da_out_freq) == 0:
                self._publish_cost_history(
                    cost_total_hist,
                    cost_data_hist,
                    cost_reg_hist,
                    accepted_iter,
                )

            self._run_step_callback(iter_py)

            grad_u_norm, grad_theta_norm = self._get_grad_norm(grad_u, grad_theta)
            self._update_step_state(
                iter_tf,
                U,
                V,
                theta_flat,
                total,
                grad_u_norm,
                grad_theta_norm,
            )
            halt_status = self._check_stopping()
            self._update_display()

            if self.debug_mode and debug_freq > 0 and iter_py % debug_freq == 0:
                self._update_debug_state(iter_tf, total, grad_u, grad_theta)
                self._debug_display()

            iter_last = iter_py

            if bool(tf.not_equal(halt_status, HaltStatus.CONTINUE.value).numpy()):
                break
            if bool((step_norm <= self.step_tol).numpy()):
                halt_status = tf.constant(HaltStatus.COMPLETED.value, dtype=tf.int32)
                break

        self._finalize_display(halt_status)

        n_keep = max(0, iter_last + 1)
        self._publish_cost_history(
            cost_total_hist,
            cost_data_hist,
            cost_reg_hist,
            n_keep,
        )
        return self._stack_history(costs_hist[:n_keep], self.precision)
