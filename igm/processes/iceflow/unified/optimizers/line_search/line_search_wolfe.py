#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Callable

from .line_search import LineSearch, ValueAndGradient


class LineSearchWolfe(LineSearch):
    def __init__(
        self,
        step_size_initial: float = 1.0,
        c1: float = 1e-4,
        c2: float = 0.9,
        max_iter: int = 20,
        zoom_max_iter: int = 10,
    ):
        super().__init__(step_size_initial)
        self.name = "wolfe"
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter
        self.zoom_max_iter = zoom_max_iter

    @tf.function
    def search(
        self,
        w: tf.Tensor,
        p: tf.Tensor,
        value_and_grad_fn: Callable[[tf.Tensor], ValueAndGradient],
    ) -> tf.Tensor:
        # Initial evaluation
        vg0 = value_and_grad_fn(tf.constant(0.0, dtype=w.dtype))
        f0, df0 = vg0.f, vg0.df

        # Constants
        c1 = tf.constant(self.c1, dtype=w.dtype)
        c2 = tf.constant(self.c2, dtype=w.dtype)

        alpha_prev = tf.constant(0.0, dtype=w.dtype)
        alpha = tf.constant(self.step_size_initial, dtype=w.dtype)
        f_prev = f0

        # State variables for loop control
        found = tf.constant(False)
        result_alpha = alpha

        # Main loop
        def condition(i, alpha, alpha_prev, f_prev, found, result_alpha):
            return tf.logical_and(i < self.max_iter, tf.logical_not(found))

        def body(i, alpha, alpha_prev, f_prev, found, result_alpha):
            vg_alpha = value_and_grad_fn(alpha)
            f_alpha, df_alpha = vg_alpha.f, vg_alpha.df

            # Check Armijo condition
            armijo_violated = tf.logical_or(
                f_alpha > f0 + c1 * alpha * df0,
                tf.logical_and(f_alpha >= f_prev, i > 0),
            )

            # Check curvature condition
            curvature_satisfied = tf.abs(df_alpha) <= -c2 * df0

            # Check if derivative is positive
            derivative_positive = df_alpha >= 0.0

            def zoom_case():
                return tf.cond(
                    armijo_violated,
                    lambda: self._zoom(
                        alpha_prev,
                        alpha,
                        value_and_grad_fn,
                        f0,
                        df0,
                        c1,
                        c2,
                    ),
                    lambda: self._zoom(
                        alpha,
                        alpha_prev,
                        value_and_grad_fn,
                        f0,
                        df0,
                        c1,
                        c2,
                    ),
                )

            def continue_case():
                return alpha * 2.0

            new_result_alpha = tf.cond(
                tf.logical_or(armijo_violated, derivative_positive),
                zoom_case,
                lambda: tf.cond(curvature_satisfied, lambda: alpha, continue_case),
            )

            new_found = tf.logical_or(
                tf.logical_or(armijo_violated, derivative_positive), curvature_satisfied
            )

            new_alpha_prev = alpha
            new_f_prev = f_alpha
            new_alpha = tf.cond(new_found, lambda: alpha, lambda: new_result_alpha)

            return (
                i + 1,
                new_alpha,
                new_alpha_prev,
                new_f_prev,
                new_found,
                new_result_alpha,
            )

        _, final_alpha, _, _, found, result_alpha = tf.while_loop(
            condition,
            body,
            [tf.constant(0), alpha, alpha_prev, f_prev, found, result_alpha],
        )

        return tf.cond(found, lambda: result_alpha, lambda: final_alpha)

    @tf.function
    def _zoom(
        self,
        alpha_lo: tf.Tensor,
        alpha_hi: tf.Tensor,
        value_and_grad_fn: Callable[[tf.Tensor], ValueAndGradient],
        f0: tf.Tensor,
        df0: tf.Tensor,
        c1: tf.Tensor,
        c2: tf.Tensor,
    ) -> tf.Tensor:

        found = tf.constant(False)
        result_alpha = (alpha_lo + alpha_hi) / 2.0

        def condition(i, alpha_lo, alpha_hi, found, result_alpha):
            return tf.logical_and(i < self.zoom_max_iter, tf.logical_not(found))

        def body(i, alpha_lo, alpha_hi, found, result_alpha):
            # Bisection
            alpha_j = (alpha_lo + alpha_hi) / 2.0
            vg_j = value_and_grad_fn(alpha_j)
            f_j, df_j = vg_j.f, vg_j.df

            vg_lo = value_and_grad_fn(alpha_lo)
            f_lo = vg_lo.f

            # Check conditions
            armijo_violated = tf.logical_or(f_j > f0 + c1 * alpha_j * df0, f_j >= f_lo)
            curvature_satisfied = tf.abs(df_j) <= -c2 * df0

            def update_hi():
                return alpha_lo, alpha_j, tf.constant(False), result_alpha

            def check_curvature():
                def found_solution():
                    return alpha_lo, alpha_hi, tf.constant(True), alpha_j

                def update_bounds():
                    new_alpha_hi = tf.cond(
                        df_j * (alpha_hi - alpha_lo) >= 0.0,
                        lambda: alpha_lo,
                        lambda: alpha_hi,
                    )
                    return alpha_j, new_alpha_hi, tf.constant(False), result_alpha

                return tf.cond(curvature_satisfied, found_solution, update_bounds)

            new_alpha_lo, new_alpha_hi, new_found, new_result_alpha = tf.cond(
                armijo_violated, update_hi, check_curvature
            )

            return i + 1, new_alpha_lo, new_alpha_hi, new_found, new_result_alpha

        _, final_alpha_lo, final_alpha_hi, found, result_alpha = tf.while_loop(
            condition, body, [tf.constant(0), alpha_lo, alpha_hi, found, result_alpha]
        )

        return tf.cond(
            found, lambda: result_alpha, lambda: (final_alpha_lo + final_alpha_hi) / 2.0
        )
