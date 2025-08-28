#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Callable

from .line_search import LineSearch, ValueAndGradient


class LineSearchArmijo(LineSearch):
    def __init__(
        self,
        step_size_initial: float = 1.0,
        c1: float = 1e-4,
        rho: float = 0.5,
        max_iter: int = 20,
    ):
        super().__init__(step_size_initial)
        self.name = "armijo"
        self.c1 = c1
        self.rho = rho
        self.max_iter = max_iter

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

        # Line search parameters
        c1 = tf.constant(self.c1, dtype=w.dtype)
        rho = tf.constant(self.rho, dtype=w.dtype)
        alpha = tf.constant(self.step_size_initial, dtype=w.dtype)

        # Backtracking loop
        for _ in tf.range(self.max_iter):
            vg_alpha = value_and_grad_fn(alpha)
            f_alpha = vg_alpha.f

            # Armijo condition: f(x + alpha*p) <= f(x) + c1*alpha*df0
            armijo_condition = f_alpha <= f0 + c1 * alpha * df0

            if armijo_condition:
                break

            alpha = alpha * rho

        return alpha
