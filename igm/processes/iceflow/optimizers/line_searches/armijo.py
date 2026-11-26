#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Callable, Optional

from .line_search import LineSearch, ValueAndGradient


class LineSearchArmijo(LineSearch):
    supports_value_only = True

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

    @tf.function(autograph=False, reduce_retracing=True)
    def search(
        self,
        w: tf.Tensor,
        p: tf.Tensor,
        value_and_grad_fn: Callable[[tf.Tensor], ValueAndGradient],
        val_0: Optional[ValueAndGradient] = None,
        value_fn: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
    ) -> tf.Tensor:
        del p
        dtype = w.dtype
        if val_0 is None:
            val_0 = value_and_grad_fn(tf.constant(0.0, dtype=dtype))
        if value_fn is None:
            value_fn = lambda alpha: value_and_grad_fn(alpha).f

        c1 = tf.cast(self.c1, dtype)
        rho = tf.cast(self.rho, dtype)
        alpha = tf.constant(self.step_size_initial, dtype=w.dtype)
        f_alpha = value_fn(alpha)
        accepted = f_alpha <= val_0.f + c1 * alpha * val_0.df

        def cond(i, alpha, f_alpha, accepted):
            del alpha, f_alpha
            return (i < self.max_iter) & tf.logical_not(accepted)

        def body(i, alpha, f_alpha, accepted):
            del f_alpha, accepted
            alpha = alpha * rho
            f_alpha = value_fn(alpha)
            accepted = f_alpha <= val_0.f + c1 * alpha * val_0.df
            return i + 1, alpha, f_alpha, accepted

        _, alpha, _, _ = tf.while_loop(
            cond,
            body,
            (tf.constant(1), alpha, f_alpha, accepted),
            parallel_iterations=1,
        )

        return alpha
