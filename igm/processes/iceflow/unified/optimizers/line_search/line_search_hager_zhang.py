#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Callable

from .line_search import LineSearch, ValueAndGradient


class LineSearchHagerZhang(LineSearch):
    def __init__(self, step_size_initial: float = 1.0):
        super().__init__(step_size_initial)

        self.name = "hager-zhang"
        try:
            import tensorflow_probability as tfp

            self.tfp = tfp
        except ImportError:
            raise ImportError(
                "❌ TensorFlow Probability is required for HagerZhangLineSearch."
            )

    @tf.function
    def search(
        self,
        w: tf.Tensor,
        p: tf.Tensor,
        value_and_grad_fn: Callable[[tf.Tensor], ValueAndGradient],
    ) -> tf.Tensor:
        result = self.tfp.optimizer.linesearch.hager_zhang(
            value_and_grad_fn,
            initial_step_size=tf.constant(self.step_size_initial, dtype=w.dtype),
        )
        return result.left.x
