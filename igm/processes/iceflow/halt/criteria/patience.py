#!/usr/bin/env python3
# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Tuple

from .criterion import Criterion
from ..metrics import Metric
from ..step_state import StepState
from igm.utils.math.norms import compute_norm


class CriterionPatience(Criterion):
    """Criterion satisfied when metric has not improved for a specified number of iterations."""

    def __init__(
        self,
        metric: Metric,
        dtype: str,
        patience: int,
        tol: float = 0.0,
        mode: str = "min",
        ord: str = "id",
    ):
        """Initialize patience criterion."""
        super().__init__(metric, dtype)
        self.name = "patience"
        self.mode = mode
        self.ord = ord
        self.patience = tf.constant(int(patience), dtype=tf.int32)
        self.tol = tf.constant(tol, dtype=self.dtype)
        self.metric_value_best = tf.Variable(
            initial_value=tf.zeros([], dtype=self.dtype),
            dtype=self.dtype,
            trainable=False,
            validate_shape=False,
            shape=tf.TensorShape(None),
        )
        self.iter_best = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.init = tf.Variable(False, dtype=tf.bool, trainable=False)
        self.name = "patience"

    def check(self, step_state: StepState) -> Tuple[tf.Tensor, tf.Tensor]:
        """Check if metric has not improved beyond tolerance for patience iterations."""
        metric_value = self.metric.compute(step_state)
        iter = step_state.iter

        def init():
            """Initialize best metric value and iteration."""
            self.metric_value_best.assign(metric_value)
            self.iter_best.assign(iter)
            self.init.assign(True)
            return tf.constant(False), tf.constant(0, dtype=self.dtype)

        def compute():
            """Compare current metric to best and check patience."""
            if self.mode == "min":
                value_delta = self.metric_value_best - metric_value
            else:
                value_delta = metric_value - self.metric_value_best

            improved = tf.greater(compute_norm(value_delta, ord=self.ord), self.tol)

            def improved_true():
                """Update best metric when improvement detected."""
                self.metric_value_best.assign(metric_value)
                self.iter_best.assign(iter)
                return tf.constant(False), tf.constant(0, dtype=self.dtype)

            def improved_false():
                """Check if patience exceeded without improvement."""
                iter_delta = iter - self.iter_best
                is_satisfied = tf.greater_equal(iter_delta, self.patience)
                return is_satisfied, tf.cast(iter_delta, dtype=self.dtype)

            return tf.cond(improved, improved_true, improved_false)

        return tf.cond(self.init, compute, init)

    def reset(self) -> None:
        """Reset patience criterion state."""
        self.init.assign(False)
