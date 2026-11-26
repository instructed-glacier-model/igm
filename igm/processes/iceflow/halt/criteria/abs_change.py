#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import tensorflow as tf
from typing import Tuple

from .criterion import Criterion
from ..metrics import Metric
from ..step_state import StepState


class CriterionAbsChange(Criterion):
    """Stop when the absolute change in a vector metric is small."""

    def __init__(
        self,
        metric: Metric,
        dtype: str,
        tol: float,
        reduction: str = "max",
        consecutive: int = 1,
    ):
        """Initialize the absolute-change criterion."""
        super().__init__(metric, dtype)
        if float(tol) <= 0.0:
            raise ValueError("tol must be positive.")
        if reduction not in ("max", "rmse"):
            raise ValueError("reduction must be either 'max' or 'rmse'.")
        if int(consecutive) < 1:
            raise ValueError("consecutive must be at least one.")

        self.tol = tf.constant(tol, dtype=self.dtype)
        self.reduction = reduction
        self.consecutive = tf.constant(consecutive, dtype=tf.int32)
        self.init = tf.Variable(False, dtype=tf.bool, trainable=False)
        self.metric_value_prev = tf.Variable(
            initial_value=tf.zeros([], dtype=self.dtype),
            dtype=self.dtype,
            trainable=False,
            validate_shape=False,
            shape=tf.TensorShape(None),
        )
        self.consecutive_count = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.name = "abs_change"

    def _stack_components(self, metric_value) -> tf.Tensor:
        """Stack a scalar or nested vector metric along a component axis."""
        components = tf.nest.flatten(metric_value)
        return tf.stack(
            [tf.cast(component, self.dtype) for component in components], axis=0
        )

    def _reduce(self, delta: tf.Tensor) -> tf.Tensor:
        """Reduce pointwise vector magnitudes with max or spatial RMSE."""
        squared_magnitude = tf.reduce_sum(tf.square(delta), axis=0)
        if self.reduction == "max":
            return tf.sqrt(tf.reduce_max(squared_magnitude))
        return tf.sqrt(tf.reduce_mean(squared_magnitude))

    def check(self, step_state: StepState) -> Tuple[tf.Tensor, tf.Tensor]:
        """Check the absolute change from the preceding sampled iterate."""
        metric_value = self._stack_components(self.metric.compute(step_state))

        def init():
            self.metric_value_prev.assign(metric_value)
            self.consecutive_count.assign(0)
            self.init.assign(True)
            return tf.constant(False), tf.constant(np.nan, self.dtype)

        def compute():
            change = self._reduce(metric_value - self.metric_value_prev)
            below_tolerance = tf.less_equal(change, self.tol)
            count = tf.where(
                below_tolerance,
                self.consecutive_count + 1,
                tf.constant(0, tf.int32),
            )
            self.metric_value_prev.assign(metric_value)
            self.consecutive_count.assign(count)
            return tf.greater_equal(count, self.consecutive), change

        return tf.cond(self.init, compute, init)

    def reset(self) -> None:
        """Reset the previous iterate and consecutive-check counter."""
        self.init.assign(False)
        self.consecutive_count.assign(0)
