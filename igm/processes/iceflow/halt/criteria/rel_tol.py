#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import tensorflow as tf
from typing import Tuple

from .criterion import Criterion
from ..metrics import Metric
from ..step_state import StepState
from igm.utils.math.norms import compute_norm


class CriterionRelTol(Criterion):
    """Criterion satisfied when relative change in metric falls below tolerance."""

    def __init__(self, metric: Metric, dtype: str, tol: float, ord: str):
        """Initialize relative tolerance criterion."""
        super().__init__(metric, dtype)
        self.tol = tol
        self.ord = ord
        self.init = tf.Variable(False, dtype=tf.bool, trainable=False)
        self.metric_value_prev = tf.Variable(
            initial_value=tf.zeros([], dtype=self.dtype),
            dtype=self.dtype,
            trainable=False,
            validate_shape=False,
            shape=tf.TensorShape(None),
        )
        self.name = "rel_tol"

    def check(self, step_state: StepState) -> Tuple[tf.Tensor, tf.Tensor]:
        """Check if relative change in metric is below tolerance."""
        metric_value = self.metric.compute(step_state)

        def init():
            """Initialize previous metric value."""
            self.metric_value_prev.assign(metric_value)
            self.init.assign(True)
            return tf.constant(False), tf.constant(np.nan, self.dtype)

        def compute():
            """Compute relative change and check against tolerance."""
            num = compute_norm(metric_value - self.metric_value_prev, ord=self.ord)
            denom = compute_norm(self.metric_value_prev, ord=self.ord) + 1e-12
            relative_change = num / denom

            is_satisfied = tf.less(relative_change, self.tol)
            self.metric_value_prev.assign(metric_value)
            return is_satisfied, relative_change

        return tf.cond(self.init, compute, init)

    def reset(self) -> None:
        """Reset relative tolerance criterion state."""
        self.init.assign(False)
