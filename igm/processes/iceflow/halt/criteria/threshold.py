#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Tuple

from .criterion import Criterion
from ..metrics import Metric
from ..step_state import StepState


class CriterionThreshold(Criterion):
    """Criterion satisfied when metric maximum exceeds threshold."""

    def __init__(
        self,
        metric: Metric,
        dtype: str,
        threshold: float,
    ):
        """Initialize threshold criterion."""
        super().__init__(metric, dtype)
        self.threshold = threshold
        self.name = "threshold"

    def check(self, step_state: StepState) -> Tuple[tf.Tensor, tf.Tensor]:
        """Check if maximum absolute metric value exceeds threshold."""
        metric_value = self.metric.compute(step_state)
        metric_max = tf.reduce_max(tf.math.abs(metric_value))
        is_satisfied = tf.greater(metric_max, self.threshold)

        return is_satisfied, metric_max
