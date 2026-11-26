#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Tuple

from .criterion import Criterion
from ..metrics import Metric
from ..step_state import StepState


class CriterionNaN(Criterion):
    """Criterion satisfied when metric contains NaN values."""

    def __init__(self, metric: Metric, dtype: str):
        """Initialize NaN detection criterion."""
        super().__init__(metric, dtype)
        self.name = "nan"

    def check(self, step_state: StepState) -> Tuple[tf.Tensor, tf.Tensor]:
        """Check if metric contains any NaN values."""
        metric_value = self.metric.compute(step_state)
        is_satisfied = tf.reduce_any(tf.math.is_nan(metric_value))
        return is_satisfied, is_satisfied
