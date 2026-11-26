#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Tuple

from .criterion import Criterion
from ..metrics import Metric
from ..step_state import StepState
from igm.utils.math.norms import compute_norm


class CriterionAbsTol(Criterion):
    """Criterion satisfied when metric norm falls below absolute tolerance."""

    def __init__(self, metric: Metric, dtype: str, tol: float, ord: str):
        """Initialize absolute tolerance criterion."""
        super().__init__(metric, dtype)
        self.tol = tol
        self.ord = ord
        self.name = "abs_tol"

    def check(self, step_state: StepState) -> Tuple[tf.Tensor, tf.Tensor]:
        """Check if metric norm is below absolute tolerance."""
        metric_value = self.metric.compute(step_state)
        metric_norm = compute_norm(metric_value, ord=self.ord)
        is_satisfied = tf.less(metric_norm, self.tol)

        return is_satisfied, metric_value
