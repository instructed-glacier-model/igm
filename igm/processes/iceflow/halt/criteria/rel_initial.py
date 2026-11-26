#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from typing import Tuple

import tensorflow as tf

from .criterion import Criterion
from ..metrics import Metric
from ..step_state import StepState
from igm.utils.math.norms import compute_norm


class CriterionRelInitial(Criterion):
    """Criterion based on the metric norm relative to its initial norm."""

    def __init__(self, metric: Metric, dtype: str, tol: float, ord: str):
        super().__init__(metric, dtype)
        self.tol = tol
        self.ord = ord
        self.initialized = tf.Variable(False, dtype=tf.bool, trainable=False)
        self.initial_norm = tf.Variable(
            tf.zeros([], dtype=self.dtype),
            dtype=self.dtype,
            trainable=False,
        )
        self.name = "rel_initial"

    def check(self, step_state: StepState) -> Tuple[tf.Tensor, tf.Tensor]:
        metric_norm = tf.cast(
            compute_norm(self.metric.compute(step_state), ord=self.ord),
            self.dtype,
        )

        def initialize():
            self.initial_norm.assign(metric_norm)
            self.initialized.assign(True)
            return tf.constant(False), tf.constant(1.0, self.dtype)

        def compare():
            tiny = tf.cast(1e-300 if self.dtype == tf.float64 else 1e-30, self.dtype)
            ratio = metric_norm / tf.maximum(self.initial_norm, tiny)
            return tf.less(ratio, self.tol), ratio

        return tf.cond(self.initialized, compare, initialize)

    def reset(self) -> None:
        self.initialized.assign(False)
