#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf

from .metric import Metric, StepState


class MetricGradThetaNorm(Metric):
    """Metric for monitoring |∂cost/∂theta|."""

    def compute_impl(self, step_state: StepState) -> tf.Tensor:
        """Return |∂cost/∂theta| from step state."""
        return step_state.grad_theta_norm
