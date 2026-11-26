#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf

from .metric import Metric, StepState


class MetricGradUNorm(Metric):
    """Metric for monitoring |∂cost/∂u|."""

    def compute_impl(self, step_state: StepState) -> tf.Tensor:
        """Return |∂cost/∂u| from step state."""
        return step_state.grad_u_norm
