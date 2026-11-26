#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

from ..step_state import StepState


TypeMetric = Union[tf.Tensor, Tuple[tf.Tensor, ...]]


class Metric(ABC):
    """Abstract base class for optimization metrics."""

    def __init__(self, normalize_factor: Optional[float] = None):
        """Initialize metric with optional normalization factor."""
        self.normalize_factor = normalize_factor

    @abstractmethod
    def compute_impl(self, step_state: StepState) -> TypeMetric:
        """Compute raw metric value (must be implemented by subclasses)."""
        raise NotImplementedError(
            "❌ The compute method is not implemented in this class."
        )

    def compute(self, step_state) -> TypeMetric:
        """Compute metric value and apply normalization if specified."""
        metric_value = self.compute_impl(step_state)

        if self.normalize_factor is None:
            return metric_value

        return tf.nest.map_structure(lambda m: m / self.normalize_factor, metric_value)
