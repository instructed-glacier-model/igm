#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from abc import ABC, abstractmethod
from typing import Tuple

import tensorflow as tf

from igm.utils.math.precision import normalize_precision

from ..metrics import Metric
from ..step_state import StepState


class Criterion(ABC):
    """Abstract base class for halting criterion."""

    def __init__(
        self,
        metric: Metric,
        dtype: str = "float32",
    ):
        """Initialize halting criterion."""
        self.metric = metric
        self.dtype = normalize_precision(dtype)
        self.name = "crit"

    @abstractmethod
    def check(self, step_state: StepState) -> Tuple[tf.Tensor, tf.Tensor]:
        """Check if criterion is satisfied (must be implemented by subclasses)."""
        raise NotImplementedError(
            "❌ The check method is not implemented in this class."
        )

    def reset(self) -> None:
        """Reset criterion state (override in subclasses if needed)."""
        pass
