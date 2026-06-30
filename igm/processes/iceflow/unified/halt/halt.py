#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import tensorflow as tf
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple

from .criteria import Criterion
from .step_state import StepState
from igm.utils.math.precision import normalize_precision


class HaltStatus(Enum):
    """Status codes for halting optimization process."""

    CONTINUE = auto()
    SUCCESS = auto()
    FAILURE = auto()
    COMPLETED = auto()


@dataclass
class HaltState:
    """State information for halting criteria evaluation."""

    status: tf.Tensor
    criterion_values: List[tf.Tensor]
    criterion_satisfied: List[tf.Tensor]

    @staticmethod
    def empty():
        """Create empty halt state with CONTINUE status."""
        return HaltState(
            status=tf.constant(HaltStatus.CONTINUE.value),
            criterion_values=[],
            criterion_satisfied=[],
        )


class Halt:
    """Halting manager for optimization with success and failure criteria."""

    def __init__(
        self,
        crit_success: Optional[List[Criterion]] = None,
        crit_failure: Optional[List[Criterion]] = None,
        freq: int = 1,
        dtype: str = "float32",
        success_mode: str = "any",
    ):
        """Initialize halting manager."""
        self.crit_success = crit_success or []
        self.crit_failure = crit_failure or []
        self.freq = freq
        self.dtype = normalize_precision(dtype)
        self.success_mode = success_mode
        self.criterion_names = self._build_criterion_names()

    def _build_criterion_names(self) -> List[str]:
        """Build list of criterion names."""
        names = []
        for crit in self.crit_success:
            names.append(crit.name)
        return names

    def reset_all(self) -> None:
        """Reset all success and failure criteria."""
        for crit in self.crit_success:
            crit.reset()
        for crit in self.crit_failure:
            crit.reset()

    def check(
        self, iter: tf.Tensor, step_state: StepState
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Check halting criteria and return status with criterion values and satisfaction flags."""
        do_check = tf.equal(tf.math.mod(iter, self.freq), 0)

        def check_criteria():
            """Evaluate all criteria and determine halt status."""
            success_values = []
            success_satisfied = []

            # Failure criteria (checked but not returned for display)
            failure = tf.constant(False)
            for crit in self.crit_failure:
                is_sat, val = crit.check(step_state)
                failure = tf.logical_or(failure, is_sat)

            # Success criteria (checked and returned for display)
            success = tf.constant(self.success_mode == "all" and len(self.crit_success) > 0)
            for crit in self.crit_success:
                is_sat, val = crit.check(step_state)
                success_values.append(val)
                success_satisfied.append(is_sat)
                if self.success_mode == "all":
                    success = tf.logical_and(success, is_sat)
                else:
                    success = tf.logical_or(success, is_sat)

            # Determine status
            if failure:
                self.reset_all()
                return (
                    tf.constant(HaltStatus.FAILURE.value),
                    success_values,
                    success_satisfied,
                )

            if success:
                self.reset_all()
                return (
                    tf.constant(HaltStatus.SUCCESS.value),
                    success_values,
                    success_satisfied,
                )

            return (
                tf.constant(HaltStatus.CONTINUE.value),
                success_values,
                success_satisfied,
            )

        def no_check():
            """Return CONTINUE status without evaluating criteria."""
            n_success_crit = len(self.crit_success)
            return (
                tf.constant(HaltStatus.CONTINUE.value),
                [tf.constant(np.nan, self.dtype)] * n_success_crit,
                [tf.constant(False)] * n_success_crit,
            )

        return tf.cond(do_check, check_criteria, no_check)
