#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Callable, Optional

# The namedtuple structure is necessary for interfacing with tfp
ValueAndGradient = namedtuple("ValueAndGradient", ["x", "f", "df"])


class LineSearch(ABC):
    supports_value_only = False

    def __init__(self, step_size_initial: float = 1.0):
        self.name = ""
        self.step_size_initial = step_size_initial

    def select_value_fn(
        self,
        value_fn: Callable[[tf.Tensor], tf.Tensor],
    ) -> Optional[Callable[[tf.Tensor], tf.Tensor]]:
        """Return the cheaper trial evaluator when this search can use it."""
        return value_fn if self.supports_value_only else None

    @abstractmethod
    @tf.function
    def search(
        self,
        w: tf.Tensor,
        p: tf.Tensor,
        value_and_grad_fn: Callable[[tf.Tensor], ValueAndGradient],
        val_0: Optional[ValueAndGradient] = None,
        value_fn: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
    ) -> tf.Tensor:
        raise NotImplementedError(
            "❌ The search function is not implemented in this class."
        )
