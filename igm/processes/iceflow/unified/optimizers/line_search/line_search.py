#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Callable

# The namedtuple structure is necessary for interfacing with tfp
ValueAndGradient = namedtuple("ValueAndGradient", ["x", "f", "df"])


class LineSearch(ABC):
    def __init__(self, step_size_initial: float = 1.0):
        self.name = ""
        self.step_size_initial = step_size_initial

    @abstractmethod
    @tf.function
    def search(
        self,
        w: tf.Tensor,
        p: tf.Tensor,
        value_and_grad_fn: Callable[[tf.Tensor], ValueAndGradient],
    ) -> tf.Tensor:
        raise NotImplementedError(
            "❌ The search function is not implemented in this class."
        )
