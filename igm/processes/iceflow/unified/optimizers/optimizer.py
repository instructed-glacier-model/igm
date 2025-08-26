#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from abc import ABC, abstractmethod
from typing import Callable, Tuple

from ..mappings import Mapping


class Optimizer(ABC):
    def __init__(
        self,
        cost_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
        map: Mapping,
    ):
        self.name = ""
        self.cost_fn = cost_fn
        self.map = map

    @abstractmethod
    def update_parameters(self) -> None:
        raise NotImplementedError(
            "❌ The parameters update function is not implemented in this class."
        )

    @abstractmethod
    def minimize(self) -> tf.Tensor:
        raise NotImplementedError(
            "❌ The minimize function is not implemented in this class."
        )

    @tf.function
    def _get_grad(
        self, inputs: tf.Tensor
    ) -> Tuple[tf.Tensor, list[tf.Tensor], list[tf.Tensor]]:
        w = self.map.get_w()
        with tf.GradientTape(persistent=True) as tape:
            U, V = self.map.get_UV(inputs)
            cost = self.cost_fn(U, V, inputs)
        grad_u = tape.gradient(cost, [U, V])
        grad_w = tape.gradient(cost, w)
        del tape
        return cost, grad_u, grad_w
