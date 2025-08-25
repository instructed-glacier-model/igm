#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
import itertools
import time
from typing import Callable

from ..mappings import Mapping
from .optimizer import Optimizer

tf.config.optimizer.set_jit(True)


class OptimizerAdam(Optimizer):
    def __init__(
        self,
        cost_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
        map: Mapping,
        lr: float = 1e-3,
        iter_max: int = int(1e5),
        print_cost: bool = False,
    ):
        super().__init__(cost_fn, map)
        self.name = "adam"
        self.iter_max = iter_max
        self.print_cost = print_cost

        version_tf = int(tf.__version__.split(".")[1])
        if (version_tf <= 10) | (version_tf >= 16):
            module_optimizer = tf.keras.optimizers
        else:
            module_optimizer = tf.keras.optimizers.legacy

        self.optim_adam = module_optimizer.Adam(learning_rate=lr)

    def update_parameters(self, lr: float, iter_max: int) -> None:
        self.iter_max = iter_max
        self.optim_adam.lr = lr

    @tf.function(jit_compile=False)
    def minimize(self, inputs: tf.Tensor) -> tf.Tensor:

        # Initial state
        w = self.map.get_w()
        U, V = self.map.get_UV(inputs[0, :, :, :])

        # TO DO -- adapt this for multiple batches
        n_batches = inputs.shape[0]
        if n_batches > 1:
            raise NotImplementedError(
                "❌ The optimizer is only implemented for 1 batch for now; "
                + f"n_batches = {n_batches}."
            )

        costs = tf.TensorArray(dtype=tf.float32, size=self.iter_max)

        for iter in range(self.iter_max):
            input = inputs[0, :, :, :, :]

            # Save previous solution
            U_prev = tf.identity(U)
            V_prev = tf.identity(V)

            # Compute cost and grad
            cost, grad_u, grad_w = self._get_grad(input)

            # Apply Adam descent
            self.optim_adam.apply_gradients(zip(grad_w, w))

            # Post-process
            U, V = self.map.get_UV(input)

            if self.print_cost:
                tf.print("Iteration", iter + 1, "/", self.iter_max, end=" ")
                tf.print(": Cost =", cost)

            costs = costs.write(iter, cost)

        return costs.stack()
