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
        self.print_cost = print_cost

        version_tf = int(tf.__version__.split(".")[1])
        if (version_tf <= 10) | (version_tf >= 16):
            module_optimizer = tf.keras.optimizers
        else:
            module_optimizer = tf.keras.optimizers.legacy

        self.iter_max = tf.Variable(iter_max)
        self.optim_adam = module_optimizer.Adam(learning_rate=tf.Variable(lr))

    def update_parameters(self, iter_max: int, lr: float) -> None:
        self.iter_max.assign(iter_max)
        self.optim_adam.learning_rate.assign(lr)

    @tf.function(jit_compile=False)
    def minimize(self, inputs: tf.Tensor) -> tf.Tensor:

        # Initial state
        w = self.map.get_w()
        U, V = self.map.get_UV(inputs[0, :, :, :])
        n_batches = inputs.shape[0]

        costs = tf.TensorArray(dtype=tf.float32, size=int(self.iter_max))

        for iter in tf.range(self.iter_max):

            batch_costs = tf.TensorArray(dtype=tf.float32, size=n_batches)

            for i in tf.range(n_batches):
                input = inputs[i, :, :, :, :]

                # Save previous solution
                U_prev = tf.identity(U)
                V_prev = tf.identity(V)

                # Compute cost and grad
                cost, grad_u, grad_w = self._get_grad(input)

                # Apply Adam descent
                self.optim_adam.apply_gradients(zip(grad_w, w))

                # Post-process
                U, V = self.map.get_UV(input)

                # Store cost for this batch
                batch_costs = batch_costs.write(i, cost)

            iter_cost = tf.reduce_mean(batch_costs.stack())

            if self.print_cost:
                tf.print("Iteration", iter + 1, "/", self.iter_max, end=" ")
                tf.print(": Cost =", iter_cost)

            costs = costs.write(iter, iter_cost)

        return costs.stack()
