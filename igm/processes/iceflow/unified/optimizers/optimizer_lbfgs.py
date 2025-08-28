#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
import itertools
import time
from collections import deque
from typing import Callable

from ..mappings import Mapping
from .optimizer import Optimizer
from .line_search import LineSearches, ValueAndGradient

tf.config.optimizer.set_jit(True)


class OptimizerLBFGS(Optimizer):
    def __init__(
        self,
        cost_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
        map: Mapping,
        line_search_method: str,
        iter_max: int = int(1e5),
        alpha_min: float = 0.0,
        memory: int = 10,
        print_cost: bool = False,
    ):
        super().__init__(cost_fn, map)
        self.name = "lbfgs"
        self.print_cost = print_cost

        self.line_search = LineSearches[line_search_method]()
        self.iter_max = tf.Variable(iter_max)
        self.alpha_min = tf.Variable(alpha_min)
        self.memory = memory

    def update_parameters(self, iter_max: int, alpha_min: float) -> None:
        self.iter_max.assign(iter_max)
        self.alpha_min.assign(alpha_min)

    @tf.function(reduce_retracing=True)
    def _two_loop_recursion(
        self, grad: tf.Tensor, s_list: tf.Tensor, y_list: tf.Tensor
    ) -> tf.Tensor:
        q = grad
        alpha_list = tf.TensorArray(dtype=grad.dtype, size=0, dynamic_size=True)
        num_elems = tf.shape(s_list)[0]

        # First loop
        for i in tf.range(num_elems - 1, -1, -1):
            s = s_list[i]
            y = y_list[i]
            rho = 1.0 / tf.tensordot(y, s, axes=1)
            alpha = rho * tf.tensordot(s, q, axes=1)
            alpha_list = alpha_list.write(i, alpha)
            q = q - alpha * y

        def compute_gamma_fn() -> tf.Tensor:
            last_y = y_list[num_elems - 1]
            last_s = s_list[num_elems - 1]
            ys = tf.tensordot(last_y, last_s, axes=1)
            yy = tf.tensordot(last_y, last_y, axes=1)
            return ys / yy

        gamma = tf.cond(
            num_elems > 0, compute_gamma_fn, lambda: tf.constant(1.0, dtype=grad.dtype)
        )

        r = gamma * q

        # Second loop
        for i in tf.range(num_elems):
            s = s_list[i]
            y = y_list[i]
            alpha = alpha_list.read(i)
            rho = 1.0 / tf.tensordot(y, s, axes=1)
            beta = rho * tf.tensordot(y, r, axes=1)
            r = r + s * (alpha - beta)

        return -r

    @tf.function
    def _line_search(
        self, w_flat: tf.Tensor, p_flat: tf.Tensor, input: tf.Tensor
    ) -> tf.Tensor:
        def value_and_gradients_function(alpha: tf.Tensor) -> ValueAndGradient:
            # Backup
            w_backup = self.map.copy_w(self.map.get_w())

            # New w
            w_alpha = w_flat + alpha * p_flat
            w_alpha = self.map.unflatten_w(w_alpha)

            # Compute grad
            self.map.set_w(w_alpha)
            f, _, grad = self._get_grad(input)
            grad_flat = self.map.flatten_w(grad)
            df = tf.reduce_sum(grad_flat * p_flat)

            # Reset backup
            self.map.set_w(w_backup)

            return ValueAndGradient(x=alpha, f=f, df=df)

        return self.line_search.search(w_flat, p_flat, value_and_gradients_function)

    @tf.function(jit_compile=False)
    def minimize(self, inputs: tf.Tensor) -> tf.Tensor:

        # TO DO -- adapt this for multiple batches
        n_batches = inputs.shape[0]
        if n_batches > 1:
            raise NotImplementedError(
                "❌ The optimizer is only implemented for 1 batch for now; "
                + f"n_batches = {n_batches}."
            )
        input = inputs[0, :, :, :, :]

        # Initial state
        w_flat = self.map.flatten_w(self.map.get_w())
        U, V = self.map.get_UV(inputs[0, :, :, :])
        history = deque(maxlen=self.memory)

        # Evaluate at intial point
        cost, grad_u, grad_w = self._get_grad(input)
        grad_w_flat = self.map.flatten_w(grad_w)

        costs = tf.TensorArray(dtype=tf.float32, size=int(self.iter_max))

        for iter in tf.range(self.iter_max):
            # Save previous solution
            U_prev = tf.identity(U)
            V_prev = tf.identity(V)
            w_flat_prev = w_flat
            grad_w_flat_prev = grad_w_flat

            # Compute direction
            if history:
                s_tensor = tf.stack([s for s, _ in history])
                y_tensor = tf.stack([y for _, y in history])
                p_flat = self._two_loop_recursion(grad_w_flat, s_tensor, y_tensor)
            else:
                p_flat = -grad_w_flat

            # Line search
            alpha = self._line_search(w_flat, p_flat, input)
            alpha = tf.maximum(alpha, self.alpha_min)

            # Apply increment
            w_flat += alpha * p_flat
            self.map.set_w(self.map.unflatten_w(w_flat))

            # Evaluate at new point
            cost, grad_u, grad_w = self._get_grad(input)
            grad_w_flat = self.map.flatten_w(grad_w)

            # Update history
            s = w_flat - w_flat_prev
            y = grad_w_flat - grad_w_flat_prev
            if tf.tensordot(y, s, axes=1) > 1e-10:
                history.append((s, y))

            # Post-process
            U, V = self.map.get_UV(input)

            if self.print_cost:
                tf.print("Iteration", iter + 1, "/", self.iter_max, end=" ")
                tf.print(": Cost =", cost)

            costs = costs.write(iter, cost)

        return costs.stack()
