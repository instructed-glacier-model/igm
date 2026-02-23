#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Callable, Optional

from .optimizer import Optimizer
from ..mappings import Mapping
from ..halt import Halt, HaltStatus

tf.config.optimizer.set_jit(True)


class OptimizerAdam(Optimizer):

    def __init__(
        self,
        cost_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
        map: Mapping,
        halt: Optional[Halt] = None,
        print_cost: bool = True,
        print_cost_freq: int = 1,
        precision: str = "float32",
        ord_grad_u: str = "l2_weighted",
        ord_grad_theta: str = "l2_weighted",
        clip_norm: Optional[float] = None,
        lr: float = 1e-3,
        iter_max: int = int(1e5),
        lr_decay: float = 0.0,
        lr_decay_steps: int = 1000,
        batch_size: int = 1,
        **kwargs
    ):
        super().__init__(
            cost_fn,
            map,
            halt,
            print_cost,
            print_cost_freq,
            precision,
            ord_grad_u,
            ord_grad_theta,
            **kwargs
        )
        self.name = "adam"
        self.batch_size = int(batch_size)

        version_tf = int(tf.__version__.split(".")[1])
        if (version_tf <= 10) | (version_tf >= 16):
            module_optimizer = tf.keras.optimizers
        else:
            module_optimizer = tf.keras.optimizers.legacy

        self.iter_max = tf.Variable(iter_max)
        if lr_decay > 0.0:
            schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=lr,
                decay_steps=lr_decay_steps,
                decay_rate=lr_decay,
            )
            self.optim_adam = module_optimizer.Adam(
                learning_rate=schedule, beta_1=0.8, beta_2=0.9995, clipnorm=clip_norm
            )
        else:
            self.optim_adam = module_optimizer.Adam(
                learning_rate=tf.Variable(lr),
                beta_1=0.8,
                beta_2=0.9995,
                clipnorm=clip_norm,
            )

    def update_parameters(
        self, iter_max: int, lr: float, lr_decay: float, lr_decay_steps: int
    ) -> None:
        self.iter_max.assign(iter_max)
        self.optim_adam.learning_rate.assign(lr)
        self.lr_decay = lr_decay
        self.lr_decay_steps = lr_decay_steps

    @tf.function(jit_compile=False)
    def minimize_impl(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Args:
            inputs: [N, ly, lx, C] non-overlapping patches. Each outer iteration
                    loops over all N patches (mini-batch SGD with batch_size B).
        """
        theta = self.map.get_theta()
        B = self.batch_size  # Python int — static at trace time
        # Use static shape so the inner loop unrolls at trace time (no while_loop)
        n_batches = max(1, int(inputs.shape[0]) // B)  # Python int

        # Define batch before loops so AutoGraph can infer a consistent type
        batch = inputs[0:B, :, :, :]
        U, V = self.map.get_UV(batch)
        self._init_step_state(U, V, theta)

        halt_status = tf.constant(HaltStatus.CONTINUE.value, dtype=tf.int32)
        iter_last = tf.constant(-1, dtype=tf.int32)
        costs = tf.TensorArray(dtype=self.precision, size=int(self.iter_max))

        for iter in tf.range(self.iter_max):

            cost_sum = tf.constant(0.0, dtype=self.precision)
            grad_u_norm_sum = tf.constant(0.0, dtype=self.precision)
            grad_theta_norm_sum = tf.constant(0.0, dtype=self.precision)

            for b in range(n_batches):  # Python range: unrolled at trace time
                # b and B are Python ints → StridedSlice infers shape [B, ly, lx, C] statically
                batch = inputs[b * B : (b + 1) * B, :, :, :]
                cost, grad_u, grad_theta = self._get_grad(batch)
                self.optim_adam.apply_gradients(zip(grad_theta, theta))

                grad_u_norm, grad_theta_norm = self._get_grad_norm(grad_u, grad_theta)

                cost_sum = cost_sum + cost
                grad_u_norm_sum = grad_u_norm_sum + grad_u_norm
                grad_theta_norm_sum = grad_theta_norm_sum + grad_theta_norm

            n_batches_f = tf.constant(float(n_batches), dtype=self.precision)
            cost_avg = cost_sum / n_batches_f
            grad_u_norm_avg = grad_u_norm_sum / n_batches_f
            grad_theta_norm_avg = grad_theta_norm_sum / n_batches_f

            # TODO: check if this is necessary
            self.map.on_step_end(iter)

            costs = costs.write(iter, cost_avg)

            U, V = self.map.get_UV(batch)
            self._update_step_state(
                iter, U, V, theta, cost_avg, grad_u_norm_avg, grad_theta_norm_avg
            )
            halt_status = self._check_stopping()
            self._update_display()

            if self.debug_mode and iter % self.debug_freq == 0:
                self._update_debug_state(iter, cost_avg, grad_u, grad_theta)
                self._debug_display()

            iter_last = iter

            if tf.not_equal(halt_status, HaltStatus.CONTINUE.value):
                break

        self._finalize_display(halt_status)

        return costs.stack()[: iter_last + 1]
