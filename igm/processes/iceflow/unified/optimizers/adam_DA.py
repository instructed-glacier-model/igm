#!/usr/bin/env python3
# Copyright (C) 2021-2025
# GNU GPL v3

from __future__ import annotations
import tensorflow as tf
from typing import Any, Callable, Tuple

from ..mappings import Mapping
from .adam import OptimizerAdam
from .da_progress_optimizer import _DAProgressOptimizer
from rich.theme import Theme
progress_theme = Theme(
    {
        "label": "bold #e5e7eb",
        "value.cost": "#f59e0b",
        "value.grad": "#06b6d4",
        "value.delta": "#a78bfa",
        "bar.incomplete": "grey35",
        "bar.complete": "#22c55e",
    }
)


class OptimizerAdamDataAssimilation(OptimizerAdam):
    """
    Adam specialization for data assimilation.

    Adds:
      - storage & display of total/data/reg costs
      - DA-specific progress display matching LBFGS DA
    """

    def __init__(
        self,
        cost_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
        map: Mapping,
        precision: str,
        print_cost: bool = True,
        print_cost_freq: int = 10,
        lr: float = 1e-3,
        iter_max: int = int(1e5),
        lr_decay: float = 0.0,
        lr_decay_steps: int = 1000,
        batch_size: int = 1,
        **kwargs: Any,
    ):
        super().__init__(
            cost_fn=cost_fn,
            map=map,
            print_cost=print_cost,
            print_cost_freq=print_cost_freq,
            precision=precision,
            lr=lr,
            iter_max=iter_max,
            lr_decay=lr_decay,
            lr_decay_steps=lr_decay_steps,
            batch_size=batch_size,
        )

        self.name = "ADAM_Data_Assimilation"

        dtype = getattr(self.map, "precision", tf.float32)

        self.last_total = tf.Variable(0.0, trainable=False, dtype=dtype)
        self.last_data = tf.Variable(0.0, trainable=False, dtype=dtype)
        self.last_reg = tf.Variable(0.0, trainable=False, dtype=dtype)

        # Use the same DA display as LBFGS DA
        self.display = _DAProgressOptimizer(
            enabled=self.display.enabled,
            freq=self.display.freq,
        )

    def _update_display(self) -> None:
        if not getattr(self.display, "enabled", False):
            return

        def update_display(iter_val, total_val, data_val, reg_val, *crit_data):
            if crit_data:
                n = len(crit_data) // 2
                values = [float(crit_data[i].numpy()) for i in range(n)]
                satisfied = [bool(crit_data[n + i].numpy()) for i in range(n)]
            else:
                values = None
                satisfied = None

            self.display.update(
                int(iter_val.numpy()),
                float(total_val.numpy()),
                float(data_val.numpy()),
                float(reg_val.numpy()),
                values,
                satisfied,
            )
            return 1.0

        should_update = self.display.should_update(self.step_state.iter)

        py_func_args = [
            self.step_state.iter,
            self.last_total.read_value(),
            self.last_data.read_value(),
            self.last_reg.read_value(),
        ]

        if self.halt_state.criterion_values and self.halt_state.criterion_satisfied:
            py_func_args.extend(self.halt_state.criterion_values)
            py_func_args.extend(self.halt_state.criterion_satisfied)

        tf.cond(
            should_update,
            lambda: tf.py_function(update_display, py_func_args, tf.float32),
            lambda: tf.constant(0.0, dtype=tf.float32),
        )

    @tf.function
    def _get_grad(
        self, inputs: tf.Tensor
    ) -> Tuple[tf.Tensor, list[tf.Tensor], list[tf.Tensor]]:
        theta = self.map.get_theta()

        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            for t in theta:
                tape.watch(t)

            U, V = self.map.get_UV(inputs)

            # Prefer the synchronized inputs already held by the DA mapping if available
            inputs_used = self.map.inputs if hasattr(self.map, "inputs") else inputs

            total, data, reg = self.cost_fn(U, V, inputs_used)

        grad_u = tape.gradient(total, [U, V])
        grad_theta = tape.gradient(total, theta)
        del tape

        # Robust against disconnected variables
        grad_theta = [tf.zeros_like(t) if g is None else g for g, t in zip(grad_theta, theta)]

        # Store the DA cost breakdown for display / diagnostics
        self.last_total.assign(tf.stop_gradient(total))
        self.last_data.assign(tf.stop_gradient(data))
        self.last_reg.assign(tf.stop_gradient(reg))

        return total, grad_u, grad_theta