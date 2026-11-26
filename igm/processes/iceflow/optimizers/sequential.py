#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Callable, List
from rich.console import Console
from rich.text import Text
from rich.theme import Theme

from .optimizer import Optimizer
from ..mappings import Mapping


class OptimizerSequential(Optimizer):
    def __init__(
        self,
        cost_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
        map: Mapping,
        optimizers: List[Optimizer],
        print_cost: bool = True,
        print_cost_freq: int = 1,
        precision: str = "float32",
        ord_grad_u: str = "l2_weighted",
        ord_grad_theta: str = "l2_weighted",
    ):
        if not optimizers:
            raise ValueError("❌ Sequential optimizer requires at least one optimizer.")

        super().__init__(
            cost_fn,
            map,
            halt=None,
            print_cost=False,
            print_cost_freq=print_cost_freq,
            precision=precision,
            ord_grad_u=ord_grad_u,
            ord_grad_theta=ord_grad_theta,
        )

        self.name = "sequential"
        self.optimizers = optimizers
        self.print_cost = print_cost
        self.iter_max = tf.Variable(self._compute_iter_max(), dtype=tf.int32)
        if print_cost:
            self.theme = Theme(
                {
                    "stage.separator": "#e5e7eb",
                    "stage.label": "bold #e5e7eb",
                    "stage.optimizer": "bold #06b6d4",
                }
            )
            self.console = Console(theme=self.theme)

    def _compute_iter_max(self) -> int:
        return sum(int(optimizer.iter_max) for optimizer in self.optimizers)

    def _print_stage_header(self, stage: int, optimizer_name: str) -> None:
        if not self.print_cost:
            return

        stage_text = Text()
        stage_text.append(
            f"Stage {stage + 1}/{len(self.optimizers)}", style="stage.label"
        )
        stage_text.append(" • ", style="stage.label")
        stage_text.append(optimizer_name, style="stage.optimizer")

        self.console.print("━" * self.console.width, style="stage.separator")
        self.console.print(stage_text)
        self.console.print("━" * self.console.width, style="stage.separator")

    def update_parameters(self) -> None:
        return

    def minimize_impl(self, inputs: tf.Tensor) -> tf.Tensor:
        self.iter_max.assign(self._compute_iter_max())

        costs = tf.TensorArray(
            dtype=self.precision, size=int(self.iter_max), dynamic_size=False
        )

        iter_last = -1

        for stage, optimizer in enumerate(self.optimizers):

            self._print_stage_header(stage, optimizer.name)

            cost = optimizer.minimize(inputs)

            iter_stage = cost.shape[0]
            for i in range(iter_stage):
                costs = costs.write(iter_last + 1 + i, cost[i])

            iter_last = iter_last + iter_stage

        return costs.stack()[: iter_last + 1]
