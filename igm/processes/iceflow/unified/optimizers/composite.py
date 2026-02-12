#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Callable, List, Tuple
from rich.console import Console
from rich.text import Text
from rich.theme import Theme

from .optimizer import Optimizer
from ..mappings import MappingComposite


class OptimizerComposite(Optimizer):
    """
    Multi-stage optimizer for MappingComposite.  Each stage specifies
    which parameters are active ("all", "grounded", or "floating") and
    which optimizer to use.  Stages run sequentially each time
    minimize() is called.

    Example stage sequence:
        [("all", adam_opt), ("floating", lbfgs_opt)]
        → first optimize everything with Adam, then fine-tune floating with L-BFGS.
    """

    def __init__(
        self,
        cost_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
        map: MappingComposite,
        stages: List[Tuple[str, Optimizer]],
        print_cost: bool = True,
        print_cost_freq: int = 1,
        precision: str = "float32",
        ord_grad_u: str = "l2_weighted",
        ord_grad_theta: str = "l2_weighted",
        **kwargs,
    ):
        if not stages:
            raise ValueError("❌ Composite optimizer requires at least one stage.")

        super().__init__(
            cost_fn,
            map,
            halt=None,
            print_cost=False,
            print_cost_freq=print_cost_freq,
            precision=precision,
            ord_grad_u=ord_grad_u,
            ord_grad_theta=ord_grad_theta,
            **kwargs,
        )

        self.name = "composite"
        self.composite_map = map
        self.stages = [{"active": active, "optimizer": opt} for active, opt in stages]

        self.iter_max = tf.Variable(self._compute_iter_max(), dtype=tf.int32)

        self._print_cost = print_cost
        if print_cost:
            self._console = Console(
                theme=Theme(
                    {
                        "sep": "#e5e7eb",
                        "label": "bold #e5e7eb",
                        "domain": "bold #06b6d4",
                        "optim": "bold #a78bfa",
                    }
                )
            )

    def _compute_iter_max(self) -> int:
        return sum(int(s["optimizer"].iter_max) for s in self.stages)

    def _print_stage_header(self, idx: int, active: str, opt_name: str) -> None:
        if not self._print_cost:
            return
        text = Text()
        text.append(f"Stage {idx + 1}/{len(self.stages)}", style="label")
        text.append(" • ", style="label")
        text.append(active, style="domain")
        text.append(" • ", style="label")
        text.append(opt_name, style="optim")
        self._console.print("━" * self._console.width, style="sep")
        self._console.print(text)
        self._console.print("━" * self._console.width, style="sep")

    def update_parameters(self) -> None:
        self.iter_max.assign(self._compute_iter_max())

    def minimize_impl(self, inputs: tf.Tensor) -> tf.Tensor:
        self.iter_max.assign(self._compute_iter_max())

        costs = tf.TensorArray(
            dtype=self.precision, size=int(self.iter_max), dynamic_size=False
        )
        iter_last = -1

        for idx, stage in enumerate(self.stages):
            active = stage["active"]
            optimizer = stage["optimizer"]

            self._print_stage_header(idx, active, optimizer.name)
            self.composite_map.active = active

            optimizer.sampler = self.sampler
            cost = optimizer.minimize(inputs)

            n = cost.shape[0]
            for i in range(n):
                costs = costs.write(iter_last + 1 + i, cost[i])
            iter_last += n

        # Restore for evaluation
        self.composite_map.active = "all"

        return costs.stack()[: iter_last + 1]
