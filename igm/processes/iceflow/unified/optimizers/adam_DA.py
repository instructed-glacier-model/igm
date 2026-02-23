#!/usr/bin/env python3
# Copyright (C) 2021-2025
# GNU GPL v3

from __future__ import annotations
import tensorflow as tf
from typing import Any, Callable, Tuple

from ..mappings import Mapping
from .adam import OptimizerAdam  # base Adam

# ---- pretty progress (same theme as your LBFGS DA) ----
from rich.theme import Theme
from rich.console import Console
from rich.progress import (
    Progress,
    BarColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TextColumn,
)

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
      - storage & display of data/physics cost components
      - rich progress bar with the same look/feel as your LBFGS DA
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

        # Make sure iter_max exists even if lr_decay > 0.0 (base Adam sets it only in one branch)
        if not hasattr(self, "iter_max"):
            self.iter_max = tf.Variable(iter_max)

        # Track individual cost components
        self.data_cost = tf.Variable(0.0, dtype=self.precision, trainable=False, name="data_cost")
        self.physics_cost = tf.Variable(0.0, dtype=self.precision, trainable=False, name="physics_cost")

        # placeholders set during _progress_setup
        self.console: Console | None = None
        self.progress: Progress | None = None
        self.task = None

    # identical shape/signature as your LBFGS-DA version
    @tf.function
    def _get_grad(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, list[tf.Tensor]]:
        w = self.map.get_theta()
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            for wi in w:
                tape.watch(wi)
            U, V = self.map.get_UV(inputs)
            processed_inputs = self.map.synchronize_inputs(inputs)
            # expect cost_fn to return (total, data, physics)
            cost, data_cost, physics_cost = self.cost_fn(U, V, processed_inputs)

        # store components with correct dtype
        self.data_cost.assign(tf.cast(data_cost, self.precision))
        self.physics_cost.assign(tf.cast(physics_cost, self.precision))

        grad_theta = tape.gradient(cost, theta)
        del tape
        return cost, grad_theta

    # same UI as your LBFGS DA
    def _progress_setup(self) -> None:
        if not self.print_cost:
            return

        self.console = Console(theme=progress_theme)
        self.progress = Progress(
            "🚀",
            BarColumn(bar_width=None, style="bar.incomplete", complete_style="bar.complete"),
            MofNCompleteColumn(),
            "[label]•[/]",
            TextColumn("[label]Cost:[/] [value.cost]{task.fields[cost]}"),
            "[label]•[/]",
            TextColumn("[label]Data:[/] [value.cost]{task.fields[data_cost]}"),
            "[label]•[/]",
            TextColumn("[label]Physics:[/] [value.cost]{task.fields[physics_cost]}"),
            "[label]•[/]",
            TextColumn("[label]Time:[/]"),
            TimeElapsedColumn(),
            "[label]•[/]",
            TextColumn("[label]ETA:[/]"),
            TimeRemainingColumn(),
            console=self.console,
            expand=True,
        )
        self.task = self.progress.add_task(
            "progress",
            total=int(self.iter_max.numpy()),
            cost="N/A",
            data_cost="N/A",
            physics_cost="N/A",
        )
        self.progress.start()

    # mirrors your LBFGS-DA logic (incl. halt criterion and convergence by grad-norm)
    def _progress_update(self, iter: tf.Tensor, cost: tf.Tensor, grad_norm: tf.Tensor) -> tf.Tensor:
        do_update = tf.logical_and(
            tf.constant(self.print_cost, dtype=tf.bool),
            tf.equal(tf.math.mod(iter, self.print_cost_freq), 0),
        )

        def update(iter_: tf.Tensor, cost_: tf.Tensor, data_cost_: tf.Tensor, physics_cost_: tf.Tensor) -> float:
            if self.progress is not None:
                self.progress.update(
                    self.task,
                    completed=int(iter_.numpy()) + 1,
                    cost=f"{float(cost_.numpy()):.4e}",
                    data_cost=f"{float(data_cost_.numpy()):.4e}",
                    physics_cost=f"{float(physics_cost_.numpy()):.4e}",
                )
            return 1.0

        tf.cond(
            do_update,
            lambda: tf.py_function(update, [iter, cost, self.data_cost, self.physics_cost], cost.dtype),
            lambda: tf.constant(0.0, dtype=cost.dtype),
        )

        # stopping conditions
        should_check = tf.greater(iter, 0)

        converged = tf.logical_and(should_check, grad_norm < self.convergence_tolerance)
        max_iter_reached = tf.greater_equal(iter + 1, self.iter_max)
        should_stop = tf.logical_or(converged, max_iter_reached)

        def finalize_with_reason(conv_val: tf.Tensor) -> int:
            if self.print_cost:
                if self.progress is not None:
                    self.progress.stop()
                if self.console is None:
                    return 0
                if conv_val.numpy():
                    self.console.print("✅ [bold green]Optimization converged![/bold green] Gradient norm below threshold.")
                else:
                    self.console.print("🏁 [bold blue]Optimization completed![/bold blue] Maximum iterations reached.")
                self.console.print()
            return 0

        tf.cond(
            should_stop,
            lambda: tf.py_function(finalize_with_reason, [converged], tf.int32),
            lambda: tf.constant(0),
        )

        return should_stop
