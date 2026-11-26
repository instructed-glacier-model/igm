#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Rich display of the initial training stopped on its error (``init_stop.display``).

Same look as the optimizer's progress bar (``optimizers/progress_optimizer.py``) and the
runner's panels. It never starts a live display of its own while the progress bar runs:
its rows go through the bar's console, which prints them above the bar.
"""

import contextlib
import math
from typing import Iterator

import numpy as np
import tensorflow as tf
from omegaconf import DictConfig
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Literal colours, as in the progress bar: the rows may print through its console, whose
# theme does not know custom style names.
LABEL = "bold #e5e7eb"
VALUE = "#06b6d4"
GOOD = "#22c55e"
BAD = "#ef4444"
WARN = "#f59e0b"

# Stop reason: border colour, panel title, phrase of the one-line summary
SUMMARY = {
    "targets": (GOOD, "✅ Initial training stopped: targets met", "targets met"),
    "plateau": (
        WARN,
        "⏸  Initial training stopped: no further gain",
        "no further gain",
    ),
    "cap": (
        "blue",
        "🏁 Initial training completed: nbit_init reached",
        "nbit_init reached",
    ),
}


def percent(value: float, target: float) -> str:
    """An error in %, green when it meets its target, red otherwise."""
    if math.isnan(value):
        return "[dim]   —    [/dim]"
    if value <= target:
        style = GOOD
    else:
        style = BAD
    return f"[{style}]{value:6.2f} %[/]"


def settings_table() -> Table:
    table = Table(box=None, show_header=False, padding=(0, 2))
    table.add_column(style=LABEL, no_wrap=True)
    table.add_column()
    return table


class InitStopDisplay:

    def __init__(self, enabled: bool, progress=None):
        self.enabled = enabled
        self.progress = progress
        self._console = Console()
        self._targets = (np.nan, np.nan)
        self._rows = 0

    @property
    def console(self) -> Console:
        """The progress bar's console while it runs (rows print above it), else our own."""
        progress = self.progress
        if progress is not None and progress.enabled and progress.console is not None:
            return progress.console
        return self._console

    def setup(self, cfg: DictConfig) -> None:
        """The settings, before the direct solve."""
        cfg_unified = cfg.processes.iceflow.unified
        cfg_init_stop = cfg_unified.init_stop
        cfg_reference = cfg_init_stop.reference
        self._targets = (
            float(cfg_init_stop.target_grounded),
            float(cfg_init_stop.target_floating),
        )
        if not self.enabled:
            return

        preconditioner = cfg_unified.cg_newton.preconditioner
        if int(cfg_init_stop.patience) > 0:
            patience = f"patience {cfg_init_stop.patience}"
        else:
            patience = "no patience"
        if int(cfg_init_stop.consecutive) > 1:
            consecutive = f" · at {cfg_init_stop.consecutive} checks in a row"
        else:
            consecutive = ""

        table = settings_table()
        table.add_row(
            "Direct solve",
            f"identity + cg_newton · [{VALUE}]{preconditioner}[/] · "
            f"≤ {cfg_reference.nbit} Newton iterations · "
            f"stop at {cfg_reference.tol} m/yr (RMSE change of u)",
        )
        table.add_row(
            "Targets",
            f"grounded ≤ [{VALUE}]{cfg_init_stop.target_grounded} %[/] · "
            f"floating ≤ [{VALUE}]{cfg_init_stop.target_floating} %[/]{consecutive} · "
            f"median |Δs| / (s + {cfg_init_stop.speed_floor} m/yr)",
        )
        table.add_row(
            "Checks",
            f"every [{VALUE}]{cfg_init_stop.freq}[/] iterations · {patience} · "
            f"cap nbit_init = [{VALUE}]{cfg_unified.nbit_init}[/]",
        )

        self.console.print()
        self.console.print(
            Panel(
                table,
                title="[bold]IGM — initial training, stopped on its error against a direct solve[/bold]",
                border_style="cyan",
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )

    @contextlib.contextmanager
    def solving(self) -> Iterator[None]:
        """A spinner during the direct solve (none if the solver shows its own bar)."""
        progress = self.progress
        shows_bar = progress is not None and progress.enabled
        if not self.enabled or shows_bar:
            yield
            return
        with self._console.status(f"[{LABEL}]Direct solve…[/]", spinner="dots"):
            yield

    def reference(self, reference, ice: tf.Tensor, grounded: tf.Tensor) -> None:
        """One line on the direct solve."""
        if not self.enabled:
            return

        n_ice = int(tf.reduce_sum(tf.cast(ice, tf.int32)))
        n_grounded = int(tf.reduce_sum(tf.cast(grounded, tf.int32)))
        speed = tf.where(ice, reference.speed, tf.zeros_like(reference.speed))
        speed_max = float(tf.reduce_max(speed))
        n_floating = n_ice - n_grounded
        if reference.converged:
            mark = f"[{GOOD}]✓[/]"
            outcome = ""
        else:
            mark = f"[{WARN}]⚠[/]"
            outcome = f" [{WARN}](not converged: the Newton cap was reached)[/]"

        self.console.print(
            f"{mark} [{LABEL}]Direct solve:[/] {reference.iterations} Newton "
            f"iterations in {reference.seconds:.1f} s · {n_grounded:,} grounded and "
            f"{n_floating:,} floating ice cells · max surface speed {speed_max:,.0f} m/yr"
            f"{outcome}"
        )

    def row(
        self,
        iterations: int,
        error_grounded: float,
        error_floating: float,
        is_best: bool,
    ) -> None:
        """One check of the training."""
        if self._rows == 0:
            self.console.print(
                f"[dim]{'iterations':>12}   {'grounded':>8}   {'floating':>8}[/dim]"
            )
        self._rows += 1

        target_grounded, target_floating = self._targets
        if is_best:
            best = f" [{WARN}]★[/]"
        else:
            best = ""
        self.console.print(
            f"{iterations:>12,}   "
            f"{percent(error_grounded, target_grounded)}   "
            f"{percent(error_floating, target_floating)}{best}"
        )

    def summary(
        self,
        reason: str,
        iterations: int,
        nbit_init: int,
        error_grounded: float,
        error_floating: float,
        target_grounded: float,
        target_floating: float,
        reference_seconds: float,
        training_seconds: float,
        log: str,
    ) -> None:
        """How the initial training ended: a panel, or one line without the display."""
        style, title, phrase = SUMMARY[reason]

        if not self.enabled:
            print(
                f"init_stop: stopped after {iterations} of {nbit_init} iterations "
                f"({phrase}): grounded {error_grounded:.2f} % (target "
                f"{target_grounded}), floating {error_floating:.2f} % (target "
                f"{target_floating}); direct solve {reference_seconds:.1f} s, "
                f"training {training_seconds:.1f} s",
                flush=True,
            )
            return

        table = settings_table()
        table.add_row(
            "Iterations", f"[{VALUE}]{iterations:,}[/] of {nbit_init:,} (nbit_init)"
        )
        table.add_row(
            "Grounded",
            f"{percent(error_grounded, target_grounded)}  (target {target_grounded} %)",
        )
        table.add_row(
            "Floating",
            f"{percent(error_floating, target_floating)}  (target {target_floating} %)",
        )
        table.add_row(
            "Time",
            f"direct solve {reference_seconds:.1f} s · training {training_seconds:.1f} s",
        )
        if log:
            table.add_row("Log", log)

        self.console.print(
            Panel(
                table,
                title=f"[bold]{title}[/bold]",
                border_style=style,
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )
        self.console.print()
