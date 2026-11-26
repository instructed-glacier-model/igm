#!/usr/bin/env python3
# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from __future__ import annotations
import tensorflow as tf
from typing import Optional, List

from .progress_optimizer import ProgressOptimizer  # NEW
from rich.console import Console
from rich.progress import Progress
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TextColumn,
)

class _DAProgressOptimizer(ProgressOptimizer):
    """Rich progress bar with Total/Data/Reg columns + optional halt criteria columns."""

    def start(self, total_iterations: int, criterion_names=None) -> None:
        if not self.enabled:
            return


        tf.print("")
        self.console = Console(theme=self.theme)
        self.list_columns_crit = self._build_list_columns_crit(criterion_names or [])

        columns = self._build_columns(self.list_columns_crit)
        self.progress = Progress(*columns, console=self.console, expand=True)

        fields = {"J_total": "N/A", "J_data": "N/A", "J_reg": "N/A"}
        for f in self.list_columns_crit:
            fields[f.field_key] = "N/A"

        self.task = self.progress.add_task("progress", total=total_iterations, **fields)
        self.progress.start()

    def update(
        self,
        iter: int,
        total: float,
        data: float,
        reg: float,
        criterion_values=None,
        criterion_satisfied=None,
    ) -> None:
        if not self.enabled:
            return

        update_dict = {
            "completed": iter + 1,
            "J_total": f"{total:.4e}",
            "J_data":  f"{data:.4e}",
            "J_reg":   f"{reg:.4e}",
        }

        values = criterion_values or []
        satisfied = criterion_satisfied or []

        # keep the original green/red behavior for actual stopping criteria
        for i, f in enumerate(self.list_columns_crit):
            if i < len(values):
                val = values[i]
                if val != val:  # NaN
                    continue
                ok = satisfied[i] if i < len(satisfied) else False
                color = "green" if ok else "red"
                update_dict[f.field_key] = f"[{color}]{val:.2e}[/{color}]"

        self.progress.update(self.task, **update_dict)

    @staticmethod
    def _build_columns(fields) -> list:
        # local imports to keep module footprint small


        columns = [
            "🎯",
            BarColumn(bar_width=None, style="bar.incomplete", complete_style="bar.complete"),
            MofNCompleteColumn(),
            "[label]•[/]",
            TextColumn("[label]Total:[/] [value.cost]{task.fields[J_total]}"),
            "[label]•[/]",
            TextColumn("[label]Data:[/] [value.grad]{task.fields[J_data]}"),
            "[label]•[/]",
            TextColumn("[label]Reg:[/]  [value.grad]{task.fields[J_reg]}"),
        ]

        for f in fields:
            columns.extend(
                [
                    "[label]•[/]",
                    TextColumn(f"[label]{f.label}:[/] {{task.fields[{f.field_key}]}}"),
                ]
            )

        columns.extend(
            [
                "[label]•[/]",
                TextColumn("[label]Time:[/]"),
                TimeElapsedColumn(),
                "[label]•[/]",
                TextColumn("[label]ETA:[/]"),
                TimeRemainingColumn(),
            ]
        )
        return columns

