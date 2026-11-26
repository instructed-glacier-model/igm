#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import math
import tensorflow as tf
from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import List, Optional
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


@dataclass(frozen=True)
class ColumnCrit:
    label: str
    field_key: str


class ProgressOptimizer:

    def __init__(self, enabled: bool = True, freq: int = 1):
        self.enabled = enabled
        self.freq = freq

        self.console: Optional[Console] = None
        self.progress: Optional[Progress] = None
        self.task = None

        self.list_columns_crit: List[ColumnCrit] = []

        self.theme = Theme(
            {
                "label": "bold #e5e7eb",
                "value.cost": "#f59e0b",
                "value.grad": "#06b6d4",
                "bar.incomplete": "grey35",
                "bar.complete": "#22c55e",
            }
        )

    def start(
        self, total_iterations: int, criterion_names: Optional[List[str]] = None
    ) -> None:
        if not self.enabled:
            return

        tf.print("")

        self.console = Console(theme=self.theme)
        self.list_columns_crit = self._build_list_columns_crit(criterion_names or [])

        columns = self._build_columns(self.list_columns_crit)

        self.progress = Progress(*columns, console=self.console, expand=True)

        fields = {"cost": "N/A"}
        for f in self.list_columns_crit:
            fields[f.field_key] = "N/A"

        self.task = self.progress.add_task("progress", total=total_iterations, **fields)
        self.progress.start()

    def should_update(self, iter: tf.Tensor) -> tf.Tensor:
        if not self.enabled:
            return tf.constant(False)
        return tf.equal(tf.math.mod(iter, self.freq), 0)

    def update(
        self,
        iter: int,
        cost: float,
        criterion_values: Optional[List[float]] = None,
        criterion_satisfied: Optional[List[bool]] = None,
    ) -> None:
        if not self.enabled:
            return

        update_dict = {
            "completed": iter + 1,
            "cost": f"{cost:.4e}",
        }

        values = criterion_values or []
        satisfied = criterion_satisfied or []

        for i, f in enumerate(self.list_columns_crit):
            if i < len(values):
                val = values[i]
                if math.isnan(val):
                    continue
                ok = satisfied[i] if i < len(satisfied) else False
                color = "green" if ok else "red"
                update_dict[f.field_key] = f"[{color}]{val:.2e}[/{color}]"

        self.progress.update(self.task, **update_dict)

    def stop(self, status) -> None:
        if not self.enabled:
            return

        self.progress.stop()

        status_name = getattr(status, "name", status).lower()

        messages = {
            "success": "✅ [bold green]Optimization converged![/bold green] Stopping criterion satisfied.",
            "failure": "❌ [bold red]Optimization failed![/bold red] Numerical instability detected.",
            "completed": "🏁 [bold blue]Optimization completed![/bold blue] Maximum iterations reached.",
        }

        self.console.print(messages.get(status_name, messages["completed"]))
        self.console.print()

    @staticmethod
    def _build_list_columns_crit(names: List[str]) -> List[ColumnCrit]:

        if not names:
            return []

        counts = Counter(names)
        seen = defaultdict(int)
        fields: List[ColumnCrit] = []

        for name in names:
            if counts[name] > 1:
                seen[name] += 1
                label = f"{name}_{seen[name]}"
                key = f"crit_{label}"
            else:
                label = name
                key = f"crit_{name}"
            fields.append(ColumnCrit(label=label, field_key=key))

        return fields

    @staticmethod
    def _build_columns(fields: List[ColumnCrit]) -> List:
        columns = [
            "🎯",
            BarColumn(
                bar_width=None, style="bar.incomplete", complete_style="bar.complete"
            ),
            MofNCompleteColumn(),
            "[label]•[/]",
            TextColumn("[label]Cost:[/] [value.cost]{task.fields[cost]}"),
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
