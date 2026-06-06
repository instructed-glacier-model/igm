#!/usr/bin/env python3
# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Tuple

import yaml


@dataclass
class History:
    """Per-epoch training/validation history."""

    train_total: List[float] = field(default_factory=list)
    val_total:   List[float] = field(default_factory=list)
    train_data:  List[float] = field(default_factory=list)
    val_data:    List[float] = field(default_factory=list)
    train_phys:  List[float] = field(default_factory=list)
    val_phys:    List[float] = field(default_factory=list)
    lambda_phys: List[float] = field(default_factory=list)

    def append_epoch(
        self,
        *,
        train_total: float, val_total: float,
        train_data: float,  val_data: float,
        train_phys: float,  val_phys: float,
        lambda_phys: float,
    ) -> None:
        self.train_total.append(float(train_total))
        self.val_total.append(float(val_total))
        self.train_data.append(float(train_data))
        self.val_data.append(float(val_data))
        self.train_phys.append(float(train_phys))
        self.val_phys.append(float(val_phys))
        self.lambda_phys.append(float(lambda_phys))


def _history_path(out_dir: Path) -> Path:
    return out_dir / "history.yaml"


def load_history_yaml(out_dir: Path) -> Tuple[int, History]:
    path = _history_path(out_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"resume=True but missing history file: {path}. "
            "Expected history.yaml alongside checkpoints."
        )

    data = yaml.safe_load(path.read_text()) or {}
    epoch = int(data.get("epoch", 0))

    def _lst(key: str) -> List[float]:
        v = data.get(key, []) or []
        return [float(x) for x in v]

    history = History(
        train_total=_lst("train_total"),
        val_total=_lst("val_total"),
        train_data=_lst("train_data"),
        val_data=_lst("val_data"),
        train_phys=_lst("train_phys"),
        val_phys=_lst("val_phys"),
        lambda_phys=_lst("lambda_phys"),
    )
    return epoch, history


def save_history_yaml(out_dir: Path, epoch: int, history: History) -> None:
    payload = {"epoch": int(epoch), **asdict(history)}

    path = _history_path(out_dir)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(yaml.safe_dump(payload, sort_keys=False))
    tmp.replace(path)
