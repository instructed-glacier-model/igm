#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numbers
import tensorflow as tf
from omegaconf import DictConfig, ListConfig
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Callable, Dict, Optional

from ...mappings import Mapping
from .. import Optimizer


class Status(Enum):
    INIT = auto()
    WARM_UP = auto()
    DEFAULT = auto()
    IDLE = auto()


def is_schedule(nbit: Any) -> bool:
    """True for an nbit schedule over model time, ``[[t_0, n_0], [t_1, n_1], ...]``."""
    return isinstance(nbit, (list, tuple, ListConfig))


def is_count(value: Any) -> bool:
    """True for a non-negative integer (booleans excluded)."""
    return (
        isinstance(value, numbers.Integral)
        and not isinstance(value, bool)
        and value >= 0
    )


def check_nbit(nbit: Any) -> None:
    """Raise a ValueError unless an nbit schedule is valid.

    A schedule ``[[t_0, n_0], [t_1, n_1], ...]`` has strictly increasing ``t_k`` and
    non-negative integers ``n_k``. Any other nbit is used as it is, as before schedules.
    """
    if not is_schedule(nbit):
        return

    if len(nbit) == 0:
        raise ValueError("❌ An nbit schedule [[t_0, n_0], [t_1, n_1], ...] is empty.")

    times = []
    for entry in nbit:
        if not is_schedule(entry) or len(entry) != 2:
            raise ValueError(
                f"❌ Each nbit schedule entry must be a pair [t_start, n], got {entry!r}."
            )
        t_start, n = entry
        if not isinstance(t_start, numbers.Real) or isinstance(t_start, bool):
            raise ValueError(
                f"❌ nbit schedule: t_start must be a number, got {t_start!r}."
            )
        if not is_count(n):
            raise ValueError(
                f"❌ nbit schedule: n must be a non-negative integer, got {n!r}."
            )
        times.append(float(t_start))

    for t_previous, t_next in zip(times, times[1:]):
        if t_next <= t_previous:
            raise ValueError(
                f"❌ nbit schedule: t_start must strictly increase, got {times}."
            )


def nbit_at(nbit: Any, t: Any = None) -> Any:
    """Iterations of one retrain at model time ``t`` (yr, a number or a tensor).

    A schedule ``[[t_0, n_0], [t_1, n_1], ...]`` is piecewise constant: ``n_k`` applies
    from ``t_k`` on, and ``n_0`` also before ``t_0`` and when ``t`` is None. Any other nbit
    is returned as it is, and ``t`` is then not read.
    """
    if not is_schedule(nbit):
        return nbit

    n = int(nbit[0][1])
    if t is None:
        return n

    t = float(t)
    for t_start, n_k in nbit:
        if t >= t_start:
            n = int(n_k)
        else:
            break
    return n


class InterfaceOptimizer(ABC):

    @staticmethod
    @abstractmethod
    def get_optimizer_args(
        cfg: DictConfig,
        cost_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
        map: Mapping,
    ) -> Dict[str, Any]:
        raise NotImplementedError(
            "❌ The get_optimizer_args static method is not implemented."
        )

    @staticmethod
    @abstractmethod
    def set_optimizer_params(
        cfg: DictConfig,
        status: Status,
        optimizer: Optimizer,
        t: Optional[float] = None,
    ) -> bool:
        """Set the optimizer's iterations and settings for this solve; return whether to solve.

        ``t`` is the model time of the step (None at initialisation), at which a scheduled
        ``nbit`` is read (``nbit_at``); it is only read for a schedule.
        """
        raise NotImplementedError(
            "❌ The set_optimizer_params static method is not implemented."
        )
