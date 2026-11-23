#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Parameters that change over model time.

A parameter is either a constant, used as it is, or a piecewise-constant schedule
``[[t_0, v_0], [t_1, v_1], ...]`` over model time (yr, as ``time.start``): ``v_k`` applies
from ``t_k`` on, and ``v_0`` also before ``t_0``.
"""

import numbers
from dataclasses import dataclass
from typing import Any, Callable, Tuple

import tensorflow as tf
from omegaconf import ListConfig


@dataclass(frozen=True, slots=True)
class Schedule:
    """A parsed schedule, traced once by a ``tf.function`` that takes it as an argument.

    The slots make it a literal for ``tf.function``, compared by value; an object that
    can be weakly referenced would be compared by identity instead.
    """

    starts: Tuple[float, ...]
    values: Tuple[Any, ...]

    def at(self, t: Any = None) -> Any:
        """Value at model time ``t``, as in ``value_at``."""
        if t is None:
            return self.values[0]

        if tf.executing_eagerly():
            t = float(t)
            k = sum(t >= t_start for t_start in self.starts[1:])
            return self.values[k]

        k = tf.reduce_sum(tf.cast(t >= tf.constant(self.starts[1:], t.dtype), tf.int32))
        return tf.gather(tf.constant(self.values, t.dtype), k)


def is_schedule(param: Any) -> bool:
    """True for a schedule ``[[t_0, v_0], [t_1, v_1], ...]``, parsed or not."""
    return isinstance(param, (list, tuple, ListConfig, Schedule))


def parse_schedule(param: Any) -> Any:
    """A schedule as a ``Schedule``; a constant or a ``Schedule`` is returned as it is."""
    if not is_schedule(param) or isinstance(param, Schedule):
        return param

    return Schedule(
        starts=tuple(float(entry[0]) for entry in param),
        values=tuple(entry[1] for entry in param),
    )


def value_at(param: Any, t: Any = None) -> Any:
    """Value of a parameter at model time ``t`` (yr).

    A constant is returned as it is, and ``t`` is not read. A schedule gives ``v_0`` when
    ``t`` is None. Otherwise, eagerly, ``t`` (a number, or a tensor such as ``state.t``) is
    read on the host and ``v_k`` is returned as it is. Inside a ``tf.function``, where ``t``
    is a tensor, the schedule is evaluated in the graph as a tensor of ``t``'s dtype: traced
    once, its value changes without retracing and without a host synchronisation.
    """
    if not is_schedule(param):
        return param
    return parse_schedule(param).at(t)


def is_count(value: Any) -> bool:
    """True for a non-negative integer (booleans excluded)."""
    return (
        isinstance(value, numbers.Integral)
        and not isinstance(value, bool)
        and value >= 0
    )


def is_positive(value: Any) -> bool:
    """True for a positive number (booleans excluded)."""
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and value > 0


def check_schedule(
    param: Any, name: str, is_valid: Callable[[Any], bool], expected: str
) -> None:
    """Raise a ValueError unless a schedule is well formed; a constant is not checked.

    A schedule is a non-empty list of pairs ``[t_k, v_k]`` with strictly increasing numbers
    ``t_k`` and values ``v_k`` accepted by ``is_valid``, which ``expected`` describes.
    """
    if not is_schedule(param):
        return

    if len(param) == 0:
        raise ValueError(
            f"❌ {name}: the schedule [[t_0, v_0], [t_1, v_1], ...] is empty."
        )

    starts = []
    for entry in param:
        if not is_schedule(entry) or len(entry) != 2:
            raise ValueError(
                f"❌ {name}: each schedule entry must be a pair [t_start, value], got {entry!r}."
            )
        t_start, value = entry
        if not isinstance(t_start, numbers.Real) or isinstance(t_start, bool):
            raise ValueError(f"❌ {name}: t_start must be a number, got {t_start!r}.")
        if not is_valid(value):
            raise ValueError(f"❌ {name}: values must be {expected}, got {value!r}.")
        starts.append(float(t_start))

    for t_previous, t_next in zip(starts, starts[1:]):
        if t_next <= t_previous:
            raise ValueError(
                f"❌ {name}: t_start must strictly increase, got {starts}."
            )
