"""``time.step_max`` and ``time.cfl`` as constants or as schedules over model time."""

from types import SimpleNamespace

import pytest
import tensorflow as tf
from omegaconf import OmegaConf

from igm.processes.time import time as time_module
from igm.processes.time.time import compute_dt_from_cfl


def _cfg(step_max, cfl):
    return OmegaConf.create(
        {
            "processes": {
                "time": {
                    "start": 0.0,
                    "end": 100.0,
                    "save": 100.0,
                    "cfl": cfl,
                    "step_max": step_max,
                    "cfl_percentile": 100.0,
                }
            }
        }
    )


def _time_steps(cfg, speed, n):
    """Time steps of the first n updates, ice moving at `speed` (m/yr) on 100 m cells."""
    state = SimpleNamespace(
        ubar=tf.fill((4, 4), speed),
        vbar=tf.zeros((4, 4)),
        dx=tf.constant(100.0),
    )
    time_module.initialize(cfg, state)

    # the first update saves the initial state, with dt = 0
    state.it = 0
    time_module.update(cfg, state)
    assert float(state.dt) == 0.0

    steps = []
    for it in range(1, n + 1):
        state.it = it
        time_module.update(cfg, state)
        steps.append(float(state.dt))
    return steps


def test_constants_are_used_as_before():
    assert _time_steps(_cfg(step_max=0.5, cfl=0.3), speed=0.0, n=3) == [0.5] * 3
    # CFL: 0.25 * 100 m / 100 m/yr
    assert _time_steps(_cfg(step_max=1.0, cfl=0.25), 100.0, 3) == [0.25] * 3
    # no CFL condition
    assert _time_steps(_cfg(step_max=0.5, cfl=0.0), 100.0, 3) == [0.5] * 3


def test_step_max_schedule_switches_at_the_model_time():
    step_max = [[0.0, 0.5], [2.0, 1.0]]
    expected = [0.5] * 4 + [1.0] * 2
    assert _time_steps(_cfg(step_max, cfl=0.3), 0.0, 6) == expected
    assert _time_steps(_cfg(step_max, cfl=0.0), 0.0, 6) == expected


def test_cfl_schedule_switches_at_the_model_time():
    cfl = [[0.0, 0.25], [1.0, 0.5]]
    expected = [0.25] * 4 + [0.5] * 2
    assert _time_steps(_cfg(step_max=10.0, cfl=cfl), 100.0, 6) == expected


def test_a_schedule_is_traced_once():
    step_max = [[0.0, 0.5], [2.0, 1.0], [4.0, 0.25]]
    _time_steps(_cfg(step_max, cfl=[[0.0, 0.3]]), 0.0, 1)
    traces = compute_dt_from_cfl.experimental_get_tracing_count()

    # a new run, with equal schedules parsed anew: no new trace either
    steps = _time_steps(_cfg(step_max, cfl=[[0.0, 0.3]]), 0.0, 12)
    assert set(steps) == {0.25, 0.5, 1.0}
    assert compute_dt_from_cfl.experimental_get_tracing_count() == traces


@pytest.mark.parametrize(
    "step_max, cfl",
    [
        ([[0.0, 0.0]], 0.3),
        ([[0.0, -1.0]], 0.3),
        ([[1.0, 0.5], [0.0, 1.0]], 0.3),
        (1.0, [[0.0, 0.3], [5.0, 0.0]]),
    ],
)
def test_invalid_schedules_are_rejected(step_max, cfl):
    with pytest.raises(ValueError, match="time"):
        time_module.initialize(_cfg(step_max, cfl), SimpleNamespace())
