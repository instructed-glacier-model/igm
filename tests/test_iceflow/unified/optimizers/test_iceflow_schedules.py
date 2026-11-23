"""``nbit``, ``adam.lr`` and ``retrain_freq`` as constants or as schedules over model time."""

from types import SimpleNamespace

import pytest
import tensorflow as tf
from omegaconf import OmegaConf

from igm.processes.iceflow.unified.optimizers.interfaces import (
    InterfaceAdam,
    InterfaceCGNewton,
    InterfaceSSESOAP,
    Status,
)
from igm.processes.iceflow.unified.solver.solver import get_status
from igm.utils.schedule import parse_schedule

SCHEDULE = [[0.0, 20], [1.0, 5], [10.0, 0]]


class _RecordingOptimizer:
    """Stands in for an optimizer: records the parameters it is given."""

    def __init__(self):
        self.map = SimpleNamespace()
        self.calls = []

    def update_parameters(self, **kwargs):
        self.calls.append(kwargs)


def _cfg(nbit=5, lr=1.0e-3, retrain_freq=5):
    return OmegaConf.create(
        {
            "processes": {
                "iceflow": {
                    "unified": {
                        "nbit_init": 300,
                        "nbit_warmup": -1,
                        "nbit": nbit,
                        "retrain_freq": retrain_freq,
                        "adam": {"lr": lr, "lr_init": 2.0e-3},
                        "ss_esoap": {"lr": 1.0e-3, "lr_init": 3.0e-4},
                        "cg_newton": {"damping": 1.0e-6},
                    }
                }
            }
        }
    )


def test_interface_reads_the_nbit_schedule_at_the_model_time():
    cfg = _cfg(nbit=SCHEDULE)
    optimizer = _RecordingOptimizer()

    assert InterfaceSSESOAP.set_optimizer_params(cfg, Status.INIT, optimizer, None)
    assert optimizer.calls[-1] == {"iter_max": 300, "lr": 3.0e-4}

    assert InterfaceSSESOAP.set_optimizer_params(cfg, Status.DEFAULT, optimizer, 0.5)
    assert optimizer.calls[-1] == {"iter_max": 20, "lr": 1.0e-3}

    assert InterfaceSSESOAP.set_optimizer_params(cfg, Status.DEFAULT, optimizer, 2.0)
    assert optimizer.calls[-1] == {"iter_max": 5, "lr": 1.0e-3}

    # nbit 0 from t = 10: nothing to train
    assert not InterfaceSSESOAP.set_optimizer_params(
        cfg, Status.DEFAULT, optimizer, 12.0
    )

    n_calls = len(optimizer.calls)
    assert not InterfaceSSESOAP.set_optimizer_params(cfg, Status.IDLE, optimizer, 2.0)
    assert len(optimizer.calls) == n_calls


def test_newton_type_interface_reads_the_schedule_and_always_solves():
    optimizer = _RecordingOptimizer()
    cfg = _cfg(nbit=SCHEDULE)

    assert InterfaceCGNewton.set_optimizer_params(cfg, Status.DEFAULT, optimizer, 0.5)
    assert optimizer.calls[-1]["iter_max"] == 20

    # as before the schedule: the Newton-type interfaces return True even without iterations
    assert InterfaceCGNewton.set_optimizer_params(cfg, Status.DEFAULT, optimizer, 12.0)
    assert optimizer.calls[-1]["iter_max"] == 0


def test_the_model_time_may_be_a_tensor():
    optimizer = _RecordingOptimizer()
    cfg = _cfg(nbit=SCHEDULE)

    InterfaceSSESOAP.set_optimizer_params(
        cfg, Status.DEFAULT, optimizer, tf.Variable(1.5)
    )
    assert optimizer.calls[-1]["iter_max"] == 5


@pytest.mark.parametrize(
    "lr, t, expected",
    [
        (1.0e-3, 50.0, 1.0e-3),
        ([[0.0, 1.0e-3], [20.0, 2.0e-4]], 5.0, 1.0e-3),
        ([[0.0, 1.0e-3], [20.0, 2.0e-4]], 20.0, 2.0e-4),
    ],
)
def test_adam_reads_the_lr_schedule_of_the_retrains(lr, t, expected):
    cfg = _cfg(lr=lr)
    optimizer = _RecordingOptimizer()

    assert InterfaceAdam.set_optimizer_params(cfg, Status.DEFAULT, optimizer, t)
    assert optimizer.calls[-1]["lr"] == expected

    # the initial training keeps lr_init
    assert InterfaceAdam.set_optimizer_params(cfg, Status.INIT, optimizer, None)
    assert optimizer.calls[-1]["lr"] == 2.0e-3


def _state(cfg, it, t):
    """The state get_status reads, with retrain_freq parsed as at initialisation."""
    retrain_freq = parse_schedule(cfg.processes.iceflow.unified.retrain_freq)
    return SimpleNamespace(
        it=it, t=tf.Variable(t), iceflow=SimpleNamespace(retrain_freq=retrain_freq)
    )


def _retrain_steps(cfg, steps):
    """Steps (it, t) at which get_status asks for a retrain."""
    retrains = []
    for it, t in steps:
        if get_status(cfg, _state(cfg, it, t)) == Status.DEFAULT:
            retrains.append(it)
    return retrains


def test_retrain_freq_as_a_constant_retrains_every_n_steps():
    steps = [(it, 0.5 * it) for it in range(1, 21)]
    assert _retrain_steps(_cfg(retrain_freq=5), steps) == [5, 10, 15, 20]
    assert _retrain_steps(_cfg(retrain_freq=0), steps) == []


def test_retrain_freq_schedule_switches_at_the_model_time():
    # every step during the first 5 years (t = it / 2 < 5), then every 4 steps
    cfg = _cfg(retrain_freq=[[0.0, 1], [5.0, 4]])
    steps = [(it, 0.5 * it) for it in range(1, 21)]
    assert _retrain_steps(cfg, steps) == list(range(1, 10)) + [12, 16, 20]

    # retraining stops from t = 5 on
    cfg = _cfg(retrain_freq=[[0.0, 1], [5.0, 0]])
    assert _retrain_steps(cfg, steps) == list(range(1, 10))


def test_the_initial_training_ignores_retrain_freq():
    cfg = _cfg(retrain_freq=[[0.0, 1], [5.0, 4]])
    assert get_status(cfg, _state(cfg, 0, 0.0), init=True) == Status.INIT
