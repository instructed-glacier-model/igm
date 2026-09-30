"""``nbit`` as an integer or as a schedule over model time."""

from types import SimpleNamespace

import pytest
import tensorflow as tf
from omegaconf import OmegaConf

from igm.processes.iceflow.unified.optimizers.interfaces import (
    InterfaceCGNewton,
    InterfaceSSESOAP,
    Status,
    check_nbit,
    nbit_at,
)

SCHEDULE = [[0.0, 20], [1.0, 5], [10.0, 0]]


class _RecordingOptimizer:
    """Stands in for an optimizer: records the parameters it is given."""

    def __init__(self):
        self.map = SimpleNamespace()
        self.calls = []

    def update_parameters(self, **kwargs):
        self.calls.append(kwargs)


def _cfg(nbit):
    return OmegaConf.create(
        {
            "processes": {
                "iceflow": {
                    "unified": {
                        "nbit_init": 300,
                        "nbit": nbit,
                        "ss_esoap": {"lr": 1.0e-3, "lr_init": 3.0e-4},
                        "cg_newton": {"damping": 1.0e-6},
                    }
                }
            }
        }
    )


@pytest.mark.parametrize("nbit", [5, 0, -1, 2.5, "5"])
def test_anything_but_a_schedule_is_used_as_before(nbit):
    check_nbit(nbit)
    assert nbit_at(nbit) == nbit
    assert nbit_at(nbit, 123.0) == nbit


@pytest.mark.parametrize(
    "t, expected",
    [
        (None, 20),
        (-3.0, 20),
        (0.0, 20),
        (0.999, 20),
        (1.0, 5),
        (9.5, 5),
        (10.0, 0),
        (1.0e6, 0),
    ],
)
def test_schedule_is_piecewise_constant_from_each_start(t, expected):
    assert nbit_at(SCHEDULE, t) == expected
    assert nbit_at(OmegaConf.create(SCHEDULE), t) == expected


@pytest.mark.parametrize("nbit", [0, 5, SCHEDULE, [[2000.0, 3]]])
def test_valid_nbit_passes_the_check(nbit):
    check_nbit(nbit)
    check_nbit(OmegaConf.create({"nbit": nbit}).nbit)


@pytest.mark.parametrize(
    "nbit",
    [
        [],
        [[1.0, 5], [0.0, 20]],
        [[0.0, 5], [0.0, 20]],
        [[0.0, -1]],
        [[0.0, 2.5]],
        [[0.0]],
        [[0.0, 1, 2]],
        [["a", 1]],
    ],
)
def test_invalid_nbit_is_rejected(nbit):
    with pytest.raises(ValueError):
        check_nbit(nbit)


def test_interface_reads_the_schedule_at_the_model_time():
    cfg = _cfg(SCHEDULE)
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
    cfg = _cfg(SCHEDULE)

    assert InterfaceCGNewton.set_optimizer_params(cfg, Status.DEFAULT, optimizer, 0.5)
    assert optimizer.calls[-1]["iter_max"] == 20

    # as before the schedule: the Newton-type interfaces return True even without iterations
    assert InterfaceCGNewton.set_optimizer_params(cfg, Status.DEFAULT, optimizer, 12.0)
    assert optimizer.calls[-1]["iter_max"] == 0


def test_the_model_time_may_be_a_tensor():
    assert nbit_at(SCHEDULE, tf.Variable(1.5)) == 5
    assert nbit_at(SCHEDULE, tf.constant(0.5, tf.float64)) == 20
