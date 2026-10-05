from types import SimpleNamespace

import tensorflow as tf
from omegaconf import OmegaConf

from igm.processes.iceflow.unified.optimizers.interfaces import (
    InterfaceOptimizers,
)
from igm.processes.iceflow.unified.optimizers.interfaces.interface import (
    Status,
    nbit_at,
)
from igm.processes.iceflow.unified.optimizers.interfaces.sequential import (
    InterfaceSequential,
)


class _StageInterface:
    @staticmethod
    def set_optimizer_params(cfg, status, optimizer, t=None):
        unified = cfg.processes.iceflow.unified
        if status == Status.INIT:
            iterations = unified.nbit_init
        else:
            iterations = nbit_at(unified.nbit, t)
        optimizer.iter_max.assign(iterations)
        return iterations > 0


class _SequentialOptimizer:
    def __init__(self, stages):
        self.optimizers = stages
        self.iter_max = tf.Variable(0, dtype=tf.int32)

    def _compute_iter_max(self):
        return sum(int(stage.iter_max) for stage in self.optimizers)


def test_sequential_interface_refreshes_outer_iteration_budget(monkeypatch):
    monkeypatch.setitem(InterfaceOptimizers, "test_stage", _StageInterface)
    cfg = OmegaConf.create(
        {
            "processes": {
                "iceflow": {
                    "unified": {
                        "nbit_init": 0,
                        "nbit": 0,
                        "sequential": {
                            "stages": [
                                {
                                    "optimizer": "test_stage",
                                    "nbit_init": 10_000,
                                    "nbit": 0,
                                },
                                {
                                    "optimizer": "test_stage",
                                    "nbit_init": 10,
                                    "nbit": 0,
                                },
                            ]
                        },
                    }
                }
            }
        }
    )
    optimizer = _SequentialOptimizer(
        [
            SimpleNamespace(iter_max=tf.Variable(0, dtype=tf.int32)),
            SimpleNamespace(iter_max=tf.Variable(0, dtype=tf.int32)),
        ]
    )

    should_run = InterfaceSequential.set_optimizer_params(cfg, Status.INIT, optimizer)

    assert should_run
    assert int(optimizer.iter_max) == 10_010


def test_sequential_interface_resolves_each_stage_schedule(monkeypatch):
    monkeypatch.setitem(InterfaceOptimizers, "test_stage", _StageInterface)
    cfg = OmegaConf.create(
        {
            "processes": {
                "iceflow": {
                    "unified": {
                        "nbit_init": 0,
                        "nbit": 1,
                        "sequential": {
                            "stages": [
                                {
                                    "optimizer": "test_stage",
                                    "nbit": [[0.0, 20], [1.0, 5]],
                                },
                                {"optimizer": "test_stage"},
                            ]
                        },
                    }
                }
            }
        }
    )
    optimizer = _SequentialOptimizer(
        [
            SimpleNamespace(iter_max=tf.Variable(0, dtype=tf.int32)),
            SimpleNamespace(iter_max=tf.Variable(0, dtype=tf.int32)),
        ]
    )

    for t, expected in ((0.5, 21), (1.0, 6), (7.0, 6)):
        should_run = InterfaceSequential.set_optimizer_params(
            cfg, Status.DEFAULT, optimizer, t
        )
        assert should_run
        assert int(optimizer.iter_max) == expected
