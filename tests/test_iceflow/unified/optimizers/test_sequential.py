from types import SimpleNamespace

import tensorflow as tf
from omegaconf import OmegaConf

from igm.processes.iceflow.unified.optimizers.interfaces import (
    InterfaceOptimizers,
)
from igm.processes.iceflow.unified.optimizers.interfaces.interface import Status
from igm.processes.iceflow.unified.optimizers.interfaces.sequential import (
    InterfaceSequential,
)


class _StageInterface:
    @staticmethod
    def set_optimizer_params(cfg, status, optimizer):
        unified = cfg.processes.iceflow.unified
        iterations = unified.nbit_init if status == Status.INIT else unified.nbit
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
