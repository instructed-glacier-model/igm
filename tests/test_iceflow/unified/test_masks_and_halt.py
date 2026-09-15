from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf
from omegaconf import OmegaConf

from igm.processes.iceflow.unified import utils
from igm.processes.iceflow.unified.halt import InterfaceHalt
from igm.processes.iceflow.unified.halt.criteria.rel_initial import (
    CriterionRelInitial,
)
from igm.processes.iceflow.unified.halt.criteria.abs_change import (
    CriterionAbsChange,
)
from igm.processes.iceflow.unified.halt.metrics import MetricGradUNorm, MetricU
from igm.processes.iceflow.utils.velocities import (
    compute_cell_ice_mask,
    compute_node_ice_mask,
)


def test_ice_masks_exclude_nodes_without_active_cell_support():
    thk = tf.constant(
        [[[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]
    )

    np.testing.assert_array_equal(
        compute_cell_ice_mask(thk).numpy(),
        np.array([[[True, False], [False, False]]]),
    )
    np.testing.assert_array_equal(
        compute_node_ice_mask(thk).numpy(),
        np.array(
            [[[True, True, False], [True, True, False], [False, False, False]]]
        ),
    )


def test_unified_cost_excludes_cells_with_an_ice_free_corner(monkeypatch):
    class CellSumComponent:
        name = "cell_sum"

        def cost(self, U, V, fieldin, discr_h, discr_v):
            del fieldin, discr_h, discr_v
            u = U[:, 0]
            v = V[:, 0]
            return (
                u[:, :-1, :-1]
                + u[:, :-1, 1:]
                + u[:, 1:, :-1]
                + u[:, 1:, 1:]
                + v[:, :-1, :-1]
                + v[:, :-1, 1:]
                + v[:, 1:, :-1]
                + v[:, 1:, 1:]
            )

    monkeypatch.setattr(
        utils, "get_energy_components", lambda cfg: [CellSumComponent()]
    )
    cfg = OmegaConf.create(
        {"processes": {"iceflow": {"unified": {"inputs": ["thk"]}}}}
    )
    state = SimpleNamespace(iceflow=SimpleNamespace(discr_h=None, discr_v=None))
    cost_fn = utils.get_cost_fn(cfg, state)
    thk = tf.constant(
        [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0]]]
    )
    inputs = thk[..., tf.newaxis]
    U = tf.Variable(tf.ones((1, 1, 3, 3)))
    V = tf.Variable(tf.ones((1, 1, 3, 3)))

    with tf.GradientTape() as tape:
        cost = cost_fn(U, V, inputs)
    grad_U, grad_V = tape.gradient(cost, [U, V])

    assert float(cost) == 6.0
    expected = np.array([[[[1, 2, 1], [2, 3, 1], [1, 1, 0]]]]) / 4.0
    np.testing.assert_allclose(grad_U.numpy(), expected)
    np.testing.assert_allclose(grad_V.numpy(), expected)


def test_rel_initial_compares_against_first_metric_norm_and_resets():
    criterion = CriterionRelInitial(
        metric=MetricGradUNorm(),
        dtype="float64",
        tol=0.1,
        ord="l2",
    )

    satisfied, ratio = criterion.check(SimpleNamespace(grad_u_norm=tf.constant(10.0)))
    assert not bool(satisfied)
    assert float(ratio) == 1.0

    satisfied, ratio = criterion.check(SimpleNamespace(grad_u_norm=tf.constant(0.5)))
    assert bool(satisfied)
    assert float(ratio) == 0.05

    criterion.reset()
    satisfied, ratio = criterion.check(SimpleNamespace(grad_u_norm=tf.constant(2.0)))
    assert not bool(satisfied)
    assert float(ratio) == 1.0


def test_rel_initial_is_available_from_halt_configuration():
    cfg = OmegaConf.create(
        {
            "processes": {
                "iceflow": {
                    "numerics": {"precision": "double"},
                    "unified": {
                        "halt": {
                            "freq": 1,
                            "success": [
                                {
                                    "criterion": "rel_initial",
                                    "metric": "grad_u_norm",
                                }
                            ],
                            "failure": [],
                            "criteria": {
                                "rel_initial": {"tol": 0.1, "ord": "l2"}
                            },
                            "metrics": {"grad_u_norm": {}},
                        }
                    },
                }
            }
        }
    )

    halt_args = InterfaceHalt.get_halt_args(cfg)

    criterion = halt_args["crit_success"][0]
    assert isinstance(criterion, CriterionRelInitial)
    assert criterion.tol == 0.1


def test_abs_change_uses_vector_magnitude_and_consecutive_checks():
    criterion = CriterionAbsChange(
        metric=MetricU(),
        dtype="float64",
        tol=5.0,
        reduction="max",
        consecutive=2,
    )

    @tf.function
    def check(U, V):
        return criterion.check(SimpleNamespace(u=[U, V]))

    U = tf.zeros([1, 2], tf.float64)
    V = tf.zeros_like(U)
    satisfied, change = check(U, V)
    assert not bool(satisfied)
    assert np.isnan(float(change))

    # A large update breaks the streak; two subsequent small updates halt.
    for u, v, expected_change, expected_stop in (
        (3.0, 4.0, 5.0, False),
        (9.0, 12.0, 10.0, False),
        (12.0, 16.0, 5.0, False),
        (15.0, 20.0, 5.0, True),
    ):
        U = tf.constant([[u, 0.0]], tf.float64)
        V = tf.constant([[v, 0.0]], tf.float64)
        satisfied, change = check(U, V)
        assert bool(satisfied) == expected_stop
        assert float(change) == expected_change

    criterion.reset()
    satisfied, change = check(U, V)
    assert not bool(satisfied)
    assert np.isnan(float(change))
    satisfied, change = check(U + [[3.0, 0.0]], V + [[4.0, 0.0]])
    assert not bool(satisfied)
    assert float(change) == 5.0


def test_abs_change_rmse_is_spatial_vector_rmse():
    criterion = CriterionAbsChange(
        metric=MetricU(),
        dtype="float64",
        tol=4.0,
        reduction="rmse",
    )
    state = SimpleNamespace(
        u=[tf.zeros([1, 2], tf.float64), tf.zeros([1, 2], tf.float64)]
    )
    criterion.check(state)

    state.u = [
        tf.constant([[3.0, 0.0]], tf.float64),
        tf.constant([[4.0, 0.0]], tf.float64),
    ]
    satisfied, change = criterion.check(state)

    assert bool(satisfied)
    assert float(change) == pytest.approx(np.sqrt(25.0 / 2.0))


def test_abs_change_can_be_enabled_from_halt_configuration():
    config_path = (
        Path(__file__).resolve().parents[3] / "igm/conf/processes/iceflow.yaml"
    )
    cfg = OmegaConf.create({"processes": OmegaConf.load(config_path)})
    cfg_halt = cfg.processes.iceflow.unified.halt
    assert InterfaceHalt.get_halt_args(cfg)["crit_success"] == []

    cfg_halt.success = [
        {
            "criterion": "abs_change",
            "metric": "u",
            "abs_change": {"reduction": "rmse", "consecutive": 3},
        }
    ]
    halt_args = InterfaceHalt.get_halt_args(cfg)

    assert len(halt_args["crit_success"]) == 1
    criterion = halt_args["crit_success"][0]
    assert isinstance(criterion, CriterionAbsChange)
    assert isinstance(criterion.metric, MetricU)
    assert float(criterion.tol) == 1.0
    assert criterion.reduction == "rmse"
    assert int(criterion.consecutive) == 3

    cfg_halt.criteria.abs_change.tol = 2.0
    halt_args = InterfaceHalt.get_halt_args(cfg)
    assert float(halt_args["crit_success"][0].tol) == 2.0

    cfg_halt.success = []
    halt_args = InterfaceHalt.get_halt_args(cfg)
    assert halt_args["crit_success"] == []
