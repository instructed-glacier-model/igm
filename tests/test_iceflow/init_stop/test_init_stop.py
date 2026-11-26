"""Stopping the initial training on its error against a direct solve (``init_stop``)."""

import csv
import os

import numpy as np
import pytest
import tensorflow as tf

import igm
import igm.processes.iceflow.init_stop.init_stop as init_stop_module
import igm.processes.iceflow.init_stop.reference as reference_module
from igm.common.runner.configuration.loader import load_yaml_recursive
from igm.processes.iceflow.evaluator.evaluator import (
    get_evaluator_inputs_from_state,
)
from igm.processes.iceflow.init_stop import (
    InitStopCriterion,
    ReferenceSolve,
    StopReason,
)
from igm.processes.iceflow.optimizers.halt import Halt
from igm.processes.iceflow.optimizers.halt.step_state import StepState

FREQ = 10
SHAPE = (12, 9)


class _FakeMapping:
    """Holds a velocity whose surface speed (second vertical level) is set directly."""

    def __init__(self, speed: np.ndarray):
        self.U = tf.Variable(np.zeros((1, 2, *speed.shape), np.float32))
        self.V = tf.Variable(np.zeros((1, 2, *speed.shape), np.float32))
        self.set_speed(speed)

    def set_speed(self, speed: np.ndarray) -> None:
        U = np.zeros((1, 2, *speed.shape), np.float32)
        U[0, 1] = speed
        self.U.assign(U)

    def get_UV(self, inputs):
        return self.U, self.V

    def get_theta(self):
        return [self.U, self.V]


def _fields(seed: int = 0):
    rng = np.random.default_rng(seed)
    speed_ref = rng.uniform(0.0, 500.0, SHAPE).astype(np.float32)
    grounded = np.zeros(SHAPE, bool)
    grounded[:, :6] = True
    return speed_ref, grounded, ~grounded


def _criterion(
    mapping,
    speed_ref,
    grounded,
    floating,
    patience=0,
    targets=(1.0, 2.5),
    consecutive=1,
):
    return InitStopCriterion(
        mapping=mapping,
        inputs=tf.zeros((1, *SHAPE, 1)),
        V_s=tf.constant([0.0, 1.0]),
        speed_ref=tf.constant(speed_ref),
        grounded=tf.constant(grounded),
        floating=tf.constant(floating),
        target_grounded=targets[0],
        target_floating=targets[1],
        consecutive=consecutive,
        speed_floor=10.0,
        freq=FREQ,
        patience=patience,
        min_gain=0.05,
        dtype="float32",
    )


def _after(iterations: int) -> StepState:
    """The step state of the check made once ``iterations`` updates are applied."""
    zero = tf.constant(0.0)
    return StepState(tf.constant(iterations - 1), [zero, zero], zero, zero, zero, zero)


def _lower_median(values: np.ndarray) -> float:
    ordered = np.sort(values)
    return float(ordered[(ordered.size - 1) // 2])


def test_errors_are_floored_medians_per_region():
    speed_ref, grounded, floating = _fields()
    rng = np.random.default_rng(1)
    speed = speed_ref * (1.0 + rng.normal(0.0, 0.05, SHAPE)).astype(np.float32)
    criterion = _criterion(_FakeMapping(speed), speed_ref, grounded, floating)

    error_grounded, error_floating = criterion.errors()

    relative = np.abs(speed - speed_ref) / (speed_ref + 10.0)
    assert float(error_grounded) == pytest.approx(
        100.0 * _lower_median(relative[grounded]), rel=1e-5
    )
    assert float(error_floating) == pytest.approx(
        100.0 * _lower_median(relative[floating]), rel=1e-5
    )


def test_stops_when_both_targets_are_met():
    speed_ref, grounded, floating = _fields()
    criterion = _criterion(
        _FakeMapping(speed_ref * 1.001), speed_ref, grounded, floating
    )
    criterion.arm()

    is_satisfied, ratio = criterion.check(_after(FREQ))

    assert bool(is_satisfied)
    assert float(ratio) <= 1.0
    assert criterion.reason == StopReason.TARGETS
    assert [row[0] for row in criterion.history] == [FREQ]


def test_checks_only_when_active_and_due():
    speed_ref, grounded, floating = _fields()
    criterion = _criterion(_FakeMapping(speed_ref), speed_ref, grounded, floating)

    is_satisfied, ratio = criterion.check(_after(FREQ))  # not armed
    assert not bool(is_satisfied) and np.isnan(float(ratio))

    criterion.arm()
    is_satisfied, ratio = criterion.check(_after(FREQ + 3))  # not a check iteration
    assert not bool(is_satisfied) and np.isnan(float(ratio))
    assert criterion.history == []

    criterion.disarm()
    is_satisfied, _ = criterion.check(_after(2 * FREQ))
    assert not bool(is_satisfied)
    assert criterion.history == []


def test_stops_on_a_plateau_and_reset_keeps_the_record():
    speed_ref, grounded, floating = _fields()
    criterion = _criterion(
        _FakeMapping(speed_ref * 1.05), speed_ref, grounded, floating, patience=20
    )
    criterion.arm()

    for iterations in (10, 20):
        is_satisfied, _ = criterion.check(_after(iterations))
        assert not bool(is_satisfied)
    is_satisfied, _ = criterion.check(_after(30))

    assert bool(is_satisfied)
    assert criterion.reason == StopReason.PLATEAU

    criterion.reset()
    assert [row[0] for row in criterion.history] == [10, 20, 30]


def test_targets_must_hold_for_consecutive_checks():
    speed_ref, grounded, floating = _fields()
    mapping = _FakeMapping(speed_ref * 1.001)
    criterion = _criterion(mapping, speed_ref, grounded, floating, consecutive=2)
    criterion.arm()

    is_satisfied, _ = criterion.check(_after(10))  # met once
    assert not bool(is_satisfied)

    mapping.set_speed(speed_ref * 1.05)  # missed: the streak restarts
    is_satisfied, _ = criterion.check(_after(20))
    assert not bool(is_satisfied)

    mapping.set_speed(speed_ref * 1.001)
    is_satisfied, _ = criterion.check(_after(30))
    assert not bool(is_satisfied)
    is_satisfied, _ = criterion.check(_after(40))  # met twice in a row

    assert bool(is_satisfied)
    assert criterion.reason == StopReason.TARGETS


def test_an_empty_region_counts_as_met():
    speed_ref, grounded, _ = _fields()
    everything = np.ones(SHAPE, bool)
    criterion = _criterion(
        _FakeMapping(speed_ref * 1.001), speed_ref, everything, ~everything
    )
    criterion.arm()

    is_satisfied, _ = criterion.check(_after(FREQ))

    assert bool(is_satisfied)
    assert np.isnan(criterion.history[0][2])


def test_halt_lists_an_added_success_criterion():
    speed_ref, grounded, floating = _fields()
    halt = Halt()
    halt.add_success(_criterion(_FakeMapping(speed_ref), speed_ref, grounded, floating))
    assert halt.criterion_names == ["init_stop"]


# ---- the whole initialisation, on a small grounded slab (CPU) ----


def _slab(nbit_init=60, **init_stop):
    state = igm.common.State()
    cfg = load_yaml_recursive(
        os.path.join(igm.__path__[0], "conf"), exclude=["assimilations/pretraining"]
    )
    Ny, Nx, dx = 24, 20, 100.0
    X = np.tile(np.arange(Nx) * dx, (Ny, 1))
    thk = 300.0 * np.sqrt(np.clip(1.0 - X / (0.9 * X.max()), 0.0, 1.0))
    topg = 500.0 - 0.05 * X
    state.thk = tf.Variable(thk.astype(np.float32))
    state.topg = tf.Variable(topg.astype(np.float32))
    state.usurf = tf.Variable((topg + thk).astype(np.float32))
    state.dX = tf.Variable(tf.ones((Ny, Nx)) * dx)
    state.it = -1

    cfg_unified = cfg.processes.iceflow.unified
    cfg_unified.nbit_init = nbit_init
    cfg_unified.network.seed = 0
    cfg_unified.init_stop.enabled = True
    cfg_unified.init_stop.freq = FREQ
    for key, value in init_stop.items():
        cfg_unified.init_stop[key] = value
    return cfg, state


def _fake_reference(cfg, state, inputs):
    speed = tf.fill(tf.shape(state.thk), tf.constant(100.0, tf.float32))
    return ReferenceSolve(speed=speed, iterations=0, seconds=0.0, converged=True)


def _log_rows(path="init_stop.csv"):
    with open(path) as file:
        return list(csv.DictReader(file))


def test_initial_training_stops_at_the_first_check_on_loose_targets(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(init_stop_module, "solve_reference", _fake_reference)
    cfg, state = _slab(target_grounded=1.0e6, target_floating=1.0e6, display=False)

    igm.processes.iceflow.initialize(cfg, state)

    rows = _log_rows()
    assert [int(row["iterations"]) for row in rows] == [FREQ]
    assert "stopped after 10 of 60 iterations (targets met)" in capsys.readouterr().out
    criterion = state.iceflow.optimizer.halt.crit_success[-1]
    assert criterion.name == "init_stop" and not bool(criterion.active)


def test_initial_training_runs_to_the_cap_when_targets_are_out_of_reach(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(init_stop_module, "solve_reference", _fake_reference)
    cfg, state = _slab(target_grounded=0.0, target_floating=0.0, patience=0)

    igm.processes.iceflow.initialize(cfg, state)

    rows = _log_rows()
    assert [int(row["iterations"]) for row in rows] == [10, 20, 30, 40, 50, 60]


def test_display_changes_nothing_but_the_output(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(init_stop_module, "solve_reference", _fake_reference)

    velocities = {}
    outputs = {}
    for display in (False, True):
        cfg, state = _slab(target_grounded=0.0, target_floating=0.0, display=display)
        igm.processes.iceflow.initialize(cfg, state)
        velocities[display] = state.U.numpy()
        outputs[display] = capsys.readouterr().out

    np.testing.assert_array_equal(velocities[False], velocities[True])
    assert "initial training, stopped on its error" in outputs[True]
    assert "Direct solve" in outputs[True]
    assert "iterations   grounded   floating" in outputs[True]
    assert "nbit_init reached" in outputs[True]
    assert "Direct solve" not in outputs[False]


def test_disabled_leaves_the_initial_training_alone(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg, state = _slab()
    cfg.processes.iceflow.unified.init_stop.enabled = False

    igm.processes.iceflow.initialize(cfg, state)

    assert not os.path.exists("init_stop.csv")


def test_real_direct_solve(tmp_path, monkeypatch):
    """The reference comes from an actual cg_newton solve (two Newton steps here)."""
    monkeypatch.chdir(tmp_path)
    cfg, state = _slab(nbit_init=20, display=False)
    cfg.processes.iceflow.unified.init_stop.reference.nbit = 2

    captured = {}
    solve = init_stop_module.solve_reference

    def spy(cfg, state, inputs):
        captured["reference"] = solve(cfg, state, inputs)
        return captured["reference"]

    monkeypatch.setattr(init_stop_module, "solve_reference", spy)
    igm.processes.iceflow.initialize(cfg, state)

    reference = captured["reference"]
    assert reference.iterations == 2
    assert float(tf.reduce_max(reference.speed)) > 0.0
    assert len(_log_rows()) == 2


def _reference_of(cfg, state, monkeypatch):
    """Initialise the ice flow and return its reference solve and init_stop criterion."""
    captured = {}
    solve = init_stop_module.solve_reference

    def spy(cfg, state, inputs):
        captured["reference"] = solve(cfg, state, inputs)
        return captured["reference"]

    monkeypatch.setattr(init_stop_module, "solve_reference", spy)
    igm.processes.iceflow.initialize(cfg, state)
    criterion = state.iceflow.optimizer.halt.crit_success[-1]
    return captured["reference"], criterion


def test_double_direct_solve_in_a_single_precision_run(tmp_path, monkeypatch):
    """reference.precision=double: the direct solve of a double run, compared in single."""
    monkeypatch.chdir(tmp_path)

    cfg, state = _slab(nbit_init=10, display=False)
    cfg.processes.iceflow.unified.init_stop.reference.nbit = 2
    cfg.processes.iceflow.unified.init_stop.reference.precision = "double"
    reference, criterion = _reference_of(cfg, state, monkeypatch)

    # the same direct solve in a double run (identity mapping: no network needed)
    cfg_double, state_double = _slab(nbit_init=1, display=False)
    cfg_double.processes.iceflow.numerics.precision = "double"
    cfg_double.processes.iceflow.unified.mapping = "identity"
    cfg_double.processes.iceflow.unified.init_stop.enabled = False
    cfg_double.processes.iceflow.unified.init_stop.reference.nbit = 2
    igm.processes.iceflow.initialize(cfg_double, state_double)
    reference_double = reference_module.solve_reference(
        cfg_double,
        state_double,
        get_evaluator_inputs_from_state(cfg_double, state_double),
    )

    assert reference.iterations == reference_double.iterations == 2
    assert reference.speed.dtype == tf.float32
    assert criterion.speed_ref.dtype == tf.float32
    assert reference_double.speed.dtype == tf.float64
    np.testing.assert_allclose(
        reference.speed.numpy(),
        reference_double.speed.numpy().astype(np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def _check_cfg_args(**overrides):
    """A config that passes InitStop.check_cfg, with ``overrides`` merged over it."""
    from types import SimpleNamespace

    from omegaconf import OmegaConf

    cfg_unified = OmegaConf.create(
        {
            "mapping": "network",
            "adaptive_patching": {"enabled": False},
            "init_stop": {
                "freq": 50,
                "patience": 500,
                "speed_floor": 10.0,
                "consecutive": 1,
                "reference": {"nbit": 200},
            },
        }
    )
    cfg_unified = OmegaConf.merge(cfg_unified, overrides)
    cfg = OmegaConf.create({"processes": {"iceflow": {"unified": cfg_unified}}})
    optimizer = SimpleNamespace(name="adam", halt=SimpleNamespace(freq=1))
    state = SimpleNamespace(iceflow=SimpleNamespace(optimizer=optimizer))
    return cfg, state


@pytest.mark.parametrize(
    "overrides",
    [
        {"mapping": "identity"},
        {"adaptive_patching": {"enabled": True}},
        {"init_stop": {"freq": 0}},
        {"init_stop": {"patience": -1}},
        {"init_stop": {"speed_floor": 0.0}},
        {"init_stop": {"consecutive": 0}},
        {"init_stop": {"reference": {"nbit": 0}}},
        {"init_stop": {"reference": {"precision": "half"}}},
    ],
)
def test_unsupported_settings_are_rejected(overrides):
    cfg, state = _check_cfg_args(**overrides)
    with pytest.raises(ValueError):
        init_stop_module.InitStop.check_cfg(cfg, state)


@pytest.mark.parametrize("precision", [None, "single", "double"])
def test_supported_settings_pass_the_check(precision):
    cfg, state = _check_cfg_args(init_stop={"reference": {"precision": precision}})
    init_stop_module.InitStop.check_cfg(cfg, state)


@pytest.mark.parametrize("optimizer", ["adam", "ss_esoap"])
def test_the_last_check_sees_the_final_network(tmp_path, monkeypatch, optimizer):
    """Each check reads the weights after the update of its iteration (not one behind)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(init_stop_module, "solve_reference", _fake_reference)
    cfg, state = _slab(
        nbit_init=30, target_grounded=0.0, target_floating=0.0, patience=0, freq=1
    )
    cfg.processes.iceflow.unified.optimizer = optimizer

    igm.processes.iceflow.initialize(cfg, state)

    criterion = state.iceflow.optimizer.halt.crit_success[-1]
    final_grounded, _ = [float(e) for e in criterion.errors()]
    iterations, last_grounded, _ = criterion.history[-1]
    assert iterations == 30
    assert last_grounded == final_grounded
