#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Tests for the TensorFlow-native implicit thickness scheme."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest
import tensorflow as tf

from igm.processes.thk.transport import implicit
from igm.processes.thk import thk as thk_module
from igm.utils.grad.compute_divflux import compute_divflux
from igm.utils.grad.compute_divflux_slope_limiter import (
    compute_divflux_slope_limiter,
)


def _solver_cfg(theta=1.0, tol=1.0e-7, max_iter=500, max_restarts=4):
    return OmegaConf.create(
        {
            "processes": {
                "thk": {
                    "scheme": "implicit",
                    "slope_type": "superbee",
                    "divflux_smooth_sigma": 0.0,
                    "ratio_density": 0.91,
                    "implicit": {
                        "theta": theta,
                        "solver": {
                            "tol": tol,
                            "max_iter": max_iter,
                            "max_restarts": max_restarts,
                        },
                    },
                    "calving_front": False,
                    "method": "sub_grid",
                }
            }
        }
    )


def _state(seed=1, ny=30, nx=40, dx=200.0, dt=1.0):
    generator = tf.random.Generator.from_seed(seed)
    return SimpleNamespace(
        thk=50.0 + 300.0 * generator.uniform((ny, nx)),
        topg=tf.zeros((ny, nx), dtype=tf.float32),
        ubar=30.0 * generator.normal((ny, nx)),
        vbar=30.0 * generator.normal((ny, nx)),
        smb=0.2 * generator.normal((ny, nx)),
        dx=tf.constant(dx, dtype=tf.float32),
        dt=tf.constant(dt, dtype=tf.float32),
        t=tf.constant(0.0, dtype=tf.float32),
        it=0,
    )


def test_stencil_matches_existing_upwind_operator():
    state = _state()
    dt_theta = tf.constant(2.3, dtype=state.thk.dtype)
    coefficients = implicit._build_operator_coefficients(
        state.ubar, state.vbar, state.dx, dt_theta
    )

    actual = implicit._apply_operator(state.thk, coefficients)
    expected = state.thk + dt_theta * compute_divflux(
        state.ubar,
        state.vbar,
        state.thk,
        state.dx,
        state.dx,
        method="upwind",
    )

    np.testing.assert_allclose(actual, expected, rtol=2.0e-6, atol=2.0e-4)


def test_theta_step_satisfies_discrete_equation():
    state = _state(dt=3.0)
    theta = tf.constant(0.65, dtype=state.thk.dtype)
    result = implicit._solve_theta_step(
        state.ubar,
        state.vbar,
        state.thk,
        state.dx,
        state.dt,
        state.smb,
        theta,
        tf.constant(1.0e-7),
        tf.constant(500),
        tf.constant(4),
    )

    coefficients = implicit._build_operator_coefficients(
        state.ubar, state.vbar, state.dx, state.dt * theta
    )
    operator_old = implicit._apply_operator(state.thk, coefficients)
    rhs = (
        state.thk
        - ((1.0 - theta) / theta) * (operator_old - state.thk)
        + state.dt * state.smb
    )
    residual = rhs - implicit._apply_operator(result.thickness, coefficients)

    assert bool(result.converged)
    assert not bool(result.breakdown)
    assert float(tf.norm(residual) / tf.norm(rhs)) < 2.0e-6


def test_backward_euler_is_stable_and_conservative_beyond_explicit_cfl():
    ny, nx = 50, 60
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, ny),
        np.linspace(-1.0, 1.0, nx),
        indexing="ij",
    )
    thickness = tf.constant(
        (200.0 + 800.0 * np.exp(-3.0 * (xx**2 + yy**2))).astype("float32")
    )
    # Smooth closed-domain circulation. Boundary-normal velocities are exactly
    # zero, hence the finite-volume divergence has zero domain integral.
    ubar = tf.constant(
        (250.0 * np.sin(np.pi * xx) * np.cos(np.pi * yy)).astype("float32")
    )
    vbar = tf.constant(
        (-250.0 * np.cos(np.pi * xx) * np.sin(np.pi * yy)).astype("float32")
    )
    dx = tf.constant(1000.0)
    dt = tf.constant(400.0)  # directional CFL ~= 100

    result = implicit._solve_theta_step(
        ubar,
        vbar,
        thickness,
        dx,
        dt,
        tf.zeros_like(thickness),
        tf.constant(1.0),
        tf.constant(1.0e-6),
        tf.constant(1000),
        tf.constant(4),
    )

    assert bool(result.converged)
    assert not bool(result.breakdown)
    assert bool(tf.reduce_all(tf.math.is_finite(result.thickness)))
    assert float(tf.reduce_min(result.thickness)) >= 0.0

    coefficients = implicit._build_operator_coefficients(ubar, vbar, dx, dt)
    true_residual = thickness - implicit._apply_operator(result.thickness, coefficients)
    true_relative_residual = tf.norm(true_residual) / tf.norm(thickness)
    assert float(true_relative_residual) <= 1.05 * float(result.effective_tolerance)
    assert float(true_relative_residual) < 3.0e-5

    relative_mass_error = tf.abs(
        tf.reduce_sum(result.thickness) - tf.reduce_sum(thickness)
    ) / tf.reduce_sum(thickness)
    # Float32 reduction order differs slightly between CPU and GPU.
    assert float(relative_mass_error) < 4.0e-6


def test_thk_update_exposes_device_resident_solver_diagnostics():
    cfg = _solver_cfg()
    state = _state(dt=4.0)

    thk_module.initialize(cfg, state)
    thickness_old = tf.identity(state.thk)
    thk_module.update(cfg, state)

    assert state.thk.shape == thickness_old.shape
    assert state.divflux.shape == thickness_old.shape
    assert state.thk_solver_iterations.dtype == tf.int32
    assert state.thk_solver_restarts.dtype == tf.int32
    assert state.thk_solver_relative_residual.dtype == state.thk.dtype
    assert state.thk_solver_effective_tolerance.dtype == state.thk.dtype
    assert bool(state.thk_solver_converged)
    assert not bool(state.thk_solver_breakdown)
    assert bool(state.thk_step_accepted)


def test_failed_implicit_solve_stops_without_accepting_the_iterate():
    cfg = _solver_cfg(tol=1.0e-30, max_iter=1, max_restarts=0)
    state = _state(dt=50.0)
    thickness_old = tf.identity(state.thk)

    thk_module.initialize(cfg, state)
    thk_module.update(cfg, state)

    assert not bool(state.thk_solver_converged)
    assert not bool(state.thk_step_accepted)
    assert not bool(state.continue_run)
    np.testing.assert_allclose(state.thk, thickness_old)
    np.testing.assert_allclose(state.divflux, state.smb)


def test_invalid_implicit_failure_policy_is_rejected():
    cfg = _solver_cfg()
    cfg.processes.thk.implicit.solver.failure_policy = "guess"
    with pytest.raises(ValueError, match="failure_policy must be"):
        thk_module.initialize(cfg, _state())


@pytest.mark.parametrize("tol", [0.0, float("nan")])
def test_invalid_implicit_tolerance_is_rejected(tol):
    with pytest.raises(ValueError, match="tol must be positive and finite"):
        thk_module.initialize(_solver_cfg(tol=tol), _state())


def test_default_is_backward_euler():
    config_path = (
        Path(thk_module.__file__).parents[2] / "conf" / "processes" / "thk.yaml"
    )
    cfg = OmegaConf.load(config_path)
    assert cfg.thk.implicit.theta == 1.0


def test_explicit_scheme_is_unchanged():
    cfg = _solver_cfg()
    cfg.processes.thk.scheme = "explicit"
    cfg.processes.thk.divflux_smooth_sigma = 0.5
    state = _state(dt=0.2)

    thk_module.initialize(cfg, state)
    thickness_old = tf.identity(state.thk)
    expected_divflux = compute_divflux_slope_limiter(
        state.ubar,
        state.vbar,
        thickness_old,
        state.dx,
        state.dx,
        state.dt,
        slope_type=cfg.processes.thk.slope_type,
        smooth_sigma=cfg.processes.thk.divflux_smooth_sigma,
    )
    expected_thickness = tf.maximum(
        thickness_old + state.dt * (state.smb - expected_divflux), 0.0
    )

    thk_module.update(cfg, state)

    np.testing.assert_allclose(state.divflux, expected_divflux, rtol=1.0e-6)
    np.testing.assert_allclose(state.thk, expected_thickness, rtol=1.0e-6)


@pytest.mark.parametrize("theta", [0.49, 1.01])
def test_invalid_theta_is_rejected(theta):
    cfg = _solver_cfg(theta=theta)
    with pytest.raises(ValueError, match="theta must be between"):
        thk_module.initialize(cfg, _state())


def test_implicit_scheme_rejects_explicit_flux_smoothing():
    cfg = _solver_cfg()
    cfg.processes.thk.divflux_smooth_sigma = 0.5
    with pytest.raises(ValueError, match="available only with scheme: explicit"):
        thk_module.initialize(cfg, _state())


def test_solver_parameters_do_not_retrace_the_graph():
    state = _state(ny=12, nx=15)
    before = implicit._solve_theta_step.experimental_get_tracing_count()

    for theta in (1.0, 0.75, 1.0):
        implicit._solve_theta_step(
            state.ubar,
            state.vbar,
            state.thk,
            state.dx,
            state.dt,
            state.smb,
            tf.constant(theta),
            tf.constant(1.0e-6),
            tf.constant(100),
            tf.constant(4),
        )

    after = implicit._solve_theta_step.experimental_get_tracing_count()
    assert after - before <= 1


@pytest.mark.parametrize("boundary_mode", ["symmetric", "periodic"])
def test_closed_implicit_boundaries_conserve_mass(boundary_mode):
    state = _state(seed=17, ny=12, nx=18, dx=1.0, dt=4.0)
    state.ubar = tf.ones_like(state.thk)
    state.vbar = -0.5 * tf.ones_like(state.thk)
    state.smb = tf.zeros_like(state.thk)
    cfg = _solver_cfg(theta=1.0, tol=1.0e-6, max_iter=1000)
    cfg.processes.thk.boundary = {
        "left": boundary_mode,
        "right": boundary_mode,
        "top": boundary_mode,
        "bottom": boundary_mode,
    }

    result = implicit.solve(state, cfg, state.smb)

    assert bool(result.converged)
    assert float(tf.reduce_min(result.thickness)) >= 0.0
    np.testing.assert_allclose(
        tf.reduce_sum(result.thickness),
        tf.reduce_sum(state.thk),
        rtol=5.0e-6,
    )


def test_implicit_projection_is_diagnosed_and_public_budget_closes():
    ny, nx = 4, 80
    thickness = np.zeros((ny, nx), dtype=np.float32)
    thickness[:, 20:40] = 100.0
    state = _state(ny=ny, nx=nx, dx=1.0, dt=5.0)
    state.thk = tf.constant(thickness)
    state.ubar = tf.ones_like(state.thk)
    state.vbar = tf.zeros_like(state.thk)
    state.smb = tf.zeros_like(state.thk)
    cfg = _solver_cfg(theta=0.5, tol=1.0e-7, max_iter=1000)

    result = implicit.solve(state, cfg, state.smb)

    assert float(result.nonnegative_correction_volume) > 0.0
    assert float(tf.reduce_min(result.thickness)) == 0.0
    np.testing.assert_allclose(
        result.thickness - state.thk,
        state.dt * (state.smb - result.divflux),
        rtol=2.0e-6,
        atol=2.0e-5,
    )
    assert float(tf.norm(result.divflux - result.transport_divflux)) > 0.0


def test_implicit_supports_a_symmetric_left_and_open_right_side():
    ny, nx = 5, 40
    state = _state(ny=ny, nx=nx, dx=1.0, dt=1.0)
    state.thk = tf.ones((ny, nx), tf.float32)
    state.ubar = -tf.ones_like(state.thk)
    state.vbar = tf.zeros_like(state.thk)
    state.smb = tf.zeros_like(state.thk)
    mixed_cfg = _solver_cfg(theta=1.0, tol=1.0e-7, max_iter=1000)
    mixed_cfg.processes.thk.boundary = {
        "left": "symmetric",
        "right": "zero",
        "top": "zero",
        "bottom": "zero",
    }
    open_cfg = _solver_cfg(theta=1.0, tol=1.0e-7, max_iter=1000)
    open_cfg.processes.thk.boundary = {
        "left": "zero",
        "right": "zero",
        "top": "zero",
        "bottom": "zero",
    }

    mixed = implicit.solve(state, mixed_cfg, state.smb)
    opened = implicit.solve(state, open_cfg, state.smb)

    assert bool(mixed.converged)
    assert bool(opened.converged)
    np.testing.assert_allclose(
        tf.reduce_sum(mixed.thickness), tf.reduce_sum(state.thk), rtol=5.0e-6
    )
    assert float(tf.reduce_sum(opened.thickness)) < float(
        tf.reduce_sum(state.thk)
    )
