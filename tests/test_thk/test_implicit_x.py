#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Tests for the batched tridiagonal x-flowline thickness backend."""

from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest
import tensorflow as tf

from igm.processes.thk.transport import implicit, implicit_x
from igm.processes.thk import thk as thk_module


def _cfg(theta=1.0, left="zero", right="zero"):
    return OmegaConf.create(
        {
            "processes": {
                "thk": {
                    "scheme": "implicit_x",
                    "slope_type": "superbee",
                    "divflux_smooth_sigma": 0.0,
                    "ratio_density": 0.91,
                    "boundary": {
                        "left": left,
                        "right": right,
                        "top": "zero",
                        "bottom": "zero",
                    },
                    "implicit": {
                        "theta": theta,
                        "solver": {
                            "tol": 1.0e-7,
                            "max_iter": 1000,
                            "max_restarts": 4,
                        },
                    },
                    "calving_front": False,
                    "method": "sub_grid",
                }
            }
        }
    )


def _state(seed=23, ny=7, nx=31, dx=2.0, dt=1.0):
    generator = tf.random.Generator.from_seed(seed)
    thickness = 50.0 + 100.0 * generator.uniform((ny, nx))
    return SimpleNamespace(
        thk=thickness,
        topg=tf.zeros_like(thickness),
        ubar=4.0 * generator.normal((ny, nx)),
        vbar=20.0 * generator.normal((ny, nx)),
        smb=0.1 * generator.normal((ny, nx)),
        dx=tf.constant(dx, tf.float32),
        dt=tf.constant(dt, tf.float32),
        it=0,
    )


def test_tridiagonal_solution_satisfies_the_theta_equation():
    # Keep this residual test in the monotone regime so the deliberately
    # separate nonnegative projection is inactive.
    state = _state(dt=0.2)
    theta = tf.constant(0.7, tf.float32)
    result = implicit_x._solve_theta_x_step(
        state.ubar,
        state.thk,
        state.dx,
        state.dt,
        state.smb,
        theta,
        "symmetric",
        "zero",
    )
    coefficients = implicit_x._build_x_coefficients(
        state.ubar, state.dx, "symmetric", "zero"
    )
    old_divflux = implicit_x._apply_x(state.thk, coefficients)
    new_divflux = implicit_x._apply_x(result.thickness, coefficients)
    residual = (
        result.thickness
        - state.thk
        + state.dt * (theta * new_divflux + (1.0 - theta) * old_divflux)
        - state.dt * state.smb
    )

    assert float(result.nonnegative_correction_volume) == 0.0
    assert float(tf.norm(residual) / tf.norm(state.thk)) < 2.0e-6


def test_direct_x_solver_matches_matrix_free_2d_solver_when_v_is_zero():
    state = _state(dt=2.0)
    state.vbar = tf.zeros_like(state.vbar)
    x_cfg = _cfg(theta=1.0, left="symmetric", right="zero")
    full_cfg = _cfg(theta=1.0, left="symmetric", right="zero")
    full_cfg.processes.thk.scheme = "implicit"

    direct = implicit_x.solve(state, x_cfg, state.smb)
    matrix_free = implicit.solve(state, full_cfg, state.smb)

    assert bool(matrix_free.converged)
    np.testing.assert_allclose(
        direct.thickness, matrix_free.thickness, rtol=8.0e-6, atol=8.0e-5
    )


def test_backward_euler_flowline_is_positive_and_conservative_at_large_cfl():
    state = _state(ny=5, nx=80, dx=1.0, dt=20.0)
    state.thk = tf.ones_like(state.thk)
    state.ubar = tf.ones_like(state.ubar)
    state.smb = tf.zeros_like(state.smb)
    result = implicit_x.solve(
        state,
        _cfg(theta=1.0, left="symmetric", right="symmetric"),
        state.smb,
    )

    assert float(tf.reduce_min(result.thickness)) >= 0.0
    np.testing.assert_allclose(
        tf.reduce_sum(result.thickness), tf.reduce_sum(state.thk), rtol=3.0e-6
    )


def test_flowline_margin_can_advance_under_transport():
    ny, nx = 2, 80
    thickness = np.zeros((ny, nx), dtype=np.float32)
    thickness[:, :20] = 100.0
    state = _state(ny=ny, nx=nx, dx=1.0, dt=4.0)
    state.thk = tf.constant(thickness)
    state.ubar = tf.ones_like(state.thk)
    state.smb = tf.zeros_like(state.thk)
    cfg = _cfg(theta=1.0, left="symmetric", right="zero")

    result = implicit_x.solve(state, cfg, state.smb)

    assert float(tf.reduce_min(result.thickness)) >= 0.0
    assert float(result.thickness[0, 20]) > 0.0
    np.testing.assert_allclose(
        tf.reduce_sum(result.thickness),
        tf.reduce_sum(state.thk),
        rtol=3.0e-6,
    )


def test_flowline_margin_can_retreat_under_ablation():
    ny, nx = 2, 40
    thickness = np.zeros((ny, nx), dtype=np.float32)
    thickness[:, :20] = 1.0
    state = _state(ny=ny, nx=nx, dx=1.0, dt=1.0)
    state.thk = tf.constant(thickness)
    state.ubar = tf.zeros_like(state.thk)
    state.smb = tf.concat(
        [tf.zeros((ny, 15)), -2.0 * tf.ones((ny, 5)), tf.zeros((ny, 20))],
        axis=1,
    )

    result = implicit_x.solve(
        state,
        _cfg(theta=1.0, left="symmetric", right="zero"),
        state.smb,
    )

    np.testing.assert_allclose(result.thickness[:, 15:20], 0.0)
    np.testing.assert_allclose(
        result.thickness - state.thk,
        state.dt * (state.smb - result.divflux),
        rtol=2.0e-6,
        atol=2.0e-6,
    )


def test_flowline_open_terminus_loses_only_boundary_outflow():
    ny, nx = 3, 30
    state = _state(ny=ny, nx=nx, dx=1.0, dt=1.0)
    state.thk = tf.ones((ny, nx), tf.float32)
    state.ubar = tf.ones_like(state.thk)
    state.smb = tf.zeros_like(state.thk)

    result = implicit_x.solve(
        state,
        _cfg(theta=1.0, left="symmetric", right="zero"),
        state.smb,
    )

    mass_loss = tf.reduce_sum(state.thk - result.thickness)
    boundary_outflow = state.dt * tf.reduce_sum(result.thickness[:, -1])
    np.testing.assert_allclose(mass_loss, boundary_outflow, rtol=3.0e-6)


def test_public_dispatch_uses_implicit_x_and_ignores_y_velocity():
    state = _state()
    cfg = _cfg()
    expected = implicit_x.solve(state, cfg, state.smb)

    thk_module.initialize(cfg, state)
    thk_module.update(cfg, state)

    np.testing.assert_allclose(state.thk, expected.thickness, rtol=2.0e-6)
    np.testing.assert_allclose(state.divflux, expected.divflux, rtol=2.0e-6)


def test_periodic_boundary_is_rejected_because_system_would_be_cyclic():
    cfg = _cfg(left="periodic", right="periodic")
    with pytest.raises(ValueError, match="implicit_x.*does not support.*periodic"):
        thk_module.initialize(cfg, _state())


@pytest.mark.parametrize("theta", [0.49, 1.01])
def test_invalid_theta_is_rejected(theta):
    with pytest.raises(ValueError, match="theta must be between 0.5 and 1.0"):
        thk_module.initialize(_cfg(theta=theta), _state())


def test_scalar_parameters_do_not_retrace_the_flowline_kernel():
    state = _state()
    before = implicit_x._solve_theta_x_step.experimental_get_tracing_count()
    for dt, theta in ((1.0, 1.0), (4.0, 0.75), (0.2, 1.0)):
        implicit_x._solve_theta_x_step(
            state.ubar,
            state.thk,
            state.dx,
            tf.constant(dt),
            state.smb,
            tf.constant(theta),
        )
    after = implicit_x._solve_theta_x_step.experimental_get_tracing_count()
    assert after - before <= 1
