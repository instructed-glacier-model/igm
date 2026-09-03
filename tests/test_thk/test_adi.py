#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Tests for Peaceman--Rachford ADI thickness evolution."""

from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
from scipy.linalg import expm
import tensorflow as tf

from igm.processes.thk.transport import adi
from igm.processes.thk import thk as thk_module
from igm.utils.grad.compute_divflux import compute_divflux


def _cfg():
    return OmegaConf.create(
        {
            "processes": {
                "thk": {
                    "scheme": "adi",
                    "slope_type": "superbee",
                    "divflux_smooth_sigma": 0.0,
                    "ratio_density": 0.91,
                    "calving_front": False,
                    "method": "sub_grid",
                }
            }
        }
    )


def _state(seed=4, ny=12, nx=15, dx=200.0, dt=1.0, dtype=tf.float32):
    generator = tf.random.Generator.from_seed(seed)
    return SimpleNamespace(
        thk=tf.cast(50.0 + 300.0 * generator.uniform((ny, nx)), dtype),
        topg=tf.zeros((ny, nx), dtype=dtype),
        ubar=tf.cast(30.0 * generator.normal((ny, nx)), dtype),
        vbar=tf.cast(30.0 * generator.normal((ny, nx)), dtype),
        smb=tf.cast(0.2 * generator.normal((ny, nx)), dtype),
        dx=tf.constant(dx, dtype=dtype),
        dt=tf.constant(dt, dtype=dtype),
        it=0,
    )


def test_directional_operators_sum_to_existing_upwind_divergence():
    state = _state()
    coefficients = adi._build_directional_coefficients(
        state.ubar, state.vbar, state.dx
    )

    actual = adi._apply_x(state.thk, coefficients) + adi._apply_y(
        state.thk, coefficients
    )
    expected = compute_divflux(
        state.ubar,
        state.vbar,
        state.thk,
        state.dx,
        state.dx,
        method="upwind",
    )

    np.testing.assert_allclose(actual, expected, rtol=2.0e-6, atol=2.0e-5)


def test_batched_tridiagonal_solves_satisfy_directional_systems():
    state = _state(dt=17.0)
    coefficients = adi._build_directional_coefficients(
        state.ubar, state.vbar, state.dx
    )
    half_dt = 0.5 * state.dt

    x_solution = adi._solve_x(state.thk, coefficients, half_dt)
    x_residual = (
        x_solution
        + half_dt * adi._apply_x(x_solution, coefficients)
        - state.thk
    )
    y_solution = adi._solve_y(state.thk, coefficients, half_dt)
    y_residual = (
        y_solution
        + half_dt * adi._apply_y(y_solution, coefficients)
        - state.thk
    )

    assert float(tf.norm(x_residual) / tf.norm(state.thk)) < 2.0e-6
    assert float(tf.norm(y_residual) / tf.norm(state.thk)) < 2.0e-6


def test_zero_velocity_source_is_integrated_exactly():
    state = _state(dt=11.0)
    result = adi._peaceman_rachford_step(
        tf.zeros_like(state.ubar),
        tf.zeros_like(state.vbar),
        state.thk,
        state.dx,
        state.dt,
        state.smb,
    )

    expected = state.thk + state.dt * state.smb
    np.testing.assert_allclose(result.thickness, expected, rtol=1.0e-6)
    np.testing.assert_allclose(result.divflux, tf.zeros_like(state.thk), atol=3.0e-6)


def _dense_divergence_matrix(ubar, vbar, dx):
    """Build a tiny dense upwind operator for temporal-order verification."""
    ny, nx = ubar.shape
    coefficients = adi._build_directional_coefficients(ubar, vbar, dx)
    columns = []
    for index in range(ny * nx):
        basis = tf.reshape(tf.one_hot(index, ny * nx, dtype=ubar.dtype), (ny, nx))
        applied = adi._apply_x(basis, coefficients) + adi._apply_y(
            basis, coefficients
        )
        columns.append(tf.reshape(applied, (-1,)).numpy())
    return np.stack(columns, axis=1)


def test_peaceman_rachford_has_second_order_local_accuracy():
    state = _state(ny=3, nx=4, dx=100.0, dtype=tf.float64)
    state.smb = tf.zeros_like(state.thk)
    divergence = _dense_divergence_matrix(state.ubar, state.vbar, state.dx)
    initial = tf.reshape(state.thk, (-1,)).numpy()

    errors = []
    for dt in (1.0, 0.5, 0.25):
        result = adi._peaceman_rachford_step(
            state.ubar,
            state.vbar,
            state.thk,
            state.dx,
            tf.constant(dt, tf.float64),
            state.smb,
        )
        exact = expm(-dt * divergence) @ initial
        errors.append(
            np.linalg.norm(tf.reshape(result.thickness, (-1,)).numpy() - exact)
        )

    # A second-order method has O(dt^3) one-step local error, hence an
    # asymptotic factor of eight each time dt is halved.
    assert errors[0] / errors[1] > 6.0
    assert errors[1] / errors[2] > 6.0


def test_closed_domain_step_beyond_explicit_cfl_is_bounded_and_conservative():
    ny, nx = 50, 60
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, ny),
        np.linspace(-1.0, 1.0, nx),
        indexing="ij",
    )
    thickness = tf.constant(
        (200.0 + 800.0 * np.exp(-3.0 * (xx**2 + yy**2))).astype("float32")
    )
    ubar = tf.constant(
        (250.0 * np.sin(np.pi * xx) * np.cos(np.pi * yy)).astype("float32")
    )
    vbar = tf.constant(
        (-250.0 * np.cos(np.pi * xx) * np.sin(np.pi * yy)).astype("float32")
    )
    dx = tf.constant(1000.0)
    dt = tf.constant(40.0)  # directional CFL approximately 10

    result = adi._peaceman_rachford_step(
        ubar, vbar, thickness, dx, dt, tf.zeros_like(thickness)
    )

    assert bool(tf.reduce_all(tf.math.is_finite(result.thickness)))
    relative_mass_error = tf.abs(
        tf.reduce_sum(result.thickness) - tf.reduce_sum(thickness)
    ) / tf.reduce_sum(thickness)
    assert float(relative_mass_error) < 5.0e-6
    assert float(tf.norm(result.thickness) / tf.norm(thickness)) < 1.1


def test_thk_update_dispatches_to_adi_backend():
    cfg = _cfg()
    state = _state(dt=2.0)

    thk_module.initialize(cfg, state)
    thickness_old = tf.identity(state.thk)
    thk_module.update(cfg, state)

    assert state.thk.shape == thickness_old.shape
    assert state.divflux.shape == thickness_old.shape
    assert bool(tf.reduce_all(tf.math.is_finite(state.thk)))
    assert float(tf.reduce_min(state.thk)) >= 0.0


def test_adi_projection_is_diagnosed_and_public_budget_closes():
    ny, nx = 4, 80
    thickness = np.zeros((ny, nx), dtype=np.float32)
    thickness[:, 20:40] = 100.0
    state = _state(ny=ny, nx=nx, dx=1.0, dt=10.0)
    state.thk = tf.constant(thickness)
    state.ubar = tf.ones_like(state.thk)
    state.vbar = tf.zeros_like(state.thk)
    state.smb = tf.zeros_like(state.thk)

    result = adi.solve(state, state.smb)

    assert float(result.nonnegative_correction_volume) > 0.0
    assert float(tf.reduce_min(result.thickness)) == 0.0
    np.testing.assert_allclose(
        result.thickness - state.thk,
        state.dt * (state.smb - result.divflux),
        rtol=2.0e-6,
        atol=2.0e-5,
    )
    assert float(tf.norm(result.divflux - result.transport_divflux)) > 0.0


def test_adi_parameters_do_not_retrace_the_graph():
    state = _state()
    before = adi._peaceman_rachford_step.experimental_get_tracing_count()

    for dt in (1.0, 5.0, 0.5):
        adi._peaceman_rachford_step(
            state.ubar,
            state.vbar,
            state.thk,
            state.dx,
            tf.constant(dt),
            state.smb,
        )

    after = adi._peaceman_rachford_step.experimental_get_tracing_count()
    assert after - before <= 1
