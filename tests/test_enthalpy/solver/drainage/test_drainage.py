#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Unit tests for drainage computations."""

import pytest
import numpy as np
import tensorflow as tf

from igm.processes.enthalpy.solver.drainage.utils import (
    compute_drainage,
    compute_fraction_drained,
)

# Data type
dtype = tf.float32

# Physical constants
L_ice = tf.constant(334000.0, dtype=dtype)
rho_ice = tf.constant(910.0, dtype=dtype)
rho_water = tf.constant(1000.0, dtype=dtype)
omega_threshold_1 = tf.constant(0.01, dtype=dtype)
omega_threshold_2 = tf.constant(0.02, dtype=dtype)
omega_threshold_3 = tf.constant(0.03, dtype=dtype)


def test_drainage_below_1() -> None:
    """Test zero drainage below first threshold."""
    omega = tf.constant([[0.005]], dtype=dtype)

    drainage = compute_drainage(
        omega, omega_threshold_1, omega_threshold_2, omega_threshold_3
    )

    drainage_expected = 0.0

    np.testing.assert_allclose(drainage.numpy(), drainage_expected, atol=1e-10)


def test_drainage_2() -> None:
    """Test linear drainage in regime 2."""
    omega = tf.constant([[0.015]], dtype=dtype)

    drainage = compute_drainage(
        omega, omega_threshold_1, omega_threshold_2, omega_threshold_3
    )

    drainage_expected = 0.5 * 0.015 - 0.005

    np.testing.assert_allclose(drainage.numpy()[0, 0], drainage_expected, rtol=1e-5)


def test_drainage_3() -> None:
    """Test steeper linear drainage in regime 3."""
    omega = tf.constant([[0.025]], dtype=dtype)

    drainage = compute_drainage(
        omega, omega_threshold_1, omega_threshold_2, omega_threshold_3
    )

    drainage_expected = 4.5 * 0.025 - 0.085

    np.testing.assert_allclose(drainage.numpy()[0, 0], drainage_expected, rtol=1e-5)


def test_drainage_above_3() -> None:
    """Test constant drainage above third threshold."""
    omega = tf.constant([[0.05]], dtype=dtype)

    drainage = compute_drainage(
        omega, omega_threshold_1, omega_threshold_2, omega_threshold_3
    )

    drainage_expected = 0.05

    np.testing.assert_allclose(drainage.numpy()[0, 0], drainage_expected, rtol=1e-5)


@pytest.mark.parametrize(
    "omega_val,drainage_expected",
    [
        (0.005, 0.0),
        (0.01, 0.0),
        (0.015, 0.5 * 0.015 - 0.005),
        (0.02, 0.5 * 0.02 - 0.005),
        (0.025, 4.5 * 0.025 - 0.085),
        (0.03, 4.5 * 0.03 - 0.085),
        (0.05, 0.05),
    ],
)
def test_drainage_regimes(omega_val: float, drainage_expected: float) -> None:
    """Test drainage across all regimes."""
    omega = tf.constant([[omega_val]], dtype=dtype)

    drainage = compute_drainage(
        omega, omega_threshold_1, omega_threshold_2, omega_threshold_3
    )

    np.testing.assert_allclose(drainage.numpy()[0, 0], drainage_expected, rtol=1e-4)


def test_fraction_drained_zero() -> None:
    """Test zero water content gives zero drained fraction."""
    nz = 5
    E_pmp = tf.constant(100000.0, dtype=dtype)
    E = tf.ones((nz, 1, 1), dtype=dtype) * E_pmp
    E_pmp_field = tf.ones((nz, 1, 1), dtype=dtype) * E_pmp
    omega_target = tf.constant(0.0, dtype=dtype)
    dz = tf.ones((nz - 1, 1, 1), dtype=dtype) * 100.0
    dt = tf.constant(1.0, dtype=dtype)

    fraction, h_drained = compute_fraction_drained(
        E,
        E_pmp_field,
        L_ice,
        omega_target,
        omega_threshold_1,
        omega_threshold_2,
        omega_threshold_3,
        rho_ice,
        rho_water,
        dz,
        dt,
    )

    fraction_expected = 0.0
    h_drained_expected = 0.0

    np.testing.assert_allclose(fraction.numpy(), fraction_expected, atol=1e-10)
    np.testing.assert_allclose(h_drained.numpy(), h_drained_expected, atol=1e-10)


def test_fraction_drained_positive() -> None:
    """Test high water content produces positive drainage."""
    nz = 5
    E_pmp = tf.constant(100000.0, dtype=dtype)
    omega_excess = 0.02
    E = tf.ones((nz, 1, 1), dtype=dtype) * (E_pmp + omega_excess * L_ice)
    E_pmp_field = tf.ones((nz, 1, 1), dtype=dtype) * E_pmp
    omega_target = tf.constant(0.01, dtype=dtype)
    dz = tf.ones((nz - 1, 1, 1), dtype=dtype) * 100.0
    dt = tf.constant(1.0, dtype=dtype)

    fraction, h_drained = compute_fraction_drained(
        E,
        E_pmp_field,
        L_ice,
        omega_target,
        omega_threshold_1,
        omega_threshold_2,
        omega_threshold_3,
        rho_ice,
        rho_water,
        dz,
        dt,
    )

    assert fraction.numpy().max() > 0.0
    assert h_drained.numpy()[0, 0] > 0.0


def test_fraction_drained_shape() -> None:
    """Test output shapes match input."""
    nz, ny, nx = 5, 3, 4
    E_pmp = tf.constant(100000.0, dtype=dtype)
    E = tf.ones((nz, ny, nx), dtype=dtype) * E_pmp
    E_pmp_field = tf.ones((nz, ny, nx), dtype=dtype) * E_pmp
    omega_target = tf.constant(0.0, dtype=dtype)
    dz = tf.ones((nz - 1, ny, nx), dtype=dtype) * 100.0
    dt = tf.constant(1.0, dtype=dtype)

    fraction, h_drained = compute_fraction_drained(
        E,
        E_pmp_field,
        L_ice,
        omega_target,
        omega_threshold_1,
        omega_threshold_2,
        omega_threshold_3,
        rho_ice,
        rho_water,
        dz,
        dt,
    )

    assert fraction.shape == (nz, ny, nx)
    assert h_drained.shape == (ny, nx)
