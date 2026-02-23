#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Unit tests for Arrhenius factor computations."""

import pytest
import numpy as np
import tensorflow as tf

from igm.processes.enthalpy.temperature.utils import compute_pa_tf
from igm.processes.enthalpy.arrhenius.utils import compute_arrhenius_3d_tf

# Data type
dtype = tf.float32

# Physical constants
rho_ice = tf.constant(910.0, dtype=dtype)
g = tf.constant(9.81, dtype=dtype)
beta = tf.constant(7.9e-8, dtype=dtype)
T_threshold = tf.constant(263.15, dtype=dtype)
A_cold = tf.constant(3.985e-13, dtype=dtype)
A_warm = tf.constant(1.916e3, dtype=dtype)
Q_cold = tf.constant(60000.0, dtype=dtype)
Q_warm = tf.constant(139000.0, dtype=dtype)
omega_coef = tf.constant(181.25, dtype=dtype)
omega_max = tf.constant(0.01, dtype=dtype)
R = tf.constant(8.314, dtype=dtype)


def test_T_pa_surface() -> None:
    """Test pressure-adjusted temperature at surface equals input T."""
    T = tf.constant([[[260.0]]], dtype=dtype)
    depth_ice = tf.constant([[[0.0]]], dtype=dtype)

    T_pa = compute_pa_tf(T, beta, rho_ice, g, depth_ice)

    np.testing.assert_allclose(T_pa.numpy(), T.numpy(), rtol=1e-6)


@pytest.mark.parametrize(
    "depth,delta_expected",
    [
        (0.0, 0.0),
        (500.0, 0.354),
        (1000.0, 0.708),
        (2000.0, 1.416),
    ],
)
def test_T_pa_depth(depth: float, delta_expected: float) -> None:
    """Test pressure adjustment scales linearly with depth."""
    T = tf.constant([[[260.0]]], dtype=dtype)
    depth_ice = tf.constant([[[depth]]], dtype=dtype)

    T_pa = compute_pa_tf(T, beta, rho_ice, g, depth_ice)

    np.testing.assert_allclose(
        T_pa.numpy()[0, 0, 0] - T.numpy()[0, 0, 0],
        delta_expected,
        rtol=1e-2,
    )


def test_arrhenius_cold() -> None:
    """Test Arrhenius factor uses cold regime parameters below threshold."""
    T_pa = tf.constant([[[250.0]]], dtype=dtype)
    omega = tf.constant([[[0.0]]], dtype=dtype)

    A = compute_arrhenius_3d_tf(
        omega,
        T_pa,
        T_threshold,
        A_cold,
        A_warm,
        Q_cold,
        Q_warm,
        omega_coef,
        omega_max,
        R,
    )

    assert A.numpy()[0, 0, 0] > 0.0
    assert np.isfinite(A.numpy()[0, 0, 0])


def test_arrhenius_warm() -> None:
    """Test Arrhenius factor uses warm regime parameters above threshold."""
    T_pa = tf.constant([[[270.0]]], dtype=dtype)
    omega = tf.constant([[[0.0]]], dtype=dtype)

    A = compute_arrhenius_3d_tf(
        omega,
        T_pa,
        T_threshold,
        A_cold,
        A_warm,
        Q_cold,
        Q_warm,
        omega_coef,
        omega_max,
        R,
    )

    assert A.numpy()[0, 0, 0] > 0.0
    assert np.isfinite(A.numpy()[0, 0, 0])


def test_arrhenius_omega() -> None:
    """Test water content enhances Arrhenius factor."""
    T_pa = tf.constant([[[270.0]]], dtype=dtype)
    omega_dry = tf.constant([[[0.0]]], dtype=dtype)
    omega_wet = tf.constant([[[0.01]]], dtype=dtype)

    A_dry = compute_arrhenius_3d_tf(
        omega_dry,
        T_pa,
        T_threshold,
        A_cold,
        A_warm,
        Q_cold,
        Q_warm,
        omega_coef,
        omega_max,
        R,
    )
    A_wet = compute_arrhenius_3d_tf(
        omega_wet,
        T_pa,
        T_threshold,
        A_cold,
        A_warm,
        Q_cold,
        Q_warm,
        omega_coef,
        omega_max,
        R,
    )

    expected_ratio = 1.0 + omega_coef.numpy() * 0.01
    np.testing.assert_allclose(
        A_wet.numpy() / A_dry.numpy(),
        expected_ratio,
        rtol=1e-5,
    )


def test_arrhenius_shape() -> None:
    """Test output shape matches input."""
    nz, ny, nx = 5, 3, 4
    T_pa = tf.ones((nz, ny, nx), dtype=dtype) * 260.0
    omega = tf.zeros((nz, ny, nx), dtype=dtype)

    A = compute_arrhenius_3d_tf(
        omega,
        T_pa,
        T_threshold,
        A_cold,
        A_warm,
        Q_cold,
        Q_warm,
        omega_coef,
        omega_max,
        R,
    )

    assert A.shape == (nz, ny, nx)
