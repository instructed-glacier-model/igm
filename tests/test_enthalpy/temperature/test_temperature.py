#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Unit tests for temperature-enthalpy conversions."""

import pytest
import numpy as np
import tensorflow as tf

from igm.processes.enthalpy.temperature.utils import (
    compute_pmp_tf,
    compute_T_tf,
    compute_omega_tf,
    compute_E_cold_tf,
)

# Data type
dtype = tf.float32

# Physical constants
rho_ice = tf.constant(910.0, dtype=dtype)
g = tf.constant(9.81, dtype=dtype)
beta = tf.constant(7.9e-8, dtype=dtype)
c_ice = tf.constant(2009.0, dtype=dtype)
L_ice = tf.constant(334000.0, dtype=dtype)
T_pmp_ref = tf.constant(273.15, dtype=dtype)
T_ref = tf.constant(223.15, dtype=dtype)


@pytest.mark.parametrize(
    "depth,T_pmp_expected",
    [
        (0.0, 273.15),
        (1000.0, 272.44),
        (3000.0, 271.03),
    ],
)
def test_T_pmp_depth(depth: float, T_pmp_expected: float) -> None:
    """Test pressure melting point decreases with depth."""
    depth_tf = tf.constant([[depth]], dtype=dtype)

    T_pmp, E_pmp = compute_pmp_tf(rho_ice, g, depth_tf, beta, c_ice, T_pmp_ref, T_ref)

    np.testing.assert_allclose(T_pmp.numpy(), T_pmp_expected, rtol=1e-3)


def test_T_pmp_surface() -> None:
    """Test pressure melting point at surface equals reference."""
    depth = tf.constant([[0.0]], dtype=dtype)

    T_pmp, E_pmp = compute_pmp_tf(rho_ice, g, depth, beta, c_ice, T_pmp_ref, T_ref)

    np.testing.assert_allclose(T_pmp.numpy(), T_pmp_ref.numpy(), rtol=1e-6)


@pytest.mark.parametrize(
    "E,E_pmp,T_expected",
    [
        (-50000.0, 0.0, 198.26),
        (1000.0, 0.0, 223.15),
        (0.0, 0.0, 223.15),
        (-100000.0, 0.0, 173.37),
    ],
)
def test_compute_T(E: float, E_pmp: float, T_expected: float) -> None:
    """Test temperature computation from enthalpy."""
    E_tf = tf.constant([[E]], dtype=dtype)
    E_pmp_tf = tf.constant([[E_pmp]], dtype=dtype)

    T = compute_T_tf(E_tf, E_pmp_tf, T_ref, c_ice)

    np.testing.assert_allclose(T.numpy(), T_expected, rtol=1e-3)


def test_T_cold_ice() -> None:
    """Test cold ice temperature is less than PMP."""
    E = tf.constant([[-50000.0]], dtype=dtype)
    E_pmp = tf.constant([[0.0]], dtype=dtype)
    T_pmp = tf.constant([[223.15]], dtype=dtype)

    T = compute_T_tf(E, E_pmp, T_ref, c_ice)

    assert T.numpy()[0, 0] < T_pmp.numpy()[0, 0]


def test_T_temperate_ice() -> None:
    """Test temperate ice temperature equals PMP."""
    E = tf.constant([[10000.0]], dtype=dtype)
    E_pmp = tf.constant([[0.0]], dtype=dtype)
    T_pmp = tf.constant([[223.15]], dtype=dtype)

    T = compute_T_tf(E, E_pmp, T_ref, c_ice)

    np.testing.assert_allclose(T.numpy(), T_pmp.numpy(), rtol=1e-6)


@pytest.mark.parametrize(
    "E,E_pmp,omega_expected",
    [
        (-50000.0, 0.0, 0.0),
        (L_ice.numpy() * 0.01, 0.0, 0.01),
        (L_ice.numpy() * 0.03, 0.0, 0.03),
        (0.0, 0.0, 0.0),
    ],
)
def test_compute_omega(E: float, E_pmp: float, omega_expected: float) -> None:
    """Test water content computation from enthalpy."""
    E_tf = tf.constant([[E]], dtype=dtype)
    E_pmp_tf = tf.constant([[E_pmp]], dtype=dtype)

    omega = compute_omega_tf(E_tf, E_pmp_tf, L_ice)

    np.testing.assert_allclose(omega.numpy(), omega_expected, rtol=1e-3)


def test_omega_cold_ice() -> None:
    """Test cold ice has zero water content."""
    E = tf.constant([[-50000.0]], dtype=dtype)
    E_pmp = tf.constant([[0.0]], dtype=dtype)

    omega = compute_omega_tf(E, E_pmp, L_ice)

    np.testing.assert_allclose(omega.numpy(), 0.0, atol=1e-10)


@pytest.mark.parametrize(
    "T,T_pmp,E_expected",
    [
        (263.15, 273.15, c_ice.numpy() * (263.15 - T_ref.numpy())),
        (283.15, 273.15, c_ice.numpy() * (273.15 - T_ref.numpy())),
    ],
)
def test_compute_E_cold(T: float, T_pmp: float, E_expected: float) -> None:
    """Test cold ice enthalpy computation from temperature."""
    T_tf = tf.constant([[T]], dtype=dtype)
    T_pmp_tf = tf.constant([[T_pmp]], dtype=dtype)

    E = compute_E_cold_tf(T_tf, T_pmp_tf, T_ref, c_ice)

    np.testing.assert_allclose(E.numpy(), E_expected, rtol=1e-5)


def test_consistency_cold() -> None:
    """Test E -> T -> E round-trip for cold ice."""
    E_original = tf.constant([[-40000.0]], dtype=dtype)
    E_pmp = tf.constant([[0.0]], dtype=dtype)
    T_pmp = tf.constant([[273.15]], dtype=dtype)

    # E -> T
    T = compute_T_tf(E_original, E_pmp, T_ref, c_ice)
    # T -> E
    E_final = compute_E_cold_tf(T, T_pmp, T_ref, c_ice)

    np.testing.assert_allclose(E_final.numpy(), E_original.numpy(), rtol=1e-5)


def test_T_omega_shapes() -> None:
    """Test that functions handle 3D arrays correctly."""
    nz, ny, nx = 5, 3, 4
    E = tf.ones((nz, ny, nx), dtype=dtype) * -30000.0
    E_pmp = tf.zeros((nz, ny, nx), dtype=dtype)

    T = compute_T_tf(E, E_pmp, T_ref, c_ice)
    omega = compute_omega_tf(E, E_pmp, L_ice)

    assert T.shape == (nz, ny, nx)
    assert omega.shape == (nz, ny, nx)
