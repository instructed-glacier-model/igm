#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Unit tests for the till_storage effective-pressure parameterisation."""

import numpy as np
import tensorflow as tf

from igm.processes.effective_pressure.till_storage import compute_N_MPa_tf

# Data type
dtype = tf.float32

# Physical constants
rho_ice = tf.constant(910.0, dtype=dtype)
rho_water = tf.constant(1000.0, dtype=dtype)
g = tf.constant(9.81, dtype=dtype)


def test_N_dry() -> None:
    """Test effective pressure for dry till (s=0)."""
    h_water_till = tf.constant([[0.0]], dtype=dtype)
    h_water_till_max = tf.constant(2.0, dtype=dtype)
    h_ice = tf.constant([[1000.0]], dtype=dtype)
    N_ref = tf.constant(1.0e-3, dtype=dtype)   # MPa
    e_ref = tf.constant(0.69, dtype=dtype)
    C_c = tf.constant(0.12, dtype=dtype)
    delta = tf.constant(0.02, dtype=dtype)

    N = compute_N_MPa_tf(
        h_water_till, h_water_till_max, rho_ice, g, h_ice, N_ref, e_ref, C_c, delta
    )

    p_ice_MPa = rho_ice.numpy() * g.numpy() * 1000.0 * 1.0e-6
    N_expected = N_ref.numpy() * 10.0 ** (e_ref.numpy() / C_c.numpy())
    N_expected = min(p_ice_MPa, N_expected)
    np.testing.assert_allclose(N.numpy()[0, 0], N_expected, rtol=1e-4)


def test_N_saturated() -> None:
    """Test effective pressure for saturated till (s=1)."""
    h_water_till = tf.constant([[2.0]], dtype=dtype)
    h_water_till_max = tf.constant(2.0, dtype=dtype)
    h_ice = tf.constant([[1000.0]], dtype=dtype)
    N_ref = tf.constant(1.0e-3, dtype=dtype)   # MPa
    e_ref = tf.constant(0.69, dtype=dtype)
    C_c = tf.constant(0.12, dtype=dtype)
    delta = tf.constant(0.02, dtype=dtype)

    N = compute_N_MPa_tf(
        h_water_till, h_water_till_max, rho_ice, g, h_ice, N_ref, e_ref, C_c, delta
    )

    p_ice_MPa = rho_ice.numpy() * g.numpy() * 1000.0 * 1.0e-6
    N_expected = delta.numpy() * p_ice_MPa
    np.testing.assert_allclose(N.numpy()[0, 0], N_expected, rtol=1e-4)


def test_N_shape() -> None:
    """Test output shape matches input."""
    ny, nx = 5, 4
    h_water_till = tf.ones((ny, nx), dtype=dtype) * 1.0
    h_water_till_max = tf.constant(2.0, dtype=dtype)
    h_ice = tf.ones((ny, nx), dtype=dtype) * 1000.0
    N_ref = tf.constant(1.0e-3, dtype=dtype)   # MPa
    e_ref = tf.constant(0.69, dtype=dtype)
    C_c = tf.constant(0.12, dtype=dtype)
    delta = tf.constant(0.02, dtype=dtype)

    N = compute_N_MPa_tf(
        h_water_till, h_water_till_max, rho_ice, g, h_ice, N_ref, e_ref, C_c, delta
    )

    assert N.shape == (ny, nx)
