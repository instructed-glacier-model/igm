#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Unit tests for the till_storage effective-pressure parameterisation."""

import numpy as np
import tensorflow as tf

from igm.processes.subglacial_hydrology.till_storage import compute_N_MPa_tf

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
    N_ref = tf.constant(1.0e-3, dtype=dtype)  # MPa
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
    N_ref = tf.constant(1.0e-3, dtype=dtype)  # MPa
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
    N_ref = tf.constant(1.0e-3, dtype=dtype)  # MPa
    e_ref = tf.constant(0.69, dtype=dtype)
    C_c = tf.constant(0.12, dtype=dtype)
    delta = tf.constant(0.02, dtype=dtype)

    N = compute_N_MPa_tf(
        h_water_till, h_water_till_max, rho_ice, g, h_ice, N_ref, e_ref, C_c, delta
    )

    assert N.shape == (ny, nx)


def test_till_is_saturated_in_contact_with_the_ocean() -> None:
    """Floating ice and ice-free ocean saturate the till (PISM); grounded ice
    follows the ODE; ice-free land has no till water."""
    from types import SimpleNamespace

    from omegaconf import OmegaConf

    from igm.processes.subglacial_hydrology.till_storage import update_h_water_till

    cfg = OmegaConf.create(
        {
            "processes": {
                "iceflow": {"physics": {"ice_density": 910.0, "water_density": 1028.0}},
                "subglacial_hydrology": {
                    "till_storage": {
                        "h_water_till_max": 2.0,
                        "drainage_rate": 0.001,
                        "water_density": 1000.0,
                    }
                },
            }
        }
    )
    # grounded ice, floating ice, ice-free ocean, ice-free land
    state = SimpleNamespace(
        thk=tf.constant([[500.0, 200.0, 0.0, 0.0]]),
        topg=tf.constant([[-100.0, -800.0, -800.0, 50.0]]),
        water_level=tf.zeros((1, 4)),
        h_water_till=tf.constant([[0.5, 0.5, 0.5, 0.5]]),
        basal_melt_rate=tf.constant([[0.01, 0.01, 0.01, 0.01]]),
        dt=tf.constant(1.0),
    )
    h = update_h_water_till(cfg, state).numpy()[0]
    expected_grounded = 0.5 + 910.0 / 1000.0 * 0.01 - 0.001
    np.testing.assert_allclose(h, [expected_grounded, 2.0, 2.0, 0.0], rtol=1e-6)
