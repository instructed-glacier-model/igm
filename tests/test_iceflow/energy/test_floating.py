from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from igm.processes.iceflow.energy.components.floating import (
    FloatingParams,
    cost_floating,
)


@pytest.mark.parametrize(
    ("ocean_surface", "has_front"),
    [(0.0, True), (10.0, False)],
    ids=["wet-neighbour", "land-neighbour"],
)
def test_floating_cost_integrates_exposed_cell_face(ocean_surface, has_front):
    dtype = tf.float64
    thickness = tf.constant(
        [[[100.0, 100.0, 0.0], [100.0, 100.0, 0.0]]], dtype
    )
    surface = tf.constant(
        [
            [
                [100.0, 100.0, ocean_surface],
                [100.0, 100.0, ocean_surface],
            ]
        ],
        dtype,
    )
    fields = {
        "thk": thickness,
        "usurf": surface,
        "water_level": tf.zeros_like(thickness),
        "dX": tf.constant(1000.0, dtype),
    }
    velocity_u = tf.fill((1, 1, 2, 3), tf.constant(2.0, dtype))
    velocity_v = tf.zeros_like(velocity_u)
    vertical = SimpleNamespace(
        V_q=tf.ones((1, 1), dtype),
        w=tf.ones((1,), dtype),
    )
    parameters = FloatingParams(
        rho=900.0,
        rho_water=1000.0,
        g=9.81,
        cf_eswn=(),
    )

    energy = cost_floating(
        velocity_u,
        velocity_v,
        fields,
        discr_h=None,
        discr_v=vertical,
        floating_params=parameters,
    )

    pressure = 0.5e-6 * 9.81 * 900.0 * 100.0**2
    expected_front = -pressure * 2.0 / 1000.0 if has_front else 0.0
    np.testing.assert_allclose(
        energy.numpy(),
        [[[expected_front, 0.0]]],
        rtol=1e-7,
        atol=1e-12,
    )
