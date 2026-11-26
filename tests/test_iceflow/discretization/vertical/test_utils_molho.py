#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import pytest
import tensorflow as tf

from igm.processes.iceflow.discretization.vertical.utils_molho import (
    grad_phi_bed,
    grad_phi_surf,
    int_phi_bed,
    int_phi_surf,
    phi_bed,
    phi_surf,
)


@pytest.mark.parametrize("exp", [0.0, 1.0, 3.0])
def test_phi_prop(exp: float) -> None:

    zeta = tf.convert_to_tensor(np.linspace(0.0, 1.0, 11))
    zeta0 = tf.convert_to_tensor(0.0)
    zeta1 = tf.convert_to_tensor(1.0)

    np.testing.assert_allclose(phi_bed(zeta0, exp), 1.0)
    np.testing.assert_allclose(phi_bed(zeta1, exp), 0.0)
    np.testing.assert_allclose(phi_surf(zeta0, exp), 0.0)
    np.testing.assert_allclose(phi_surf(zeta1, exp), 1.0)

    sum_phi = phi_bed(zeta, exp) + phi_surf(zeta, exp)
    np.testing.assert_allclose(sum_phi, tf.ones_like(sum_phi))


@pytest.mark.parametrize("exp", [0.0, 1.0, 3.0])
def test_grad_phi_prop(exp: float) -> None:
    zeta = tf.convert_to_tensor(np.linspace(0.0, 1.0, 11))

    sum_grad_phi = grad_phi_bed(zeta, exp) + grad_phi_surf(zeta, exp)
    np.testing.assert_allclose(sum_grad_phi, tf.zeros_like(sum_grad_phi))


@pytest.mark.parametrize("exp", [0.0, 1.0, 3.0])
def test_int_phi_prop(exp: float) -> None:
    zeta = tf.convert_to_tensor(np.linspace(0.0, 1.0, 11))

    sum_int_phi = int_phi_bed(zeta, exp) + int_phi_surf(zeta, exp)
    np.testing.assert_allclose(sum_int_phi, zeta)
