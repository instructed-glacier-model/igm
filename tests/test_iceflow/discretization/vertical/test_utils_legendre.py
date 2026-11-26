#!/usr/bin/env python3
# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file


import numpy as np
import pytest
import tensorflow as tf
from numpy.polynomial.legendre import Legendre

from igm.processes.iceflow.discretization.vertical.utils_legendre import (
    compute_basis,
    compute_basis_grad,
    compute_basis_int,
    dxdzeta,
    zeta_to_x,
)


@pytest.mark.parametrize("mode", [0, 1, 3])
def test_basis_grad_int_against_legendre(mode: int) -> None:
    z = tf.linspace(tf.constant(0.0), tf.constant(1.0), 21)
    z_inner = z[1:-1]
    x = zeta_to_x(z).numpy()
    x_inner = zeta_to_x(z_inner).numpy()

    computed_basis = compute_basis(mode)
    computed_basis_grad = compute_basis_grad(mode)
    computed_basis_int = compute_basis_int(mode)

    computed_points = computed_basis(z).numpy()
    computed_grad_points = computed_basis_grad(z_inner).numpy()
    computed_int_points = computed_basis_int(z).numpy()

    expected_basis = Legendre.basis(mode)
    expected_basis_grad = Legendre.basis(mode).deriv()
    expected_basis_int = Legendre.basis(mode).integ()

    expected_points = expected_basis(x)
    expected_grad_points = expected_basis_grad(x_inner) * dxdzeta()
    expected_int_points = 0.5 * (expected_basis_int(x) - expected_basis_int(-1.0))

    rtol = 1e-5
    atol = 1e-7

    np.testing.assert_allclose(computed_points, expected_points, rtol, atol)
    np.testing.assert_allclose(computed_grad_points, expected_grad_points, rtol, atol)
    np.testing.assert_allclose(computed_int_points, expected_int_points, rtol, atol)
