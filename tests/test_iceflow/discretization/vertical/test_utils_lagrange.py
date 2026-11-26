#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import pytest
import tensorflow as tf

from igm.processes.iceflow.discretization.vertical.utils import (
    compute_zeta,
    compute_zetas,
)
from igm.processes.iceflow.discretization.vertical.utils_lagrange import (
    BasisP1,
    Element,
    compute_basis,
    compute_basis_grad,
    compute_basis_int,
)


def test_basis_p1() -> None:
    xi = tf.constant([0.0, 0.25, 0.5, 1.0])

    phi0 = BasisP1.phi0(xi)
    phi1 = BasisP1.phi1(xi)
    grad_phi0 = BasisP1.grad_phi0(xi)
    grad_phi1 = BasisP1.grad_phi1(xi)
    int_phi0 = BasisP1.int_phi0(xi)
    int_phi1 = BasisP1.int_phi1(xi)

    np.testing.assert_allclose(phi0 + phi1, 1.0)
    np.testing.assert_allclose(phi0[0], 1.0)
    np.testing.assert_allclose(phi0[-1], 0.0)
    np.testing.assert_allclose(phi1[0], 0.0)
    np.testing.assert_allclose(phi1[-1], 1.0)

    np.testing.assert_allclose(grad_phi0, -1.0)
    np.testing.assert_allclose(grad_phi1, 1.0)

    np.testing.assert_allclose(int_phi0[0], 0.0)
    np.testing.assert_allclose(int_phi1[0], 0.0)
    np.testing.assert_allclose(int_phi0[-1], 0.5)
    np.testing.assert_allclose(int_phi1[-1], 0.5)


@pytest.mark.parametrize("x0,x1", [(0.0, 1.0), (0.1, 0.7), (-2.0, -0.5)])
def test_element(x0: float, x1: float) -> None:
    x0 = tf.convert_to_tensor(x0)
    x1 = tf.convert_to_tensor(x1)

    elem = Element(x0, x1)

    np.testing.assert_allclose(elem.xi(x0), 0.0)
    np.testing.assert_allclose(elem.xi(x1), 1.0)
    np.testing.assert_allclose(elem.jac(), x1 - x0)


@pytest.mark.parametrize("Nz", [2, 5, 10])
@pytest.mark.parametrize("slope_init", [1.0, 0.5, 0.0])
def test_basis_kronecker(Nz: int, slope_init: float) -> None:
    nodes = compute_zeta(Nz, slope_init)
    basis = [compute_basis(nodes, i) for i in range(Nz)]

    for i in range(Nz):
        xi_i = nodes[i]
        basis_ij = [b(xi_i).numpy() for b in basis]
        for j, val_j in enumerate(basis_ij):
            if j == i:
                np.testing.assert_allclose(val_j, 1.0)
            else:
                np.testing.assert_allclose(val_j, 0.0)


@pytest.mark.parametrize("Nz", [2, 5, 10])
@pytest.mark.parametrize("slope_init", [1.0, 0.5, 0.0])
def test_basis_partition_unity(Nz: int, slope_init: float) -> None:
    nodes, nodes_mid, _ = compute_zetas(Nz, slope_init)
    basis = [compute_basis(nodes, i) for i in range(Nz)]

    for points in [nodes, nodes_mid]:
        sum_basis = tf.zeros_like(points)
        for b in basis:
            sum_basis = sum_basis + b(points)
        np.testing.assert_allclose(sum_basis, tf.ones_like(sum_basis))


def test_compute_basis() -> None:
    nodes = tf.constant([0.0, 0.5, 1.0])
    basis = [compute_basis(nodes, i) for i in range(len(nodes))]
    points = tf.constant([0.0, 0.25, 0.5, 0.75, 1.0])

    computed = [basis[i](points).numpy() for i in range(len(nodes))]
    expected = np.array(
        [
            [1.0, 0.5, 0.0, 0.0, 0.0],
            [0.0, 0.5, 1.0, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.5, 1.0],
        ]
    )

    np.testing.assert_allclose(computed, expected)


def test_compute_basis_grad() -> None:
    nodes = tf.constant([0.0, 0.5, 1.0])
    basis_grad = [compute_basis_grad(nodes, i) for i in range(len(nodes))]
    points = tf.constant([0.0, 0.25, 0.5, 0.75, 1.0])

    computed = [basis_grad[i](points).numpy() for i in range(len(nodes))]
    expected = np.array(
        [
            [-2.0, -2.0, 0.0, 0.0, 0.0],
            [2.0, 2.0, -2.0, -2.0, 0.0],
            [0.0, 0.0, 2.0, 2.0, 2.0],
        ]
    )

    np.testing.assert_allclose(computed, expected)


def test_compute_basis_int() -> None:
    nodes = tf.constant([0.0, 0.5, 1.0])
    basis_int = [compute_basis_int(nodes, i) for i in range(len(nodes))]
    points = tf.constant([0.0, 0.25, 0.5, 0.75, 1.0])

    computed = [basis_int[i](points).numpy() for i in range(len(nodes))]
    expected = np.array(
        [
            [0.0, 0.25 - 0.0625, 0.25, 0.25, 0.25],
            [0.0, 0.0625, 0.25, 0.4375, 0.5],
            [0.0, 0.0, 0.0, 0.0625, 0.25],
        ]
    )

    np.testing.assert_allclose(computed, expected)
