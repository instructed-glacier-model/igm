#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import pytest
import tensorflow as tf

from igm.processes.iceflow.discretization.vertical.utils import (
    compute_dzeta,
    compute_zeta,
    compute_zeta_linear,
    compute_zeta_mid,
    compute_zeta_quadratic,
    compute_zetas,
)


@pytest.mark.parametrize("Nz", [2, 5, 10])
def test_zeta_linear_properties(Nz: int) -> None:
    zeta = compute_zeta_linear(Nz)

    assert zeta.shape == (Nz,)
    np.testing.assert_allclose(zeta[0], 0.0)
    np.testing.assert_allclose(zeta[-1], 1.0)
    assert tf.reduce_all(zeta[1:] > zeta[:-1])


def test_zeta_linear_exact() -> None:
    zeta = compute_zeta_linear(2)
    expected = [0.0, 1.0]
    np.testing.assert_allclose(zeta, expected)

    zeta = compute_zeta_linear(3)
    expected = [0.0, 0.5, 1.0]
    np.testing.assert_allclose(zeta, expected)

    zeta = compute_zeta_linear(5)
    expected = [0.0, 0.25, 0.5, 0.75, 1.0]
    np.testing.assert_allclose(zeta, expected)


@pytest.mark.parametrize("Nz", [2, 5, 10])
@pytest.mark.parametrize("slope_init", [1.0, 0.5, 0.0])
def test_zeta_quadratic_properties(Nz: int, slope_init: float) -> None:
    zeta = compute_zeta_quadratic(Nz, slope_init)

    assert zeta.shape == (Nz,)
    np.testing.assert_allclose(zeta[0], 0.0)
    np.testing.assert_allclose(zeta[-1], 1.0)

    if slope_init == 1.0:
        zeta_linear = compute_zeta_linear(Nz)
        np.testing.assert_allclose(zeta, zeta_linear)


def test_zeta_quadratic_exact() -> None:
    zeta = compute_zeta_quadratic(5, 1.0)
    expected = [0.0, 0.25, 0.5, 0.75, 1.0]
    np.testing.assert_allclose(zeta, expected)

    zeta = compute_zeta_quadratic(5, 0.0)
    expected = [0.0, 0.0625, 0.25, 0.5625, 1.0]
    np.testing.assert_allclose(zeta, expected)

    zeta = compute_zeta_quadratic(3, 0.5)
    expected = [0.0, 0.375, 1.0]
    np.testing.assert_allclose(zeta, expected)


@pytest.mark.parametrize("Nz", [2, 5, 10])
@pytest.mark.parametrize("slope_init", [1.0, 0.5, 0.0])
def test_zeta(Nz: int, slope_init: float) -> None:
    zeta = compute_zeta(Nz, slope_init)
    zeta_quad = compute_zeta_quadratic(Nz, slope_init)

    np.testing.assert_allclose(zeta, zeta_quad)


@pytest.mark.parametrize("Nz", [2, 5, 10])
def test_zeta_mid(Nz: int) -> None:
    zeta = compute_zeta(Nz)
    zeta_mid = compute_zeta_mid(zeta)

    assert zeta_mid.shape == (Nz - 1,)

    expected = (zeta[1:] + zeta[:-1]) / 2.0
    np.testing.assert_allclose(zeta_mid, expected)


def test_zeta_mid_one() -> None:
    zeta = compute_zeta(1)
    zeta_mid = compute_zeta_mid(zeta)

    assert zeta_mid.shape == (1,)

    expected = [0.5]
    np.testing.assert_allclose(zeta_mid, expected)


@pytest.mark.parametrize("Nz", [2, 5, 10])
def test_dzeta(Nz: int) -> None:
    zeta = compute_zeta(Nz)
    dzeta = compute_dzeta(zeta)

    assert dzeta.shape == (Nz - 1,)

    expected = zeta[1:] - zeta[:-1]
    np.testing.assert_allclose(dzeta, expected)
    np.testing.assert_allclose(tf.reduce_sum(dzeta), 1.0)


def test_dzeta_one() -> None:
    zeta = compute_zeta(1)
    dzeta = compute_dzeta(zeta)

    assert dzeta.shape == (1,)

    expected = [1.0]
    np.testing.assert_allclose(dzeta, expected)
    np.testing.assert_allclose(tf.reduce_sum(dzeta), 1.0)


@pytest.mark.parametrize("Nz", [2, 5, 10])
@pytest.mark.parametrize("slope_init", [1.0, 0.5, 0.0])
def test_compute_zetas_consistency(Nz: int, slope_init: float) -> None:
    zeta, zeta_mid, dzeta = compute_zetas(Nz, slope_init)

    assert zeta.shape == (Nz,)
    assert zeta_mid.shape == (Nz - 1,)
    assert dzeta.shape == (Nz - 1,)

    expected_zeta = compute_zeta(Nz, slope_init)
    expected_zeta_mid = compute_zeta_mid(expected_zeta)
    expected_dzeta = compute_dzeta(expected_zeta)

    np.testing.assert_allclose(zeta, expected_zeta)
    np.testing.assert_allclose(zeta_mid, expected_zeta_mid)
    np.testing.assert_allclose(dzeta, expected_dzeta)
