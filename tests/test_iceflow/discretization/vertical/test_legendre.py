#!/usr/bin/env python3
# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import os

import numpy as np
import pytest
from numpy.polynomial.legendre import Legendre
from omegaconf import DictConfig

import igm
from igm.common.runner.configuration.loader import load_yaml_recursive
from igm.processes.iceflow.discretization.vertical import VerticalDiscrs
from igm.processes.iceflow.discretization.vertical.legendre import LegendreDiscr
from igm.processes.iceflow.discretization.vertical.utils import compute_gauss_quad


@pytest.fixture
def cfg() -> DictConfig:
    cfg = load_yaml_recursive(os.path.join(igm.__path__[0], "conf"))
    cfg.processes.iceflow.numerics.Nz = 2
    return cfg


def test_registry_name() -> None:
    assert "legendre" in VerticalDiscrs
    assert VerticalDiscrs["legendre"] == LegendreDiscr


def test_registry_instantiate(cfg: DictConfig) -> None:
    assert isinstance(VerticalDiscrs["legendre"](cfg), LegendreDiscr)


@pytest.mark.parametrize("Nz", [2, 5, 10])
def test_matrices_shapes(cfg: DictConfig, Nz: int) -> None:
    cfg.processes.iceflow.numerics.Nz = Nz
    discr = LegendreDiscr(cfg)

    assert discr.w.shape == (Nz,)
    assert discr.zeta.shape == (Nz,)
    assert discr.V_q.shape == (Nz, Nz)
    assert discr.V_q_grad.shape == (Nz, Nz)
    assert discr.V_b.shape == (Nz,)
    assert discr.V_s.shape == (Nz,)
    assert discr.V_bar.shape == (Nz,)
    assert discr.V_int.shape == (Nz, Nz)
    assert discr.V_corr_b.shape == (Nz, Nz)
    assert discr.V_corr_s.shape == (Nz, Nz)
    assert discr.V_const.shape == (Nz,)


@pytest.mark.parametrize("Nz", [2, 5, 6])
def test_orthogonality(cfg: DictConfig, Nz: int) -> None:

    cfg.processes.iceflow.numerics.Nz = Nz
    discr = LegendreDiscr(cfg)

    V_q = discr.V_q.numpy()
    w = discr.w.numpy()
    int_ij_computed = (V_q * w[:, None]).T @ V_q

    diag_expected = np.array([1.0 / (2 * k + 1) for k in range(Nz)])
    int_ij_expected = np.diag(diag_expected)

    rtol = 1e-5
    atol = 1e-7

    np.testing.assert_allclose(int_ij_computed, int_ij_expected, rtol, atol)


@pytest.mark.parametrize("Nz", [2, 5, 6])
def test_b(cfg: DictConfig, Nz: int) -> None:
    cfg.processes.iceflow.numerics.Nz = Nz
    discr = LegendreDiscr(cfg)

    V_b_computed = discr.V_b.numpy()
    V_b_expected = np.where(np.arange(Nz) % 2 == 0, 1.0, -1.0)

    rtol = 1e-5
    atol = 1e-7

    np.testing.assert_allclose(V_b_computed, V_b_expected, rtol, atol)


@pytest.mark.parametrize("Nz", [2, 5, 6])
def test_s(cfg: DictConfig, Nz: int) -> None:

    cfg.processes.iceflow.numerics.Nz = Nz
    discr = LegendreDiscr(cfg)

    V_s_computed = discr.V_s.numpy()
    V_s_expected = np.ones(Nz)

    rtol = 1e-5
    atol = 1e-7

    np.testing.assert_allclose(V_s_computed, V_s_expected, rtol, atol)


@pytest.mark.parametrize("Nz", [2, 5, 6])
def test_bar(cfg: DictConfig, Nz: int) -> None:

    cfg.processes.iceflow.numerics.Nz = Nz
    discr = LegendreDiscr(cfg)

    V_bar_computed = discr.V_bar.numpy()
    V_bar_expected = np.zeros(Nz)
    V_bar_expected[0] = 1.0

    rtol = 1e-5
    atol = 1e-7

    np.testing.assert_allclose(V_bar_computed, V_bar_expected, rtol, atol)


def test_matrices_example(cfg: DictConfig) -> None:
    discr = LegendreDiscr(cfg)

    w_computed = discr.w.numpy()
    zeta_computed = discr.zeta.numpy()
    V_q_computed = discr.V_q.numpy()
    V_q_grad_computed = discr.V_q_grad.numpy()
    V_b_computed = discr.V_b.numpy()
    V_s_computed = discr.V_s.numpy()
    V_bar_computed = discr.V_bar.numpy()
    V_int_computed = discr.V_int.numpy()
    V_corr_b_computed = discr.V_corr_b.numpy()
    V_corr_s_computed = discr.V_corr_s.numpy()
    V_const_computed = discr.V_const.numpy()

    a = 1.0 / np.sqrt(3.0)
    z1 = (1.0 - a) * 0.5
    z2 = (1.0 + a) * 0.5

    w_expected = np.array([0.5, 0.5])
    zeta_expected = np.array([z1, z2])
    V_q_expected = np.array([[1.0, -a], [1.0, a]])
    V_q_grad_expected = np.array([[0.0, 2.0], [0.0, 2.0]])
    V_b_expected = np.array([1.0, -1.0])
    V_s_expected = np.array([1.0, 1.0])
    V_bar_expected = np.array([1.0, 0.0])
    V_int_expected = np.array([[0.5, -1.0 / 6.0], [0.5, 0.0]])
    V_corr_b_expected = np.array([[0.0, 2.0 / 3.0], [0.0, 0.5]])
    V_corr_s_expected = np.array([[0.0, 1.0 / 3.0], [0.0, 0.5]])
    V_const_expected = np.array([1.0, 0.0])

    rtol = 1e-5
    atol = 1e-7

    np.testing.assert_allclose(w_computed, w_expected, rtol, atol)
    np.testing.assert_allclose(zeta_computed, zeta_expected, rtol, atol)
    np.testing.assert_allclose(V_q_computed, V_q_expected, rtol, atol)
    np.testing.assert_allclose(V_q_grad_computed, V_q_grad_expected, rtol, atol)
    np.testing.assert_allclose(V_b_computed, V_b_expected, rtol, atol)
    np.testing.assert_allclose(V_s_computed, V_s_expected, rtol, atol)
    np.testing.assert_allclose(V_bar_computed, V_bar_expected, rtol, atol)
    np.testing.assert_allclose(V_int_computed, V_int_expected, rtol, atol)
    np.testing.assert_allclose(V_corr_b_computed, V_corr_b_expected, rtol, atol)
    np.testing.assert_allclose(V_corr_s_computed, V_corr_s_expected, rtol, atol)
    np.testing.assert_allclose(V_const_computed, V_const_expected, rtol, atol)
