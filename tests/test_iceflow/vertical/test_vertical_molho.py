#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import os
import pytest
import numpy as np
from omegaconf import DictConfig

import igm
from igm.common.runner.configuration.loader import load_yaml_recursive
from igm.processes.iceflow.vertical import VerticalDiscrs
from igm.processes.iceflow.vertical.utils import compute_gauss_quad
from igm.processes.iceflow.vertical.vertical_molho import MOLHODiscr


@pytest.fixture
def cfg() -> DictConfig:
    cfg = load_yaml_recursive(os.path.join(igm.__path__[0], "conf"))
    cfg.processes.iceflow.numerics.Nz = 2
    cfg.processes.iceflow.physics.viscosity.exponent = 3.0
    return cfg


def test_registry_name() -> None:
    assert "molho" in VerticalDiscrs
    assert VerticalDiscrs["molho"] == MOLHODiscr


def test_registry_instantiate(cfg: DictConfig) -> None:
    assert isinstance(VerticalDiscrs["molho"](cfg), MOLHODiscr)


def test_nz(cfg: DictConfig) -> None:
    cfg.processes.iceflow.numerics.Nz = 3
    with pytest.raises(ValueError):
        MOLHODiscr(cfg)


def test_matrices_shapes(cfg: DictConfig) -> None:
    discr = MOLHODiscr(cfg)

    assert discr.w.shape == (5,)
    assert discr.zeta.shape == (5,)
    assert discr.V_q.shape == (5, 2)
    assert discr.V_q_grad.shape == (5, 2)
    assert discr.V_b.shape == (2,)
    assert discr.V_s.shape == (2,)
    assert discr.V_bar.shape == (2,)
    assert discr.V_int.shape == (2, 2)
    assert discr.V_corr_b.shape == (2, 2)
    assert discr.V_corr_s.shape == (2, 2)
    assert discr.V_const.shape == (2,)


def test_matrices_properties(cfg: DictConfig) -> None:
    discr = MOLHODiscr(cfg)

    sum_rows_V_q = np.sum(discr.V_q.numpy(), axis=1)
    sum_rows_V_q_grad = np.sum(discr.V_q_grad.numpy(), axis=1)

    np.testing.assert_allclose(sum_rows_V_q, 1.0)
    np.testing.assert_allclose(sum_rows_V_q_grad, 0.0)


def test_matrices_example(cfg: DictConfig) -> None:
    cfg.processes.iceflow.physics.viscosity.exponent = 0.0
    discr = MOLHODiscr(cfg)

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

    z_quad, w_quad = compute_gauss_quad(order=5)
    z = z_quad.numpy()
    w = w_quad.numpy()

    w_expected = w
    zeta_expected = z
    V_q_expected = np.stack([1.0 - z, z], axis=1)
    V_q_grad_expected = np.tile(np.array([-1.0, 1.0]), (len(z), 1))
    V_b_expected = np.array([1.0, 0.0])
    V_s_expected = np.array([0.0, 1.0])
    V_bar_expected = np.array([np.sum(w * (1.0 - z)), np.sum(w * z)])
    V_int_expected = np.array([[0.0, 0.0], [0.5, 0.5]])
    V_corr_b_expected = np.array([[0.0, 0.0], [-0.5, 0.5]])
    V_corr_s_expected = np.array([[0.0, 0.0], [-0.5, 0.5]])
    V_const_expected = np.array([1.0, 1.0])

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
