#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import os

import numpy as np
import pytest
import tensorflow as tf
from omegaconf import DictConfig

import igm
from igm.common.runner.configuration.loader import load_yaml_recursive
from igm.processes.iceflow.discretization.vertical import VerticalDiscrs
from igm.processes.iceflow.discretization.vertical.ssa import SSADiscr
from igm.processes.iceflow.discretization.vertical.utils import compute_matrices


@pytest.fixture
def cfg() -> DictConfig:
    cfg = load_yaml_recursive(os.path.join(igm.__path__[0], "conf"))
    cfg.processes.iceflow.numerics.Nz = 1
    return cfg


def test_registry_name() -> None:
    assert "ssa" in VerticalDiscrs
    assert VerticalDiscrs["ssa"] == SSADiscr


def test_registry_instantiate(cfg: DictConfig) -> None:
    assert isinstance(VerticalDiscrs["ssa"](cfg), SSADiscr)


def test_nz(cfg: DictConfig) -> None:
    cfg.processes.iceflow.numerics.Nz = 2
    with pytest.raises(ValueError):
        SSADiscr(cfg)


def test_matrices_shapes(cfg: DictConfig) -> None:
    discr = SSADiscr(cfg)

    assert discr.w.shape == (1,)
    assert discr.zeta.shape == (1,)
    assert discr.V_q.shape == (1, 1)
    assert discr.V_q_grad.shape == (1, 1)
    assert discr.V_b.shape == (1,)
    assert discr.V_s.shape == (1,)
    assert discr.V_bar.shape == (1,)
    assert discr.V_int.shape == (1, 1)
    assert discr.V_corr_b.shape == (1, 1)
    assert discr.V_corr_s.shape == (1, 1)
    assert discr.V_const.shape == (1,)


def test_matrices_properties(cfg: DictConfig) -> None:
    discr = SSADiscr(cfg)

    sum_rows_V_q = np.sum(discr.V_q.numpy(), axis=1)
    sum_rows_V_q_grad = np.sum(discr.V_q_grad.numpy(), axis=1)

    np.testing.assert_allclose(sum_rows_V_q, 1.0)
    np.testing.assert_allclose(sum_rows_V_q_grad, 0.0)


def test_matrices_basis(cfg: DictConfig) -> None:
    discr = SSADiscr(cfg)

    phi0 = lambda x: tf.ones_like(x)
    grad_phi0 = lambda x: tf.zeros_like(x)

    x_quad = tf.constant([0.5])
    w_quad = tf.constant([1.0])

    V_q, V_q_grad, V_b, V_s, V_bar = compute_matrices(
        (phi0,), (grad_phi0,), x_quad, w_quad
    )

    np.testing.assert_allclose(discr.w, w_quad)
    np.testing.assert_allclose(discr.zeta, x_quad)
    np.testing.assert_allclose(discr.V_q, V_q)
    np.testing.assert_allclose(discr.V_q_grad, V_q_grad)
    np.testing.assert_allclose(discr.V_b, V_b)
    np.testing.assert_allclose(discr.V_s, V_s)
    np.testing.assert_allclose(discr.V_bar, V_bar)
    np.testing.assert_allclose(discr.V_int, [[0.0]])
    np.testing.assert_allclose(discr.V_corr_b, [[0.0]])
    np.testing.assert_allclose(discr.V_corr_s, [[0.0]])
    np.testing.assert_allclose(discr.V_const, [1.0])
