#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from omegaconf import DictConfig

from .utils import (
    compute_matrices,
    compute_V_corr_nodal,
    compute_V_int_nodal,
    compute_zetas,
)
from .utils_lagrange import compute_basis, compute_basis_grad, compute_basis_int
from .vertical import VerticalDiscr


class LagrangeDiscr(VerticalDiscr):
    """Lagrange vertical discretization (P1 hat function)."""

    def _compute_discr(self, cfg: DictConfig):
        """Compute Lagrange discretization matrices. Returns basis functions."""
        cfg_numerics = cfg.processes.iceflow.numerics

        Nz = cfg_numerics.Nz
        slope_init = 1.0 / cfg_numerics.vert_spacing

        zeta, zeta_mid, dzeta = compute_zetas(Nz, slope_init, self.dtype)

        basis_fct = [compute_basis(zeta, i) for i in range(Nz)]
        basis_fct_grad = [compute_basis_grad(zeta, i) for i in range(Nz)]
        basis_fct_int = [compute_basis_int(zeta, i) for i in range(Nz)]

        x_quad = zeta_mid
        w_quad = dzeta

        V_q, V_q_grad, V_b, V_s, V_bar = compute_matrices(
            basis_fct,
            basis_fct_grad,
            x_quad,
            w_quad,
        )

        V_int = compute_V_int_nodal(basis_fct_int, zeta)
        V_corr_b, V_corr_s = compute_V_corr_nodal(basis_fct, basis_fct_int, zeta)

        self.w = tf.cast(w_quad, self.dtype)
        self.zeta = tf.cast(x_quad, self.dtype)
        self.V_q = tf.cast(V_q, self.dtype)
        self.V_q_grad = tf.cast(V_q_grad, self.dtype)
        self.V_b = tf.cast(V_b, self.dtype)
        self.V_s = tf.cast(V_s, self.dtype)
        self.V_bar = tf.cast(V_bar, self.dtype)
        self.V_int = tf.cast(V_int, self.dtype)
        self.V_corr_b = tf.cast(V_corr_b, self.dtype)
        self.V_corr_s = tf.cast(V_corr_s, self.dtype)
        self.V_const = tf.ones(Nz, dtype=self.dtype)

        return tuple(basis_fct)
