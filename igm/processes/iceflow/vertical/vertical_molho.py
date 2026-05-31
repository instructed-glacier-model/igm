#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from omegaconf import DictConfig
from typing import Callable, Tuple

from .utils import (
    compute_matrices,
    compute_gauss_quad,
    compute_V_int_nodal,
    compute_V_corr_nodal,
)
from .utils_molho import (
    phi_bed,
    phi_surf,
    grad_phi_bed,
    grad_phi_surf,
    int_phi_bed,
    int_phi_surf,
)
from .vertical import VerticalDiscr


class MOLHODiscr(VerticalDiscr):
    """MOno-Layer Higher-Order (MOLHO) vertical discretization (two layers)."""

    def _compute_discr(
        self, cfg: DictConfig
    ) -> Tuple[Callable[[tf.Tensor], tf.Tensor], ...]:
        """Compute MOLHO discretization matrices. Returns basis functions."""

        cfg_numerics = cfg.processes.iceflow.numerics
        cfg_physics = cfg.processes.iceflow.physics

        Nz = cfg_numerics.Nz
        n = cfg_physics.viscosity.exponent

        if Nz != 2:
            raise ValueError("❌ MOLHO vertical basis only supports Nz=2.")

        x_quad, w_quad = compute_gauss_quad(order=5)

        basis_fct = (
            lambda z: phi_bed(z, n),
            lambda z: phi_surf(z, n),
        )
        basis_fct_grad = (
            lambda z: grad_phi_bed(z, n),
            lambda z: grad_phi_surf(z, n),
        )
        basis_fct_int = (
            lambda z: int_phi_bed(z, n),
            lambda z: int_phi_surf(z, n),
        )

        V_q, V_q_grad, V_b, V_s, V_bar = compute_matrices(
            basis_fct,
            basis_fct_grad,
            x_quad,
            w_quad,
        )

        zeta_nodes = tf.constant([0.0, 1.0], dtype=x_quad.dtype)
        V_int = compute_V_int_nodal(basis_fct_int, zeta_nodes)
        V_corr_b, V_corr_s = compute_V_corr_nodal(basis_fct, basis_fct_int, zeta_nodes)

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

        return basis_fct
