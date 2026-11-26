#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from typing import Callable, Tuple

import numpy as np
import tensorflow as tf
from omegaconf import DictConfig

from .utils import (
    compute_gauss_quad,
    compute_matrices,
    compute_V_corr_orthogonal,
    compute_V_int_orthogonal,
)
from .utils_legendre import compute_basis, compute_basis_grad, compute_basis_int
from .vertical import VerticalDiscr


class LegendreDiscr(VerticalDiscr):
    """Legendre vertical discretization."""

    def _compute_discr(
        self, cfg: DictConfig
    ) -> Tuple[Callable[[tf.Tensor], tf.Tensor], ...]:
        """Compute Legendre discretization matrices. Returns basis functions."""
        cfg_numerics = cfg.processes.iceflow.numerics

        Nz = cfg_numerics.Nz

        x_quad, w_quad = compute_gauss_quad(Nz)

        basis_fct = tuple(compute_basis(i) for i in range(Nz))
        basis_fct_grad = tuple(compute_basis_grad(i) for i in range(Nz))
        basis_fct_int = tuple(compute_basis_int(i) for i in range(Nz))

        V_q, V_q_grad, V_b, V_s, V_bar = compute_matrices(
            basis_fct,
            basis_fct_grad,
            x_quad,
            w_quad,
        )

        normalization = tf.constant(2 * np.arange(Nz) + 1, dtype=x_quad.dtype)
        V_int = compute_V_int_orthogonal(
            basis_fct, basis_fct_int, x_quad, w_quad, normalization
        )
        V_corr_b, V_corr_s = compute_V_corr_orthogonal(
            basis_fct, basis_fct_int, x_quad, w_quad, normalization
        )
        V_const = tf.concat([[1.0], tf.zeros(Nz - 1, dtype=x_quad.dtype)], axis=0)

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
        self.V_const = tf.cast(V_const, self.dtype)

        return basis_fct
