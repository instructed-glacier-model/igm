#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from typing import Callable, Tuple

import tensorflow as tf
from omegaconf import DictConfig

from .vertical import VerticalDiscr


class SSADiscr(VerticalDiscr):
    """Shallow Shelf Approximation (SSA) vertical discretization (single layer)."""

    def _compute_discr(
        self, cfg: DictConfig
    ) -> Tuple[Callable[[tf.Tensor], tf.Tensor], ...]:
        """Compute SSA discretization matrices. Returns basis functions."""

        cfg_numerics = cfg.processes.iceflow.numerics

        Nz = cfg_numerics.Nz

        if Nz != 1:
            raise ValueError("❌ SSA vertical basis only supports Nz=1.")

        self.w = tf.constant([1.0], dtype=self.dtype)
        self.zeta = tf.constant([0.5], dtype=self.dtype)
        self.V_q = tf.constant([[1.0]], dtype=self.dtype)
        self.V_q_grad = tf.constant([[0.0]], dtype=self.dtype)
        self.V_b = tf.constant([1.0], dtype=self.dtype)
        self.V_s = tf.constant([1.0], dtype=self.dtype)
        self.V_bar = tf.constant([1.0], dtype=self.dtype)
        self.V_int = tf.constant([[0.0]], dtype=self.dtype)
        self.V_corr_b = tf.constant([[0.0]], dtype=self.dtype)
        self.V_corr_s = tf.constant([[0.0]], dtype=self.dtype)
        self.V_const = tf.constant([1.0], dtype=self.dtype)

        basis_fct = (lambda z: tf.ones_like(z),)

        return basis_fct
