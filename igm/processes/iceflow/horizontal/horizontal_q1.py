#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import math
import tensorflow as tf
from typing import Tuple
from omegaconf import DictConfig

from .horizontal import HorizontalDiscr


class Q1Discr(HorizontalDiscr):
    """Bilinear quadrilateral (Q1) finite-element discretization.

    Implements standard Q1 finite elements with 2×2 Gaussian quadrature. Each
    quadrilateral cell has four evaluation points located at Gauss points
    (ξ, η) = (0.5 ± 1/√3, 0.5 ± 1/√3) in the reference element [0,1]².

    Gradients are computed by:
        1. Computing edge derivatives: dX/dx along south/north edges,
           dX/dy along west/east edges.
        2. Interpolating these edge derivatives to Gauss points using
           bilinear basis functions: (1-η)·south + η·north for dX/dx,
           (1-ξ)·west + ξ·east for dX/dy.

    Interpolation uses the standard bilinear formula:
        X(ξ,η) = (1-ξ)(1-η)·X_sw + ξ(1-η)·X_se
               + (1-ξ)η·X_nw + ξη·X_ne

    Attributes:
        w_h: Quadrature weights [0.25, 0.25, 0.25, 0.25] for equal weighting
            of the 4 Gauss points.
        gp_xi: ξ-coordinates of Gauss points in reference element [0,1].
        gp_eta: η-coordinates of Gauss points in reference element [0,1].
    """

    def _compute_discr(self, cfg: DictConfig) -> None:
        self.w_h = tf.constant([0.25, 0.25, 0.25, 0.25], self.dtype)

        gp_a = 0.5 - 0.5 / math.sqrt(3.0)
        gp_b = 0.5 + 0.5 / math.sqrt(3.0)

        self.gp_xi = tf.constant([gp_a, gp_a, gp_b, gp_b], self.dtype)
        self.gp_eta = tf.constant([gp_a, gp_b, gp_a, gp_b], self.dtype)

    @tf.function
    def grad_h(self, X: tf.Tensor, dX: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # Extract corner values
        X_sw = X[..., :-1, :-1]
        X_se = X[..., :-1, 1:]
        X_nw = X[..., 1:, :-1]
        X_ne = X[..., 1:, 1:]

        # Per-sample cell spacing: [B, Ny-1, Nx-1]
        dx_cell = dX[:, 1:, 1:]
        dx_inv = 1.0 / dx_cell

        # Broadcast over any extra axes between batch and space, e.g. Nz
        extra_ndims = len(X_sw.shape) - len(dx_inv.shape)
        for _ in range(extra_ndims):
            dx_inv = dx_inv[:, tf.newaxis, ...]

        # Edge derivatives
        dXdx_s = (X_se - X_sw) * dx_inv
        dXdx_n = (X_ne - X_nw) * dx_inv
        dXdy_w = (X_nw - X_sw) * dx_inv
        dXdy_e = (X_ne - X_se) * dx_inv

        # Reshape for broadcasting
        ndim = len(X_sw.shape) - 1
        eta = tf.reshape(self.gp_eta, [1, 4] + [1] * ndim)
        xi = tf.reshape(self.gp_xi, [1, 4] + [1] * ndim)

        # Vectorized interpolation
        dXdx = (1.0 - eta) * dXdx_s[:, tf.newaxis, ...] + eta * dXdx_n[
            :, tf.newaxis, ...
        ]
        dXdy = (1.0 - xi) * dXdy_w[:, tf.newaxis, ...] + xi * dXdy_e[:, tf.newaxis, ...]

        return dXdx, dXdy

    @tf.function
    def interp_h(self, X: tf.Tensor) -> tf.Tensor:
        # Extract corner values
        X_sw = X[..., :-1, :-1]
        X_se = X[..., :-1, 1:]
        X_nw = X[..., 1:, :-1]
        X_ne = X[..., 1:, 1:]

        # Reshape for broadcasting
        ndim = len(X_sw.shape) - 1
        xi = tf.reshape(self.gp_xi, [1, 4] + [1] * ndim)
        eta = tf.reshape(self.gp_eta, [1, 4] + [1] * ndim)

        # Bilinear interpolation
        return (
            (1.0 - xi) * (1.0 - eta) * X_sw[:, tf.newaxis, ...]
            + xi * (1.0 - eta) * X_se[:, tf.newaxis, ...]
            + (1.0 - xi) * eta * X_nw[:, tf.newaxis, ...]
            + xi * eta * X_ne[:, tf.newaxis, ...]
        )
