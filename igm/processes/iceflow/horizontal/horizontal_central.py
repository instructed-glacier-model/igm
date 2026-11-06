#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Tuple
from omegaconf import DictConfig

from .horizontal import HorizontalDiscr
from igm.utils.stag.stag import stag4h
from igm.utils.grad.grad import grad_xy


class CentralDiscr(HorizontalDiscr):
    """Central difference discretization on a cell-centered grid.

    Implements standard central finite differences with values averaged to
    cell centers. This is the simplest discretization scheme, using a single
    evaluation point per cell located at the cell center.

    Gradients are computed via the grad_xy utility function:
        1. Compute differences: dX/dx = (X[i+1] - X[i]) / dx at east edges
        2. Average to cell centers using staggered grid operators

    Interpolation uses the stag4h utility to average the four corner values
    to the cell center:
        X_center = (X_sw + X_se + X_nw + X_ne) / 4

    This scheme has the lowest computational cost and memory footprint, making
    it suitable for problems where gradient accuracy within cells is less
    critical than overall computational efficiency. However, central differences
    on collocated grids can exhibit spurious checkerboard modes in the solution,
    where alternating grid points oscillate independently.

    Attributes:
        w_h: Quadrature weight [1.0] for the single cell-center evaluation point.
    """

    def _compute_discr(self, cfg: DictConfig) -> None:
        self.w_h = tf.constant([1.0], self.dtype)

    @tf.function
    def grad_h(self, X: tf.Tensor, dX: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:

        dUdx, dUdy = grad_xy(X, dX, dX, staggered_grid=True)

        # quadrature axis right after the batch: [B, ...] -> [B, 1, ...] for 2D ([B, Ny, Nx])
        # and 3D ([B, Nz, Ny, Nx]) fields alike; X[..., newaxis, :, :, :] put it before the
        # batch of a 2D field, and the energies then summed over the tiles of a batch
        return tf.expand_dims(dUdx, 1), tf.expand_dims(dUdy, 1)

    @tf.function
    def interp_h(self, X: tf.Tensor) -> tf.Tensor:
        return tf.expand_dims(stag4h(X), 1)
