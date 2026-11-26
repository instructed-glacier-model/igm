#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from typing import Tuple

import tensorflow as tf
from omegaconf import DictConfig

from .horizontal import HorizontalDiscr


class MACDiscr(HorizontalDiscr):
    """Marker-and-Cell (MAC) staggered grid discretization.

    Implements the classic MAC staggered grid approach where different gradient
    components are evaluated at different cell edges. This arrangement naturally
    satisfies discrete conservation properties and avoids checker-board
    instabilities common in collocated schemes.

    Gradient evaluation locations:
        - Eval 0: dX/dx at south edge midpoint, dX/dy at west edge midpoint
        - Eval 1: dX/dx at north edge midpoint, dX/dy at east edge midpoint

    Gradients are computed as simple edge differences:
        - dX/dx = (X_east - X_west) / dx at horizontal edges
        - dX/dy = (X_north - X_south) / dx at vertical edges

    Interpolation computes cell-center values as the average of four corners:
        X_center = (X_sw + X_se + X_nw + X_ne) / 4

    The same center value is returned twice (for both evaluation sets) to
    maintain consistent tensor shapes with other discretizations.

    Attributes:
        w_h: Quadrature weights [0.5, 0.5] for combining the two evaluation sets.
    """

    def _compute_discr(self, cfg: DictConfig) -> None:
        self.w_h = tf.constant([0.5, 0.5], self.dtype)
        self._one_quarter = tf.constant(0.25, self.dtype)

    @tf.function
    def grad_h(self, X: tf.Tensor, dX: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # Extract corner values
        X_sw = X[..., :-1, :-1]
        X_se = X[..., :-1, 1:]
        X_nw = X[..., 1:, :-1]
        X_ne = X[..., 1:, 1:]

        dx_inv = 1.0 / dX[0, 1:, 1:]

        # Eval 0: south edge for dX/dx, west edge for dX/dy
        dXdx_0 = (X_se - X_sw) * dx_inv
        dXdy_0 = (X_nw - X_sw) * dx_inv

        # Eval 1: north edge for dX/dx, east edge for dX/dy
        dXdx_1 = (X_ne - X_nw) * dx_inv
        dXdy_1 = (X_ne - X_se) * dx_inv

        return tf.stack([dXdx_0, dXdx_1], axis=1), tf.stack([dXdy_0, dXdy_1], axis=1)

    @tf.function
    def interp_h(self, X: tf.Tensor) -> tf.Tensor:
        X_c = (
            X[..., :-1, :-1] + X[..., :-1, 1:] + X[..., 1:, :-1] + X[..., 1:, 1:]
        ) * self._one_quarter

        return tf.stack([X_c, X_c], axis=1)
