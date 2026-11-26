#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Tuple

from .bc import BoundaryCondition, TV


class NoInflow(BoundaryCondition):
    """Open boundary: outflow allowed, inflow suppressed (normal component only)."""

    def apply(self, U: TV, V: TV) -> Tuple[TV, TV]:
        # --- 1) copy interior values to the boundary (Neumann-ish) to avoid a "dam"
        # y boundaries
        U = tf.concat([U[:, :, 1:2, :], U[:, :, 1:-1, :], U[:, :, -2:-1, :]], axis=2)
        V = tf.concat([V[:, :, 1:2, :], V[:, :, 1:-1, :], V[:, :, -2:-1, :]], axis=2)
        # x boundaries
        U = tf.concat([U[:, :, :, 1:2], U[:, :, :, 1:-1], U[:, :, :, -2:-1]], axis=3)
        V = tf.concat([V[:, :, :, 1:2], V[:, :, :, 1:-1], V[:, :, :, -2:-1]], axis=3)

        # --- 2) no-inflow on NORMAL component
        # west boundary (x=0): outward normal is -x, so outflow means U <= 0
        U_w = tf.minimum(U[:, :, :, 0:1], 0.0)
        # east boundary (x=end): outward normal is +x, so outflow means U >= 0
        U_e = tf.maximum(U[:, :, :, -1:], 0.0)
        U = tf.concat([U_w, U[:, :, :, 1:-1], U_e], axis=3)

        # south boundary (y=0): outward normal is -y, so outflow means V <= 0
        V_s = tf.minimum(V[:, :, 0:1, :], 0.0)
        # north boundary (y=end): outward normal is +y, so outflow means V >= 0
        V_n = tf.maximum(V[:, :, -1:, :], 0.0)
        V = tf.concat([V_s, V[:, :, 1:-1, :], V_n], axis=2)

        return U, V
