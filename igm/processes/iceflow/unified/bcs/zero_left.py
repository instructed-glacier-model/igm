#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Tuple

from .bc import BoundaryCondition, TV


class ZeroLeft(BoundaryCondition):
    """Zero boundary condition on the left edge (x=0)."""

    def apply(self, U: TV, V: TV) -> Tuple[TV, TV]:
        """Set velocity to zero at x=0 (left boundary)."""
        # U, V shape: [batch, Nz, Ny, Nx]
        U = tf.concat([tf.zeros_like(U[:, :, :, :1]), U[:, :, :, 1:]], axis=3)
        V = tf.concat([tf.zeros_like(V[:, :, :, :1]), V[:, :, :, 1:]], axis=3)
        return U, V
