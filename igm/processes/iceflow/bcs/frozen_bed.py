#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Tuple

from .bc import BoundaryCondition, TV


class FrozenBed(BoundaryCondition):
    """Frozen bed boundary condition enforcing zero basal velocity."""

    def __init__(self, V_b: tf.Tensor):
        """Initialize weights to enforce zero basal velocity."""
        if V_b[0] == 0:
            raise ValueError(f"❌ The frozen bed BC requires V_b ≠ 0.")
        self.weights = -V_b[1:] / V_b[0]

    def apply(self, U: TV, V: TV) -> Tuple[TV, TV]:
        """Apply frozen bed condition by enforcing zero basal velocity."""
        U0 = tf.einsum("i,bijk->bjk", self.weights, U[:, 1:, :, :])
        V0 = tf.einsum("i,bijk->bjk", self.weights, V[:, 1:, :, :])

        U0 = tf.expand_dims(U0, axis=1)
        V0 = tf.expand_dims(V0, axis=1)

        U = tf.concat([U0, U[:, 1:, :, :]], axis=1)
        V = tf.concat([V0, V[:, 1:, :, :]], axis=1)

        return U, V
