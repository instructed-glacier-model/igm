#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Tuple

from .bc import BoundaryCondition, TV


class PeriodicWE(BoundaryCondition):
    """Periodic boundary condition in west-east direction."""

    def apply(self, U: TV, V: TV) -> Tuple[TV, TV]:
        """Apply periodic boundary condition in west-east direction."""
        U = tf.concat([U[:, :, :, :-1], U[:, :, :, :1]], axis=3)
        V = tf.concat([V[:, :, :, :-1], V[:, :, :, :1]], axis=3)
        return U, V


class PeriodicWEGlobal(BoundaryCondition):
    """Periodic boundary condition in west-east direction (with patching)."""

    def __init__(self, Nx: int, Ny: int, Nz: int):
        """Save original shape to enforce periodic boundary condition (with patching)."""
        self.original_shape = (Nz, Ny, Nx)

    def apply(self, U: TV, V: TV) -> Tuple[TV, TV]:
        """Apply periodic boundary condition in west-east direction (with patching)."""
        current_shape = U.shape

        # ! Assumes reshaping with correct order (and noneven patching...) - check with Seb as this is probably wrong!
        # ! maybe make it not reshape if its already batch size of 1
        U = tf.reshape(U, [1, *self.original_shape])
        V = tf.reshape(V, [1, *self.original_shape])

        U = tf.concat([U[:, :, :, :-1], U[:, :, :, :1]], axis=3)
        V = tf.concat([V[:, :, :, :-1], V[:, :, :, :1]], axis=3)

        U = tf.reshape(U, current_shape)
        V = tf.reshape(V, current_shape)

        return U, V
