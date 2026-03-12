#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Tuple
from .bc import BoundaryCondition, TV

class DirichletBoundary(BoundaryCondition):
    """Dirichlet boundary condition on specified edges."""

    def __init__(self, left: float = None, right: float = None,
                 top: float = None, bottom: float = None):
        """
        Parameters
        ----------
        left : float, optional
            Value to enforce on left edge (x=0). None means no condition applied.
        right : float, optional
            Value to enforce on right edge (x=-1).
        top : float, optional
            Value to enforce on top edge (y=-1).
        bottom : float, optional
            Value to enforce on bottom edge (y=0).
        """
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom

    def apply(self, U: TV, V: TV) -> Tuple[TV, TV]:
        """Apply Dirichlet boundary conditions on specified edges.
        
        U, V shape: [batch, Nz, Ny, Nx]
        """
        if self.left is not None:
            fill = tf.fill(tf.shape(U[:, :, :, :1]), tf.cast(self.left, U.dtype))
            U = tf.concat([fill, U[:, :, :, 1:]], axis=3)
            fill = tf.fill(tf.shape(V[:, :, :, :1]), tf.cast(self.left, V.dtype))
            V = tf.concat([fill, V[:, :, :, 1:]], axis=3)

        if self.right is not None:
            fill = tf.fill(tf.shape(U[:, :, :, -1:]), tf.cast(self.right, U.dtype))
            U = tf.concat([U[:, :, :, :-1], fill], axis=3)
            fill = tf.fill(tf.shape(V[:, :, :, -1:]), tf.cast(self.right, V.dtype))
            V = tf.concat([V[:, :, :, :-1], fill], axis=3)

        if self.bottom is not None:
            fill = tf.fill(tf.shape(U[:, :, :1, :]), tf.cast(self.bottom, U.dtype))
            U = tf.concat([fill, U[:, :, 1:, :]], axis=2)
            fill = tf.fill(tf.shape(V[:, :, :1, :]), tf.cast(self.bottom, V.dtype))
            V = tf.concat([fill, V[:, :, 1:, :]], axis=2)

        if self.top is not None:
            fill = tf.fill(tf.shape(U[:, :, -1:, :]), tf.cast(self.top, U.dtype))
            U = tf.concat([U[:, :, :-1, :], fill], axis=2)
            fill = tf.fill(tf.shape(V[:, :, -1:, :]), tf.cast(self.top, V.dtype))
            V = tf.concat([V[:, :, :-1, :], fill], axis=2)

        return U, V