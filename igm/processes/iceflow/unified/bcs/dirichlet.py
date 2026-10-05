#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Optional, Sequence, Tuple, Union
from .bc import BoundaryCondition, TV

EdgeValue = Optional[Union[float, Sequence[float]]]


def _components(value: EdgeValue) -> Optional[Tuple[float, float]]:
    """``(u, v)`` enforced on an edge: a scalar sets both, a pair each one."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value), float(value)
    values = [float(v) for v in value]
    if len(values) != 2:
        raise ValueError(
            "A Dirichlet edge value must be a scalar or a pair [u, v]; "
            f"got {list(value)!r}."
        )
    return values[0], values[1]


class DirichletBoundary(BoundaryCondition):
    """Dirichlet boundary condition on specified edges.

    Note: the values set here are applied directly to the degrees of freedom (DOFs)
    of the velocity representation, not to the velocity field itself. For nodal bases
    (e.g., Lagrange, Molho, SSA) the DOFs coincide with pointwise velocity values,
    but for other (e.g., modal) bases non-zero values will not have a straightforward
    physical interpretation.
    """

    def __init__(
        self,
        left: EdgeValue = None,
        right: EdgeValue = None,
        top: EdgeValue = None,
        bottom: EdgeValue = None,
    ):
        """
        Parameters
        ----------
        left : float or [u, v], optional
            Value to enforce on left edge (x=0). None means no condition applied.
            A scalar sets both velocity components; a pair sets u and v.
        right : float or [u, v], optional
            Value to enforce on right edge (x=-1).
        top : float or [u, v], optional
            Value to enforce on top edge (y=-1).
        bottom : float or [u, v], optional
            Value to enforce on bottom edge (y=0).
        """
        self.left = _components(left)
        self.right = _components(right)
        self.top = _components(top)
        self.bottom = _components(bottom)

    def apply(self, U: TV, V: TV) -> Tuple[TV, TV]:
        """Apply Dirichlet boundary conditions on specified edges.

        U, V shape: [batch, Nz, Ny, Nx]
        """
        if self.left is not None:
            u, v = self.left
            U = tf.concat(
                [
                    tf.fill(tf.shape(U[:, :, :, :1]), tf.cast(u, U.dtype)),
                    U[:, :, :, 1:],
                ],
                axis=3,
            )
            V = tf.concat(
                [
                    tf.fill(tf.shape(V[:, :, :, :1]), tf.cast(v, V.dtype)),
                    V[:, :, :, 1:],
                ],
                axis=3,
            )

        if self.right is not None:
            u, v = self.right
            U = tf.concat(
                [
                    U[:, :, :, :-1],
                    tf.fill(tf.shape(U[:, :, :, -1:]), tf.cast(u, U.dtype)),
                ],
                axis=3,
            )
            V = tf.concat(
                [
                    V[:, :, :, :-1],
                    tf.fill(tf.shape(V[:, :, :, -1:]), tf.cast(v, V.dtype)),
                ],
                axis=3,
            )

        if self.bottom is not None:
            u, v = self.bottom
            U = tf.concat(
                [
                    tf.fill(tf.shape(U[:, :, :1, :]), tf.cast(u, U.dtype)),
                    U[:, :, 1:, :],
                ],
                axis=2,
            )
            V = tf.concat(
                [
                    tf.fill(tf.shape(V[:, :, :1, :]), tf.cast(v, V.dtype)),
                    V[:, :, 1:, :],
                ],
                axis=2,
            )

        if self.top is not None:
            u, v = self.top
            U = tf.concat(
                [
                    U[:, :, :-1, :],
                    tf.fill(tf.shape(U[:, :, -1:, :]), tf.cast(u, U.dtype)),
                ],
                axis=2,
            )
            V = tf.concat(
                [
                    V[:, :, :-1, :],
                    tf.fill(tf.shape(V[:, :, -1:, :]), tf.cast(v, V.dtype)),
                ],
                axis=2,
            )

        return U, V
