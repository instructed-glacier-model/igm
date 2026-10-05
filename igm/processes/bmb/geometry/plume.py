#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Basal slope and grounding-line depth for the plume parametrisations.

A buoyant plume rises from the grounding line along the ice base, so its
melt depends on the local basal slope and on the depth z_gl of the
grounding line it starts from (Lazeroms et al., 2018, 2019). As in PICOP
(Pelle et al., 2019), z_gl is carried along the ice flow from the grounding
line. Following the upstream scheme of Kori-ULB, stabilised by a small
diffusion, it solves

    u . grad(z_gl) - epsilon lap(z_gl) = 0   on the shelf,

with z_gl the bed depth (clipped at 0) at grounded neighbours and 0 at the
other neighbours (open ocean, holes). First-order upwinding makes every
update a convex combination of neighbour values, so the Jacobi iteration
used here is monotone and reaches the solution after about as many sweeps as
the longest flow path across the shelf. As in Pelle et al. (2019), z_gl is
then lowered to the local ice draft where it is shallower; a node fed from
at or above sea level has no plume source and gets z_gl = 0.
"""

from typing import Tuple

import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State

from ..utils import EDGE, neighbours
from .geometry import Geometry

_SWEEPS = 8  # sweeps between two convergence checks


@tf.function(autograph=False, jit_compile=True)
def basal_slope(z: tf.Tensor, shelf: tf.Tensor, dx: tf.Tensor) -> tf.Tensor:
    """Sine of the slope angle of the ice base ``z`` on the shelf.

    Central differences between shelf nodes, one-sided at the shelf edges
    and 0 across an isolated node, as in Kori-ULB's ``CheckSlope1D`` but
    without its periodic wrap-around.
    """
    dx = tf.cast(dx, z.dtype)
    # Neighbours at the previous and next row (y) and column (x), as in EDGE.
    z_y0, z_y1, z_x0, z_x1 = neighbours(z, EDGE)
    in_y0, in_y1, in_x0, in_x1 = neighbours(shelf, EDGE, False)

    def derivative(z_minus, z_plus, in_minus, in_plus):
        return tf.where(
            in_minus & in_plus,
            (z_plus - z_minus) / (2.0 * dx),
            tf.where(
                in_plus,
                (z_plus - z) / dx,
                tf.where(in_minus, (z - z_minus) / dx, 0.0),
            ),
        )

    gradient = tf.sqrt(
        derivative(z_x0, z_x1, in_x0, in_x1) ** 2
        + derivative(z_y0, z_y1, in_y0, in_y1) ** 2
    )
    return tf.where(shelf, gradient / tf.sqrt(1.0 + gradient**2), 0.0)


@tf.function(autograph=False, jit_compile=True)
def _grounding_line_depth(
    z0: tf.Tensor,
    draft: tf.Tensor,
    grounded: tf.Tensor,
    shelf: tf.Tensor,
    u: tf.Tensor,
    v: tf.Tensor,
    dx: tf.Tensor,
    epsilon: float,
    tol: float,
    max_iter: int,
) -> Tuple[tf.Tensor, tf.Tensor]:
    # Neighbour weights (yr-1), in the order of EDGE: the row above is
    # upwind when v > 0, the column to the left when u > 0.
    upwind = [
        tf.maximum(v, 0.0),
        tf.maximum(-v, 0.0),
        tf.maximum(u, 0.0),
        tf.maximum(-u, 0.0),
    ]
    inside = neighbours(tf.ones_like(shelf), EDGE, False)
    weights = tf.stack(
        [
            (w / dx + epsilon / dx**2) * tf.cast(i, u.dtype)
            for w, i in zip(upwind, inside)
        ]
    )
    total = tf.reduce_sum(weights, axis=0)
    fixed = tf.where(grounded, tf.minimum(draft, 0.0), 0.0)

    def sweep(z):
        average = tf.reduce_sum(weights * tf.stack(neighbours(z, EDGE)), axis=0)
        return tf.where(shelf, tf.math.divide_no_nan(average, total), fixed)

    def body(z, change, it):
        y = z
        for _ in range(_SWEEPS):
            y = sweep(y)
        return y, tf.reduce_max(tf.abs(y - z)), it + _SWEEPS

    z, _, _ = tf.while_loop(
        lambda z, change, it: (change > tol) & (it < max_iter),
        body,
        (tf.where(shelf, z0, fixed), tf.constant(float("inf"), z0.dtype), 0),
    )
    source = tf.where(z < 0.0, tf.minimum(z, draft), 0.0)
    return tf.where(shelf, source, 0.0), z


def grounding_line_depth(cfg: DictConfig, state: State, geom: Geometry) -> tf.Tensor:
    """Depth of the grounding line feeding each shelf node (m, <= 0).

    Publishes it as ``state.grounding_line_depth`` (0 where there is no
    source below sea level). The solution is lowered to the ice draft after
    solving; the unclipped solution, a fixed point of the
    iteration, is kept in ``state.grounding_line_iterate`` as the initial
    guess of the next call, which then converges in a few sweeps.
    """
    p = cfg.processes.bmb.grounding_line_depth
    dtype = geom.draft.dtype
    z0 = getattr(state, "grounding_line_iterate", geom.draft)
    state.grounding_line_depth, state.grounding_line_iterate = _grounding_line_depth(
        tf.cast(z0, dtype),
        geom.draft,
        geom.grounded,
        geom.shelf,
        tf.cast(state.ubar, dtype),
        tf.cast(state.vbar, dtype),
        tf.cast(state.dx, dtype),
        float(p.epsilon),
        float(p.tol),
        int(p.max_iter),
    )
    return state.grounding_line_depth
