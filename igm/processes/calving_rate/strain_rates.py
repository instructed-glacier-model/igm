#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Principal horizontal strain rates next to an ice front.

No difference crosses the front into the ice-free cells, whose velocity is
not ice velocity. Along each axis, a derivative is central where both edge
neighbours carry velocity. Where only one of them does (at the front), it is
the second-order one-sided difference over the two nodes behind, or the
first-order one with a single node behind; without such a neighbour it is
zero. PISM (``StressBalance.cc``) uses the first-order one-sided difference,
which is the strain rate half a cell inside the ice: at the front of a
spreading shelf, where the strain rate decreases towards the front, it
overestimates the strain rate at the front node (a few tenths of a percent of
the von Mises rate on the Albrecht et al. (2011) shelf at 5 km).
"""

from typing import Tuple

import tensorflow as tf


def _derivative(
    f: Tuple[tf.Tensor, ...], w: Tuple[tf.Tensor, ...], dx: tf.Tensor
) -> tf.Tensor:
    """d/ds of ``f`` along one axis from the values and masks at s - 2 .. s + 2."""
    f_mm, f_m, f_0, f_p, f_pp = f
    w_mm, w_m, w_p, w_pp = w
    central = (f_p - f_m) / (2.0 * dx)
    backward = tf.where(
        w_mm, (3.0 * f_0 - 4.0 * f_m + f_mm) / (2.0 * dx), (f_0 - f_m) / dx
    )
    forward = tf.where(
        w_pp, (-3.0 * f_0 + 4.0 * f_p - f_pp) / (2.0 * dx), (f_p - f_0) / dx
    )
    return tf.where(
        w_m & w_p,
        central,
        tf.where(w_m, backward, tf.where(w_p, forward, tf.zeros_like(f_0))),
    )


def _derivatives(
    f: tf.Tensor, mask: tf.Tensor, dx: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Masked d/dx and d/dy of ``f`` (rows are y, columns x)."""
    fp = tf.pad(f, [[2, 2], [2, 2]])
    mp = tf.pad(mask, [[2, 2], [2, 2]], constant_values=False)
    x = [fp[2:-2, k : k + f.shape[1]] for k in range(5)]
    y = [fp[k : k + f.shape[0], 2:-2] for k in range(5)]
    mx = [mp[2:-2, k : k + f.shape[1]] for k in (0, 1, 3, 4)]
    my = [mp[k : k + f.shape[0], 2:-2] for k in (0, 1, 3, 4)]
    return _derivative(x, mx, dx), _derivative(y, my, dx)


@tf.function(jit_compile=True)
def principal_strain_rates(
    ubar: tf.Tensor, vbar: tf.Tensor, mask: tf.Tensor, dx: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Eigenvalues ``e1 >= e2`` (1/yr) of the horizontal strain-rate tensor.

    Evaluated on the cells of the bool ``mask`` (the nodes carrying
    velocity) from their masked neighbours; zero elsewhere.
    """
    mask = tf.cast(mask, tf.bool)
    ux, uy = _derivatives(ubar, mask, dx)
    vx, vy = _derivatives(vbar, mask, dx)
    mean = 0.5 * (ux + vy)
    radius = tf.sqrt(tf.square(0.5 * (ux - vy)) + tf.square(0.5 * (uy + vx)))
    zero = tf.zeros_like(ubar)
    return tf.where(mask, mean + radius, zero), tf.where(mask, mean - radius, zero)
