#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Flood fill, component labels and graph distances on raster masks.

All three are the fixed point of one max-plus relaxation over the four edge
neighbours of each cell,

    x  <-  where(domain, max(x, max4(x) - decrement), 0),

run on int32 values, so the result is exact for any grid size:

* :func:`reach` floods a 0/1 seed (decrement 0);
* :func:`label_components` floods the linear index + 1 (decrement 0), which
  gives every component the largest index it contains;
* :func:`graph_distance` lets a large seed value decay by one per step
  (decrement 1), which counts the edge-neighbour steps from the seed.

Instead of one cell per sweep, each iteration propagates along whole rows
and columns of the domain, in both directions, by recursive doubling: the
step over ``s`` cells, ``x[i] <- max(x[i], x[i - s] - s * decrement)``, is
applied where cells ``i - s`` to ``i`` all lie in the domain, for s = 1, 2,
4, ... A straight run of any length is therefore covered in one iteration,
and the loop stops after about as many iterations as the paths have turns.
Every step is a valid relaxation, and the one-cell steps are among them, so
the fixed point is the same as that of the one-cell sweeps.
"""

from typing import Tuple, Union

import tensorflow as tf

_FAR = 2**30  # seed value of a distance, larger than any graph distance


def _shift(
    x: tf.Tensor, step: tf.Tensor, axis: int, fill: Union[int, bool]
) -> tf.Tensor:
    """``x[i - step]`` along ``axis`` of a rank-3 tensor, ``fill`` entering.

    ``step`` may be a tensor, so the shape need not be static.
    """
    size = tf.shape(x)[axis]
    index = tf.range(size)
    outside = tf.where(step > 0, index < step, index >= size + step)
    shape = [1, 1, 1]
    shape[axis] = -1
    return tf.where(tf.reshape(outside, shape), fill, tf.roll(x, step, axis))


def _line_scans(x: tf.Tensor, domain: tf.Tensor, decrement: tf.Tensor) -> tf.Tensor:
    """Propagate along the rows and columns of ``domain``, both ways."""
    for axis in (1, 2):
        size = tf.shape(x)[axis]
        for sign in (1, -1):

            def double(s, x, linked):
                reached = _shift(x, sign * s, axis, 0) - decrement * s
                x = tf.where(linked, tf.maximum(x, reached), x)
                return 2 * s, x, linked & _shift(linked, sign * s, axis, False)

            linked = domain & _shift(domain, tf.constant(sign), axis, False)
            _, x, _ = tf.while_loop(
                lambda s, x, linked: s < size, double, (tf.constant(1), x, linked)
            )
    return x


@tf.function(autograph=False, jit_compile=True)
def propagate(
    x0: tf.Tensor, domain: tf.Tensor, decrement: Union[int, Tuple[int, ...]]
) -> tf.Tensor:
    """Fixed point of ``x <- where(domain, max(x, max4(x) - decrement), 0)``.

    ``x0`` is a non-negative int32 field (0 = no value) and ``domain`` a bool
    mask, both of shape (ny, nx) or (channels, ny, nx). ``decrement`` is 0 or
    1, or a tuple with one value per channel.
    """
    squeeze = x0.shape.rank == 2
    if squeeze:
        x0, domain = x0[tf.newaxis], domain[tf.newaxis]
    domain = tf.cast(domain, tf.bool)
    decrement = tf.reshape(tf.constant(decrement, tf.int32), [-1, 1, 1])

    def body(x, changed):
        y = _line_scans(x, domain, decrement)
        return y, tf.reduce_any(tf.not_equal(y, x))

    x, _ = tf.while_loop(
        lambda x, changed: changed,
        body,
        (tf.where(domain, x0, tf.zeros_like(x0)), tf.constant(True)),
        maximum_iterations=tf.size(x0),
    )
    return x[0] if squeeze else x


def reach(seed: tf.Tensor, domain: tf.Tensor) -> tf.Tensor:
    """Cells of the bool ``domain`` connected to ``seed`` through cell edges."""
    return propagate(tf.cast(tf.cast(seed, tf.bool), tf.int32), domain, 0) > 0


def component_index(domain: tf.Tensor) -> tf.Tensor:
    """Seed of :func:`label_components`: the linear index + 1 (int32)."""
    return tf.reshape(tf.range(1, tf.size(domain) + 1), tf.shape(domain))


def distance_seed(seed: tf.Tensor) -> tf.Tensor:
    """Seed of a graph distance for :func:`propagate` with decrement 1."""
    return tf.where(tf.cast(seed, tf.bool), _FAR, 0)


def as_distance(x: tf.Tensor) -> tf.Tensor:
    """Graph distance from the result of :func:`propagate` on a distance seed."""
    return tf.where(x > 0, _FAR + 1 - x, 0)


def label_components(domain: tf.Tensor) -> tf.Tensor:
    """Label the edge-connected components of the bool ``domain``.

    Returns int32 labels that are distinct per component, in ``[1, ny*nx]``
    but not consecutive (the largest linear index of the component, plus
    one), and 0 outside ``domain``. They can serve directly as segment ids
    with ``num_segments = ny*nx + 1``.
    """
    return propagate(component_index(domain), domain, 0)


def graph_distance(seed: tf.Tensor, domain: tf.Tensor) -> tf.Tensor:
    """Edge-neighbour steps from ``seed`` within ``domain`` (int32).

    Seed cells are at distance 1; cells of ``domain`` that the seed cannot
    reach, and cells outside it, are 0.
    """
    return as_distance(propagate(distance_seed(seed), domain, 1))
