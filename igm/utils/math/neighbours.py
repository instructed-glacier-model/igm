#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Reductions over the four edge neighbours of each cell of a 2-D raster.

A cell outside the domain is never a neighbour: the field is padded with a
constant ``fill`` (0 or False), so every reduction below ignores it. All
functions are shape-agnostic in the last two axes and trace to slices, which
XLA fuses (no gather).
"""

from typing import List, Union

import tensorflow as tf


def neighbours(x: tf.Tensor, fill: Union[int, float, bool] = 0) -> List[tf.Tensor]:
    """The four edge neighbours of ``x``: rows - 1, rows + 1, cols - 1, cols + 1."""
    p = tf.pad(x, [[1, 1], [1, 1]], constant_values=fill)
    return [p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]]


def any_neighbour(mask: tf.Tensor) -> tf.Tensor:
    """True where at least one edge neighbour of the bool ``mask`` is True."""
    n = neighbours(mask, False)
    return n[0] | n[1] | n[2] | n[3]


def count_neighbours(mask: tf.Tensor, dtype: tf.DType) -> tf.Tensor:
    """Number of edge neighbours where the bool ``mask`` is True, as ``dtype``."""
    return tf.add_n(neighbours(tf.cast(mask, dtype), 0))


def neighbour_sum(x: tf.Tensor) -> tf.Tensor:
    """Sum of ``x`` over the four edge neighbours (0 beyond the domain)."""
    return tf.add_n(neighbours(x, 0))


def neighbour_mean(x: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    """Mean of ``x`` over the edge neighbours where ``mask``; 0 if there is none."""
    weight = tf.cast(mask, x.dtype)
    total = neighbour_sum(x * weight)
    count = neighbour_sum(weight)
    return tf.where(count > 0.0, total / tf.maximum(count, 1.0), tf.zeros_like(x))


def dilate(mask: tf.Tensor, steps: int) -> tf.Tensor:
    """Edge-connected dilation of the bool ``mask`` by ``steps`` cells."""
    for _ in range(int(steps)):
        mask = mask | any_neighbour(mask)
    return mask
