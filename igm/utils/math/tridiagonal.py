#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""TensorFlow utilities for scalar and block-tridiagonal linear systems."""

import math
from typing import Tuple

import tensorflow as tf


def invert_2x2(
    a: tf.Tensor,
    b: tf.Tensor,
    c: tf.Tensor,
    d: tf.Tensor,
    relative_floor: tf.Tensor,
) -> tf.Tensor:
    """Invert pointwise 2x2 blocks with a determinant floor."""
    determinant = a * d - b * c
    tiny = 1e-300 if a.dtype == tf.float64 else 1e-30
    floor = tf.maximum(
        tf.cast(relative_floor, a.dtype) * tf.reduce_max(tf.abs(determinant)),
        tf.cast(tiny, a.dtype),
    )
    sign = tf.where(
        determinant < 0.0,
        -tf.ones_like(determinant),
        tf.ones_like(determinant),
    )
    determinant = sign * tf.maximum(tf.abs(determinant), floor)
    adjugate = tf.stack(
        [
            tf.stack([d, -b], axis=1),
            tf.stack([-c, a], axis=1),
        ],
        axis=1,
    )
    return adjugate * tf.math.reciprocal(
        determinant[:, tf.newaxis, tf.newaxis]
    )


def _shift_left_zero(field: tf.Tensor, stride: int) -> tf.Tensor:
    return tf.pad(field[:, :-stride], [[0, 0], [stride, 0]])


def _shift_right_zero(field: tf.Tensor, stride: int) -> tf.Tensor:
    return tf.pad(field[:, stride:], [[0, 0], [0, stride]])


def solve_tridiagonal_pcr(
    lower: tf.Tensor,
    diagonal: tf.Tensor,
    upper: tf.Tensor,
    rhs: tf.Tensor,
) -> tf.Tensor:
    """Solve batches of scalar tridiagonal systems by cyclic reduction.

    The chain length must be static so the logarithmic reduction can be
    unrolled while tracing an enclosing ``tf.function``.
    """
    size = diagonal.shape[-1]
    if size is None:
        raise ValueError("PCR requires a statically known chain length.")

    stride = 1
    while stride < size:
        lower_left = _shift_left_zero(lower, stride)
        diagonal_left = _shift_left_zero(diagonal, stride)
        upper_left = _shift_left_zero(upper, stride)
        rhs_left = _shift_left_zero(rhs, stride)
        lower_right = _shift_right_zero(lower, stride)
        diagonal_right = _shift_right_zero(diagonal, stride)
        upper_right = _shift_right_zero(upper, stride)
        rhs_right = _shift_right_zero(rhs, stride)

        alpha = tf.math.divide_no_nan(lower, diagonal_left)
        gamma = tf.math.divide_no_nan(upper, diagonal_right)
        lower, diagonal, upper, rhs = (
            -alpha * lower_left,
            diagonal - alpha * upper_left - gamma * lower_right,
            -gamma * upper_right,
            rhs - alpha * rhs_left - gamma * rhs_right,
        )
        stride *= 2

    return tf.math.divide_no_nan(rhs, diagonal)


def _shift_gather(x: tf.Tensor, shift: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """Shift along the chain axis and return the valid-position mask."""
    size = x.shape[-1]
    indices = tf.range(size) + shift
    valid = tf.logical_and(indices >= 0, indices < size)
    indices = tf.clip_by_value(indices, 0, size - 1)
    return tf.gather(x, indices, axis=-1), valid


def _matmul_blocks(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    return tf.einsum("bijx,bjkx->bikx", x, y)


def _matvec_blocks(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    return tf.einsum("bijx,bjx->bix", x, y)


def _invert_blocks(block: tf.Tensor, floor: tf.Tensor) -> tf.Tensor:
    return invert_2x2(
        block[:, 0, 0],
        block[:, 0, 1],
        block[:, 1, 0],
        block[:, 1, 1],
        floor,
    )


@tf.function
def solve_block_tridiagonal(
    lower: tf.Tensor,
    diagonal: tf.Tensor,
    upper: tf.Tensor,
    rhs: tf.Tensor,
    eigenvalue_floor: float = 1e-12,
) -> tf.Tensor:
    """Solve batched 2x2-block tridiagonal systems with cyclic reduction.

    The three matrix bands have shape ``(batch, 2, 2, size)`` and ``rhs``
    has shape ``(batch, 2, size)``. The chain length must be static so the
    logarithmic reduction can be unrolled while tracing.
    """
    size = diagonal.shape[-1]
    if size is None:
        raise ValueError(
            "Block PCR requires a statically known chain length."
        )

    dtype = diagonal.dtype
    floor = tf.cast(eigenvalue_floor, dtype)
    rounds = max(1, int(math.ceil(math.log2(max(size, 2)))))

    a, b, c, d = lower, diagonal, upper, rhs
    stride = 1
    for _ in range(rounds):
        a_left, valid_left = _shift_gather(a, -stride)
        b_left, _ = _shift_gather(b, -stride)
        c_left, _ = _shift_gather(c, -stride)
        d_left, _ = _shift_gather(d, -stride)
        a_right, valid_right = _shift_gather(a, stride)
        b_right, _ = _shift_gather(b, stride)
        c_right, _ = _shift_gather(c, stride)
        d_right, _ = _shift_gather(d, stride)

        alpha = _matmul_blocks(a, _invert_blocks(b_left, floor))
        gamma = _matmul_blocks(c, _invert_blocks(b_right, floor))
        alpha *= tf.cast(valid_left, dtype)
        gamma *= tf.cast(valid_right, dtype)

        a, b, c, d = (
            -_matmul_blocks(alpha, a_left),
            b
            - _matmul_blocks(alpha, c_left)
            - _matmul_blocks(gamma, a_right),
            -_matmul_blocks(gamma, c_right),
            d
            - _matvec_blocks(alpha, d_left)
            - _matvec_blocks(gamma, d_right),
        )
        stride *= 2

    return _matvec_blocks(_invert_blocks(b, floor), d)
