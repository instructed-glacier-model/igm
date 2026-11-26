#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Preconditioned conjugate gradients as a pure function of callables.

The recurrence mirrors ``OptimizerCGNewton._cg_solve`` but is split into an
initialisation and a resumable ``pcg_run`` so a caller can stop after ``k``
iterations, inspect the iterate, and continue from the same Krylov state.
Everything is written with ``tf.while_loop`` and no Python side effects, so
the functions can be traced inside any ``tf.function``.
"""

from typing import Callable, NamedTuple

import tensorflow as tf


TensorFn = Callable[[tf.Tensor], tf.Tensor]


class PCGState(NamedTuple):
    """Krylov state after ``iters`` iterations of PCG on ``A x = b``."""

    iters: tf.Tensor  # int32 scalar
    x: tf.Tensor
    r: tf.Tensor  # b - A x
    d: tf.Tensor  # search direction
    z: tf.Tensor  # preconditioned residual P^-1 r
    delta: tf.Tensor  # <r, z>
    delta_0: tf.Tensor  # <b, P^-1 b>, used for the relative tolerance
    xb: tf.Tensor  # <x, b> = ||x||^2_A when started from x0 = 0


class PCGResult(NamedTuple):
    x: tf.Tensor
    iters: tf.Tensor
    relres: tf.Tensor
    xb: tf.Tensor


def pcg_init(A: TensorFn, pre: TensorFn, b: tf.Tensor, x0: tf.Tensor) -> PCGState:
    """Initialise PCG for ``A x = b`` from ``x0``."""
    r = b - A(x0)
    z = pre(r)
    delta = tf.tensordot(r, z, axes=1)
    delta_0 = tf.tensordot(b, pre(b), axes=1)
    xb = tf.tensordot(x0, b, axes=1)
    return PCGState(
        iters=tf.constant(0, tf.int32),
        x=x0,
        r=r,
        d=z,
        z=z,
        delta=delta,
        delta_0=delta_0,
        xb=xb,
    )


def pcg_run(
    A: TensorFn,
    pre: TensorFn,
    state: PCGState,
    n_iter: tf.Tensor,
    tol: tf.Tensor,
) -> PCGState:
    """Advance PCG by at most ``n_iter`` iterations.

    Stops early once ``<r, z> <= tol^2 <b, P^-1 b>``. The tolerance is
    relative to ``b`` so it is meaningful across resumed segments.
    """
    n_iter = tf.cast(n_iter, tf.int32)
    tol = tf.cast(tol, state.delta.dtype)
    target = state.iters + n_iter
    threshold = tol * tol * state.delta_0

    def cond(i, x, r, d, z, delta, xb):
        del x, r, d, z, xb
        return tf.logical_and(i < target, delta > threshold)

    def body(i, x, r, d, z, delta, xb):
        q = A(d)
        alpha = delta / tf.tensordot(d, q, axes=1)
        x = x + alpha * d
        r = r - alpha * q
        z = pre(r)
        delta_old = delta
        delta = tf.tensordot(r, z, axes=1)
        beta = delta / delta_old
        d = z + beta * d
        # <x, b> accumulates alpha_i <r_i, z_i> when x0 = 0.
        xb = xb + alpha * delta_old
        return i + 1, x, r, d, z, delta, xb

    i, x, r, d, z, delta, xb = tf.while_loop(
        cond,
        body,
        [state.iters, state.x, state.r, state.d, state.z, state.delta, state.xb],
        parallel_iterations=1,
    )
    return PCGState(i, x, r, d, z, delta, state.delta_0, xb)


def pcg_relres(A: TensorFn, b: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
    """True relative residual ``||b - A x|| / ||b||`` (recomputed, not recursive)."""
    r_true = b - A(x)
    b_sq = tf.tensordot(b, b, axes=1)
    rs = tf.tensordot(r_true, r_true, axes=1)
    tiny = tf.cast(1e-30, b.dtype)
    return tf.sqrt(rs / tf.maximum(b_sq, tiny))


def pcg_solve(
    A: TensorFn,
    pre: TensorFn,
    b: tf.Tensor,
    x0: tf.Tensor,
    iter_max: tf.Tensor,
    tol: tf.Tensor,
) -> PCGResult:
    """Solve ``A x = b`` with PCG; ``iter_max == 0`` returns ``x0`` unchanged."""
    state = pcg_init(A, pre, b, x0)
    state = pcg_run(A, pre, state, iter_max, tol)
    return PCGResult(
        x=state.x,
        iters=state.iters,
        relres=pcg_relres(A, b, state.x),
        xb=state.xb,
    )
