#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Callable, Tuple


@tf.function
def dot_uv(aU: tf.Tensor, aV: tf.Tensor, bU: tf.Tensor, bV: tf.Tensor) -> tf.Tensor:
    return tf.reduce_sum(aU * bU) + tf.reduce_sum(aV * bV)


@tf.function
def value_and_grad(
    cost_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
    U: tf.Tensor,
    V: tf.Tensor,
    inputs_batched: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    with tf.GradientTape() as tape:
        tape.watch([U, V])
        cost = cost_fn(tf.expand_dims(U, 0), tf.expand_dims(V, 0), inputs_batched)
    gU, gV = tape.gradient(cost, [U, V])
    return cost, gU, gV


def make_hvp_fn(
    cost_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
    inputs_batched: tf.Tensor,
    U0: tf.Tensor,
    V0: tf.Tensor,
    mask: tf.Tensor,
    shift: float = 0.0,
) -> Callable[[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
    shift_t = tf.cast(shift, U0.dtype)
    do_shift = shift != 0.0

    def hvp(dU: tf.Tensor, dV: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        mdU = dU * mask
        mdV = dV * mask

        with tf.GradientTape() as outer:
            outer.watch([U0, V0])
            with tf.GradientTape() as inner:
                inner.watch([U0, V0])
                cost = cost_fn(
                    tf.expand_dims(U0, 0),
                    tf.expand_dims(V0, 0),
                    inputs_batched,
                )
            gU, gV = inner.gradient(cost, [U0, V0])
            g_dot_d = tf.reduce_sum(gU * mdU) + tf.reduce_sum(gV * mdV)

        HvU, HvV = outer.gradient(g_dot_d, [U0, V0])
        HvU = HvU * mask
        HvV = HvV * mask

        if do_shift:
            HvU = HvU + shift_t * mdU
            HvV = HvV + shift_t * mdV

        return HvU, HvV

    return hvp


@tf.function
def cg(
    gU: tf.Tensor,
    gV: tf.Tensor,
    hvp_fn: Callable[[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]],
    mask: tf.Tensor,
    tol: float,
    maxit: int,
    x0U: tf.Tensor = None,
    x0V: tf.Tensor = None,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:

    dtype = gU.dtype
    eps = tf.cast(1e-30, dtype)
    tol_t = tf.cast(tol, dtype)
    zero = tf.cast(0.0, dtype)

    gnorm2 = tf.maximum(dot_uv(gU, gV, gU, gV), eps)

    if x0U is not None:
        xU_init = x0U * mask
        xV_init = x0V * mask
        HvU0, HvV0 = hvp_fn(xU_init, xV_init)
        rU = (gU - HvU0) * mask
        rV = (gV - HvV0) * mask
    else:
        xU_init = tf.zeros_like(gU)
        xV_init = tf.zeros_like(gV)
        rU = gU * mask
        rV = gV * mask

    rz = dot_uv(rU, rV, rU, rV)

    init = (
        xU_init,
        xV_init,
        rU,
        rV,
        tf.identity(rU),
        tf.identity(rV),
        rz,
        tf.constant(0, tf.int32),
        tf.cast(1.0, dtype),
        tf.constant(False),
        tf.constant(False),
    )

    def cond(xU, xV, rU, rV, pU, pV, rz, it, relres, stop, converged):
        return tf.logical_and(it < maxit, tf.logical_not(stop))

    def body(xU, xV, rU, rV, pU, pV, rz, it, relres, stop, converged):
        pUm = pU * mask
        pVm = pV * mask

        qU, qV = hvp_fn(pUm, pVm)
        denom = dot_uv(pUm, pVm, qU, qV)

        safe = tf.math.is_finite(denom) & (denom > eps)
        alpha = tf.where(safe, rz / tf.maximum(denom, eps), zero)

        nxU = (xU + alpha * pUm) * mask
        nxV = (xV + alpha * pVm) * mask
        nrU = (rU - alpha * qU) * mask
        nrV = (rV - alpha * qV) * mask

        nrz = dot_uv(nrU, nrV, nrU, nrV)
        new_relres = tf.sqrt(tf.maximum(nrz, eps) / gnorm2)

        new_bad = (
            tf.reduce_any(tf.math.is_nan(nrU))
            | tf.reduce_any(tf.math.is_nan(nrV))
            | ~tf.math.is_finite(dot_uv(gU, gV, nxU, nxV))
            | ~safe
        )
        new_converged = new_relres <= tol_t
        new_stop = new_bad | new_converged

        beta = nrz / tf.maximum(rz, eps)
        npU = nrU + beta * pUm
        npV = nrV + beta * pVm

        return (
            nxU,
            nxV,
            nrU,
            nrV,
            npU,
            npV,
            nrz,
            it + 1,
            new_relres,
            new_stop,
            new_converged,
        )

    xU, xV, _, _, _, _, _, niter, relres, _, converged = tf.while_loop(
        cond,
        body,
        init,
        maximum_iterations=maxit,
        parallel_iterations=1,
    )
    return xU, xV, niter, relres, converged
