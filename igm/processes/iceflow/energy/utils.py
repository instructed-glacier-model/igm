#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
import numpy as np

@tf.function()
def stag2(B):
    return (B[..., 1:] + B[..., :-1]) / 2

@tf.function()
def stag2v(B):
    if B.shape[-3] > 1:
        B = (B[..., :-1, :, :] + B[..., 1:, :, :]) / 2
    return B

@tf.function()
def stag4h(B):
    return (
        B[..., 1:, 1:] + B[..., 1:, :-1] + B[..., :-1, 1:] + B[..., :-1, :-1]
    ) / 4

def psia(zeta,exp_glen):
    return ( 1 - (1 - zeta) ** (exp_glen + 1) )

def psiap(zeta,exp_glen):
    return (exp_glen + 1) * (1 - zeta) ** exp_glen

def gauss_points_and_weights(ord_gauss):
    # Get nodes and weights on [-1, 1]
    x, w = np.polynomial.legendre.leggauss(ord_gauss)

    # Shift to [0, 1]
    zeta = 0.5 * (x + 1)
    dzeta = 0.5 * w

    # Convert to TensorFlow tensors (with dummy dims for batch/spatial broadcasting)
    zeta_tf = tf.constant(zeta, dtype=tf.float32)[None, :, None, None]
    dzeta_tf = tf.constant(dzeta, dtype=tf.float32)[None, :, None, None]
    return zeta_tf, dzeta_tf

def legendre_basis(zeta):
    ord = zeta.shape[-3]
    x = 2.0 * zeta - 1.0  # Map from [0,1] to [-1,1]
    P = [tf.ones_like(x), x]  # P_0, P_1
    for k in range(1, ord):
        Pk = ((2 * k + 1) * x * P[k] - k * P[k - 1]) / (k + 1)
        P.append(Pk)

    return tf.stack(P[:ord + 1], axis=-3) 
