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

def legendre_basis(zeta, order):
    """
    Compute the derivative of Legendre basis matrix evaluated at points zeta in [0,1].

    Parameters:
    - zeta: tf.Tensor of shape (..., n_points), values in [0, 1]
    - order: int, number of basis functions (max degree = order - 1)

    Returns the Vandermonde matrix V and its derivative dV/dz.
    """
    x = 2.0 * zeta - 1.0
 
    P = [tf.ones_like(x), x]
    for k in range(2, order):
        P.append(((2 * k - 1) * x * P[-1] - (k - 1) * P[-2]) / k)

    V = tf.stack(P, axis=-2)

    dP = [tf.zeros_like(x)]
    for k in range(1, order):
        dP.append(k * (x * P[k] - P[k - 1]) / (x**2 - 1.0)) 

    dVdz = 2.0 * tf.stack(dP, axis=-2)

    return tf.transpose(V), tf.transpose(dVdz)  
