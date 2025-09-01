#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
import numpy as np
from numpy.polynomial.legendre import Legendre
from numpy.polynomial.polynomial import Polynomial

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
    zeta_tf = tf.constant(zeta, dtype=tf.float32)
    dzeta_tf = tf.constant(dzeta, dtype=tf.float32)
    return zeta_tf, dzeta_tf
 
def get_shifted_legendre_antideriv_coeffs(order):
    """
    Compute coefficients of ∫₀^ζ P_k(2ζ-1) dζ using exact Legendre antiderivative formula.

    Returns:
    - coeffs_list: list of arrays, one per basis function, ready for tf.math.polyval (highest degree first)
    """
    coeffs_list = []
    for k in range(order):
        Pkp1 = Legendre.basis(k + 1)
        Pkm1 = Legendre.basis(k - 1) if k > 0 else Legendre([0.0])
        antideriv = (Pkp1 - Pkm1) * (1.0 / (2 * k + 1))
        coeffs_monomial = antideriv.convert(kind=Polynomial).coef[::-1]  # reverse for tf.math.polyval
        coeffs_list.append(coeffs_monomial)
    return coeffs_list

def legendre_basis(zeta, order):
    """
    Compute Legendre Vandermonde matrix, derivatives, and exact integrals.

    Parameters:
    - zeta: tf.Tensor of shape (n_points,), values in [0, 1]
    - order: int, number of basis functions

    Returns:
    - V: (n_points, order)
    - dVdz: (n_points, order)
    - I: (n_points, order) — exact ∫₀^ζ P_k(2s-1) ds
    """
    x = 2.0 * zeta - 1.0
    n_points = tf.shape(zeta)[-1]

    # Compute basis P_k(2zeta - 1)
    P = [tf.ones_like(x)]
    if order > 1:
        P.append(x)
    for k in range(2, order):
        Pk = ((2 * k - 1) * x * P[-1] - (k - 1) * P[-2]) / k
        P.append(Pk)
    V = tf.stack(P, axis=-2)

    # Derivative of P_k
    dP = [tf.zeros_like(x)]
    for k in range(1, order):
        dP.append(k * (x * P[k] - P[k - 1]) / (x**2 - 1.0))
    dVdz = 2.0 * tf.stack(dP, axis=-2)

    # Exact integral: I_k(ζ) = 0.5 * ∫_{-1}^{2ζ - 1} P_k(x) dx
    antideriv_coeffs = get_shifted_legendre_antideriv_coeffs(order)
    I = tf.stack([
        0.5 * tf.math.polyval([tf.constant(v, dtype=zeta.dtype) for v in c], x)
        for c in antideriv_coeffs
    ], axis=0)

    return tf.transpose(V), tf.transpose(dVdz), tf.transpose(I)