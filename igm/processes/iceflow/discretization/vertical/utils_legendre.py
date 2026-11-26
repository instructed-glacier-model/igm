#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from typing import Callable, List

import numpy as np
import tensorflow as tf
from numpy.polynomial.legendre import Legendre
from numpy.polynomial.polynomial import Polynomial


def zeta_to_x(zeta: tf.Tensor) -> tf.Tensor:
    """Map from reference coordinate [0,1] to Legendre domain [-1,1]."""
    return 2.0 * zeta - 1.0


def dxdzeta() -> tf.Tensor:
    """Derivative of coordinate transformation from zeta to x."""
    return 2.0


def flip(x: np.ndarray) -> np.ndarray:
    """Reverse array order."""
    return x[::-1]


def get_coefs(leg: Legendre) -> np.ndarray:
    """Convert Legendre polynomial to standard polynomial coefficients."""
    poly = leg.convert(kind=Polynomial)
    return flip(poly.coef)


def array_to_tf(array: np.ndarray, dtype: tf.DType) -> List[tf.Tensor]:
    """Convert numpy array to list of TensorFlow tensors."""
    return [tf.constant(elem, dtype=dtype) for elem in array]


def polyval_tf(coefs: np.ndarray, x: tf.Tensor) -> tf.Tensor:
    """Evaluate polynomial with given coefficients at x using TensorFlow."""
    coefs_tf = [tf.constant(c, dtype=x.dtype) for c in coefs]
    y = tf.math.polyval(coefs_tf, x)
    return y + tf.zeros_like(x)


def compute_basis(order: int) -> Callable[[tf.Tensor], tf.Tensor]:
    """Compute Legendre basis function of given order."""
    coefs = get_coefs(Legendre.basis(order))

    def basis_fn(zeta: tf.Tensor) -> tf.Tensor:
        x = zeta_to_x(zeta)
        return polyval_tf(coefs, x)

    return basis_fn


def compute_basis_grad(order: int) -> Callable[[tf.Tensor], tf.Tensor]:
    """Compute gradient of Legendre basis function of given order."""
    coefs_grad = get_coefs(Legendre.basis(order).deriv())

    def basis_grad_fn(zeta: tf.Tensor) -> tf.Tensor:
        x = zeta_to_x(zeta)
        return polyval_tf(coefs_grad, x) * dxdzeta()

    return basis_grad_fn


def compute_basis_int(order: int) -> Callable[[tf.Tensor], tf.Tensor]:
    """Compute integral of Legendre basis function of given order."""
    basis_int = Legendre.basis(order).integ()
    coefs_int = get_coefs(basis_int)
    coefs_int[-1] -= basis_int(-1.0)
    coefs_int *= 0.5

    def basis_int_fn(zeta: tf.Tensor) -> tf.Tensor:
        x = zeta_to_x(zeta)
        return polyval_tf(coefs_int, x)

    return basis_int_fn
