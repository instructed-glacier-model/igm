#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf


def polynomial_sia(zeta: tf.Tensor, exp: tf.Tensor) -> tf.Tensor:
    """SIA polynomial basis function with given exponent."""
    return 1.0 - tf.pow(1.0 - zeta, exp + 1.0)


def grad_polynomial_sia(zeta: tf.Tensor, exp: tf.Tensor) -> tf.Tensor:
    """Gradient of SIA polynomial basis function."""
    return (exp + 1.0) * tf.pow(1 - zeta, exp)


def int_polynomial_sia(zeta: tf.Tensor, exp: tf.Tensor) -> tf.Tensor:
    """Integral of SIA polynomial basis function."""
    return zeta - (1.0 - tf.pow(1.0 - zeta, exp + 2.0)) / (exp + 2.0)


def phi_bed(zeta: tf.Tensor, exp: tf.Tensor) -> tf.Tensor:
    """Basis function at the bed."""
    return 1.0 - polynomial_sia(zeta, exp)


def phi_surf(zeta: tf.Tensor, exp: tf.Tensor) -> tf.Tensor:
    """Basis function at the surface."""
    return polynomial_sia(zeta, exp)


def grad_phi_bed(zeta: tf.Tensor, exp: tf.Tensor) -> tf.Tensor:
    """Gradient of bed basis function."""
    return -grad_polynomial_sia(zeta, exp)


def grad_phi_surf(zeta: tf.Tensor, exp: tf.Tensor) -> tf.Tensor:
    """Gradient of surface basis function."""
    return grad_polynomial_sia(zeta, exp)


def int_phi_bed(zeta: tf.Tensor, exp: tf.Tensor) -> tf.Tensor:
    """Integral of bed basis function."""
    return zeta - int_polynomial_sia(zeta, exp)


def int_phi_surf(zeta: tf.Tensor, exp: tf.Tensor) -> tf.Tensor:
    """Integral of surface basis function."""
    return int_polynomial_sia(zeta, exp)
