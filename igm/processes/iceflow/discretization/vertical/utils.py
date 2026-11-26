#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from typing import Callable, Tuple

import numpy as np
import tensorflow as tf


@tf.function()
def compute_zeta_linear(Nz: int, dtype: tf.DType = tf.float32) -> tf.Tensor:
    """Compute linearly spaced vertical coordinates from 0 to 1."""
    return tf.cast(tf.range(Nz) / (Nz - 1), dtype)


@tf.function()
def compute_zeta_quadratic(
    Nz: int, slope_init: float, dtype: tf.DType = tf.float32
) -> tf.Tensor:
    """Compute quadratically spaced vertical coordinates with specified initial slope."""
    zeta = compute_zeta_linear(Nz, dtype)
    return slope_init * zeta + (1.0 - slope_init) * zeta**2


@tf.function()
def compute_zeta(
    Nz: int, slope_init: float = 1.0, dtype: tf.DType = tf.float32
) -> tf.Tensor:
    """Compute vertical coordinate distribution (default quadratic)."""
    return compute_zeta_quadratic(Nz, slope_init, dtype)


@tf.function()
def compute_zeta_mid(zeta: tf.Tensor) -> tf.Tensor:
    """Compute midpoints between consecutive zeta values."""
    Nz = zeta.shape[0]
    if Nz > 1:
        return (zeta[1:] + zeta[:-1]) / 2.0
    else:
        return 0.5 * tf.ones((1), dtype=zeta.dtype)


@tf.function()
def compute_dzeta(zeta: tf.Tensor) -> tf.Tensor:
    """Compute spacings between consecutive zeta values."""
    Nz = zeta.shape[0]
    if Nz > 1:
        return zeta[1:] - zeta[:-1]
    else:
        return 1.0 * tf.ones((1), dtype=zeta.dtype)


@tf.function
def compute_zetas(
    Nz: int, slope_init: float = 1.0, dtype: tf.DType = tf.float32
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Compute zeta coordinates, midpoints, and spacings."""
    zeta = compute_zeta(Nz, slope_init, dtype)
    zeta_mid = compute_zeta_mid(zeta)
    dzeta = compute_dzeta(zeta)
    return zeta, zeta_mid, dzeta


@tf.function()
def compute_depth(dzeta: tf.Tensor) -> tf.Tensor:
    """Compute normalized depth below top surface at each level."""
    zero = tf.zeros((1,), dtype=dzeta.dtype)
    D = tf.concat([dzeta, zero], axis=0)
    return tf.math.cumsum(D, axis=0, reverse=True)


def compute_gauss_quad(
    order: int, dtype: tf.DType = tf.float32
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Compute Gauss-Legendre quadrature points and weights on [0,1]."""
    x_quad, w_quad = np.polynomial.legendre.leggauss(order)

    x_quad = 0.5 * (x_quad + 1.0)
    w_quad = 0.5 * w_quad

    x_quad_tf = tf.constant(x_quad, dtype=dtype)
    w_quad_tf = tf.constant(w_quad, dtype=dtype)

    return x_quad_tf, w_quad_tf


def compute_midpoint_quad(
    zeta: tf.Tensor, dzeta: tf.Tensor, dtype: tf.DType = tf.float32
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Compute midpoint quadrature points and weights on [0,1]."""
    x_quad = tf.cast(compute_zeta_mid(zeta), dtype)
    w_quad = tf.cast(dzeta, dtype)

    return x_quad, w_quad


def compute_trap_quad(
    zeta: tf.Tensor, dzeta: tf.Tensor, dtype: tf.DType = tf.float32
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Compute trapezoidal quadrature points and weights on [0,1]."""
    x_quad = tf.cast(zeta, dtype)
    Nz = zeta.shape[0]

    if Nz == 1:
        w_quad = tf.ones((1,), dtype=dtype)
    elif Nz == 2:
        w_quad = tf.concat([dzeta[:1] / 2.0, dzeta[-1:] / 2.0], axis=0)
    else:
        w_surface = dzeta[:1] / 2.0
        w_interior = (dzeta[:-1] + dzeta[1:]) / 2.0
        w_bed = dzeta[-1:] / 2.0
        w_quad = tf.concat([w_surface, w_interior, w_bed], axis=0)

    w_quad = tf.cast(w_quad, dtype)

    return x_quad, w_quad


def compute_basis_vector(
    basis: Tuple[Callable[[tf.Tensor], tf.Tensor], ...], x: tf.Tensor
) -> tf.Tensor:
    """Evaluate all basis functions at a single point."""
    V = [fct(x) for fct in basis]
    return V


def compute_basis_matrix(
    basis: Tuple[Callable[[tf.Tensor], tf.Tensor], ...], x: tf.Tensor
) -> tf.Tensor:
    """Evaluate all basis functions at multiple points to form a matrix."""
    M = [fct(x) for fct in basis]
    M = tf.stack(M, axis=1)
    return M


def compute_matrices(
    basis_fct: Tuple[Callable[[tf.Tensor], tf.Tensor], ...],
    basis_fct_grad: Tuple[Callable[[tf.Tensor], tf.Tensor], ...],
    x_quad: tf.Tensor,
    w_quad: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Compute basis matrices at quadrature points, boundaries, and vertical average."""

    V_q = compute_basis_matrix(basis_fct, x_quad)
    V_q_grad = compute_basis_matrix(basis_fct_grad, x_quad)

    x_b = tf.constant(0.0, dtype=x_quad.dtype)
    x_s = tf.constant(1.0, dtype=x_quad.dtype)

    V_b = compute_basis_vector(basis_fct, x_b)
    V_s = compute_basis_vector(basis_fct, x_s)

    V_b = tf.stack(V_b)
    V_s = tf.stack(V_s)

    V_bar = tf.reduce_sum(V_q * w_quad[:, None], axis=0)

    return V_q, V_q_grad, V_b, V_s, V_bar


def compute_V_int_nodal(
    basis_fct_int: Tuple[Callable[[tf.Tensor], tf.Tensor], ...],
    zeta_nodes: tf.Tensor,
) -> tf.Tensor:
    """
    Compute integration matrix for nodal bases.

    V_int[i,n] = Φₙ(ζᵢ) = ∫₀^ζᵢ φₙ dζ'
    """
    return compute_basis_matrix(basis_fct_int, zeta_nodes)


def compute_V_int_orthogonal(
    basis_fct: Tuple[Callable[[tf.Tensor], tf.Tensor], ...],
    basis_fct_int: Tuple[Callable[[tf.Tensor], tf.Tensor], ...],
    x_quad: tf.Tensor,
    w_quad: tf.Tensor,
    normalization: tf.Tensor,
) -> tf.Tensor:
    """
    Compute integration matrix for orthogonal bases.

    Uses exact projection: V_int = V_proj @ V_q_int
    """
    V_q = compute_basis_matrix(basis_fct, x_quad)
    V_q_int = compute_basis_matrix(basis_fct_int, x_quad)
    V_proj = normalization[:, None] * w_quad[None, :] * tf.transpose(V_q)
    return tf.matmul(V_proj, V_q_int)


def compute_basis_corr_b(
    basis_fct: Tuple[Callable[[tf.Tensor], tf.Tensor], ...],
    basis_fct_int: Tuple[Callable[[tf.Tensor], tf.Tensor], ...],
) -> Tuple[Callable[[tf.Tensor], tf.Tensor], ...]:
    """
    Compute bed terrain correction basis functions.

    ψ^b_n(ζ) = ∫₀^ζ φ'_n(ζ')(1-ζ') dζ' = (1-ζ)φ_n(ζ) - φ_n(0) + Φ_n(ζ)

    Derived via integration by parts.
    """

    def make_psi_b(phi_n, Phi_n):
        def psi_b_n(zeta):
            return (1.0 - zeta) * phi_n(zeta) - phi_n(tf.zeros_like(zeta)) + Phi_n(zeta)

        return psi_b_n

    return tuple(
        make_psi_b(basis_fct[n], basis_fct_int[n]) for n in range(len(basis_fct))
    )


def compute_basis_corr_s(
    basis_fct: Tuple[Callable[[tf.Tensor], tf.Tensor], ...],
    basis_fct_int: Tuple[Callable[[tf.Tensor], tf.Tensor], ...],
) -> Tuple[Callable[[tf.Tensor], tf.Tensor], ...]:
    """
    Compute surface terrain correction basis functions.

    ψ^s_n(ζ) = ∫₀^ζ φ'_n(ζ') ζ' dζ' = ζ φ_n(ζ) - Φ_n(ζ)

    Derived via integration by parts.
    """

    def make_psi_s(phi_n, Phi_n):
        def psi_s_n(zeta):
            return zeta * phi_n(zeta) - Phi_n(zeta)

        return psi_s_n

    return tuple(
        make_psi_s(basis_fct[n], basis_fct_int[n]) for n in range(len(basis_fct))
    )


def compute_V_corr_nodal(
    basis_fct: Tuple[Callable[[tf.Tensor], tf.Tensor], ...],
    basis_fct_int: Tuple[Callable[[tf.Tensor], tf.Tensor], ...],
    zeta_nodes: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Compute terrain correction matrices for nodal bases.

    V_corr_b[i,n] = ψ^b_n(ζᵢ)
    V_corr_s[i,n] = ψ^s_n(ζᵢ)
    """
    basis_corr_b = compute_basis_corr_b(basis_fct, basis_fct_int)
    basis_corr_s = compute_basis_corr_s(basis_fct, basis_fct_int)

    V_corr_b = compute_basis_matrix(basis_corr_b, zeta_nodes)
    V_corr_s = compute_basis_matrix(basis_corr_s, zeta_nodes)

    return V_corr_b, V_corr_s


def compute_V_corr_orthogonal(
    basis_fct: Tuple[Callable[[tf.Tensor], tf.Tensor], ...],
    basis_fct_int: Tuple[Callable[[tf.Tensor], tf.Tensor], ...],
    x_quad: tf.Tensor,
    w_quad: tf.Tensor,
    normalization: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Compute terrain correction matrices for orthogonal bases.

    Projects ψ^b_n and ψ^s_n onto the basis using exact quadrature.
    """
    basis_corr_b = compute_basis_corr_b(basis_fct, basis_fct_int)
    basis_corr_s = compute_basis_corr_s(basis_fct, basis_fct_int)

    V_q = compute_basis_matrix(basis_fct, x_quad)
    V_q_corr_b = compute_basis_matrix(basis_corr_b, x_quad)
    V_q_corr_s = compute_basis_matrix(basis_corr_s, x_quad)

    V_proj = normalization[:, None] * w_quad[None, :] * tf.transpose(V_q)

    V_corr_b = tf.matmul(V_proj, V_q_corr_b)
    V_corr_s = tf.matmul(V_proj, V_q_corr_s)

    return V_corr_b, V_corr_s
