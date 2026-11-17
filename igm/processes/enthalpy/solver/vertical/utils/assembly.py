#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Tuple


@tf.function
def assemble_system(
    E: tf.Tensor,
    dt: tf.Tensor,
    dz: tf.Tensor,
    w: tf.Tensor,
    K: tf.Tensor,
    f: tf.Tensor,
    BCB: tf.Tensor,
    VB: tf.Tensor,
    VS: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    TensorFlow function to assemble finite difference system for enthalpy equation.

    Discretizes the 1D vertical enthalpy equation: dE/dt + w·dE/dz = d/dz(K·dE/dz) + f
    using implicit time stepping and upwind advection.

    Args:
        E: Current enthalpy field (J kg^-1).
        dt: Time step (s).
        dz: Vertical layer thickness (m).
        w: Vertical velocity field (m s^-1).
        K: Thermal diffusivity field (m^2 s^-1).
        f: Volumetric source term (W kg^-1).
        BCB: Bottom boundary condition flag (1=Neumann, 0=Dirichlet) (-).
        VB: Bottom boundary condition value (J kg^-1 or J kg^-1 m^-1).
        VS: Surface boundary condition value (J kg^-1).

    Returns:
        Tuple of tridiagonal system components (L, M, U, R).
    """
    # K is defined at interfaces; each interior node owns half of
    # the neighboring intervals. Form fluxes first, then their divergence.
    cell_width = 0.5 * (dz[:-1] + dz[1:])
    lower = dt * K[:-1] / (cell_width * dz[:-1])
    upper = dt * K[1:] / (cell_width * dz[1:])
    zero = tf.zeros_like(E[:1])
    one = tf.ones_like(E[:1])

    L = tf.concat([-lower, zero], axis=0)
    M = tf.concat([one, 1.0 + lower + upper, one], axis=0)
    U = tf.concat([zero, -upper], axis=0)
    R = E + dt * f

    # Bottom boundary condition
    M = tf.concat(
        [tf.where(BCB == 1, -tf.ones_like(BCB), tf.ones_like(BCB))[None, :, :], M[1:]],
        axis=0,
    )
    U = tf.concat(
        [tf.where(BCB == 1, tf.ones_like(BCB), tf.zeros_like(BCB))[None, :, :], U[1:]],
        axis=0,
    )
    R = tf.concat([tf.where(BCB == 1, VB * dz[0], VB)[None, :, :], R[1:]], axis=0)

    # Surface boundary condition
    M = tf.concat([M[:-1], tf.ones_like(BCB)[None, :, :]], axis=0)
    L = tf.concat([L[:-1], tf.zeros_like(BCB)[None, :, :]], axis=0)
    R = tf.concat([R[:-1], VS[None, :, :]], axis=0)

    # Upwind advection (implicit)
    wdivdz = dt * (w[1:] + w[:-1]) / (2.0 * dz)
    L = tf.concat([L[:-1] + tf.where(w[1:-1] > 0, -wdivdz[:-1], 0), L[-1:]], axis=0)
    M = tf.concat(
        [M[:1], M[1:-1] + tf.where(w[1:-1] > 0, wdivdz[:-1], -wdivdz[1:]), M[-1:]],
        axis=0,
    )
    U = tf.concat([U[:1], U[1:] + tf.where(w[1:-1] <= 0, wdivdz[1:], 0)], axis=0)

    return L, M, U, R
