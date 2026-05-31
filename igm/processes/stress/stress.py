#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from igm.processes.iceflow.utils.vertical_discretization import (
    compute_levels,
    compute_dz,
    compute_depth,
)


def initialize(cfg, state):

    if "iceflow" not in cfg.processes:
        raise ValueError("The 'iceflow' module is required for the 'stress' module.")

    state.tau_xx = tf.zeros_like(state.U)
    state.tau_yy = tf.zeros_like(state.U)
    state.tau_zz = tf.zeros_like(state.U)
    state.tau_xy = tf.zeros_like(state.U)
    state.tau_xz = tf.zeros_like(state.U)
    state.tau_yz = tf.zeros_like(state.U)


def update(cfg, state):
    if hasattr(state, "logger"):
        state.logger.info("Update STRESS at time : " + str(state.t.numpy()))

    # get the vertical discretization
    levels = compute_levels(
        cfg.processes.iceflow.numerics.Nz, cfg.processes.iceflow.numerics.vert_spacing
    )
    dz = compute_dz(state.thk, levels)
    depth = compute_depth(dz)

    B = (tf.expand_dims(state.arrhenius, axis=0)) ** (
        -1.0 / cfg.processes.iceflow.physics.viscosity.exponent
    )

    Exx, Eyy, Ezz, Exy, Exz, Eyz = compute_strainratetensor_tf(
        state.U, state.V, state.dx, dz, thr=1.0
    )

    strainrate = (
        0.5
        * (
            Exx**2
            + Exy**2
            + Exz**2
            + Exy**2
            + Eyy**2
            + Eyz**2
            + Exz**2
            + Eyz**2
            + Ezz**2
            + cfg.processes.iceflow.physics.viscosity.regularization**2
        )
        ** 0.5
    )

    mu = 0.5 * B * strainrate ** (1.0 / cfg.processes.iceflow.physics.viscosity.exponent - 1)

    state.tau_xx = 2 * mu * Exx
    state.tau_yy = 2 * mu * Eyy
    state.tau_zz = 2 * mu * Ezz
    state.tau_xy = 2 * mu * Exy
    state.tau_xz = 2 * mu * Exz
    state.tau_yz = 2 * mu * Eyz

    # Compute hydrostatic pressure
    state.p = (
        cfg.processes.iceflow.physics.ice_density
        * cfg.processes.iceflow.physics.gravity_cst
        * depth
        * 1.0e-6
    )  # Convert to MPa

    state.sigma1 = (
        compute_largest_eigenvalue_trace0_sym3x3(
            state.tau_xx,
            state.tau_yy,
            state.tau_zz,
            state.tau_xy,
            state.tau_xz,
            state.tau_yz,
        )
        - state.p
    )

    state.sigmaI = -3.0 * state.p

    state.tauII = tf.sqrt(
        3.0
        * 0.5
        * (
            state.tau_xx**2
            + state.tau_xy**2
            + state.tau_xz**2
            + state.tau_xy**2
            + state.tau_yy**2
            + state.tau_yz**2
            + state.tau_xz**2
            + state.tau_yz**2
            + state.tau_zz**2
        )
    )


def finalize(cfg, state):
    pass


def compute_largest_eigenvalue_trace0_sym3x3(
    tau_xx, tau_yy, tau_zz, tau_xy, tau_xz, tau_yz
):
    # Assumes trace = tau_xx + tau_yy + tau_zz = 0

    # Compute p = -½ tr(sigma²)
    p = -0.5 * (
        tau_xx**2 + tau_yy**2 + tau_zz**2 + 2 * (tau_xy**2 + tau_xz**2 + tau_yz**2)
    )

    # Compute determinant q = -det(sigma)
    # Using expansion by minors (for symmetric matrix)
    det = (
        tau_xx * (tau_yy * tau_zz - tau_yz**2)
        - tau_xy * (tau_xy * tau_zz - tau_yz * tau_xz)
        + tau_xz * (tau_xy * tau_yz - tau_yy * tau_xz)
    )
    q = -det

    # Avoid numerical issues: ensure p < 0 and handle borderline cases
    eps = 1e-10
    sqrt_term = tf.sqrt(tf.maximum(-p / 3.0, eps))
    acos_arg = tf.clip_by_value(-1.5 * q / (p * sqrt_term), -1.0 + eps, 1.0 - eps)
    theta = tf.acos(acos_arg)

    # Compute largest eigenvalue
    lambda_max = 2.0 * sqrt_term * tf.cos(theta / 3.0)

    return lambda_max


# @tf.function()
# def compute_strainratetensor_tf(U, V, dx, dz, thr):

#     Ui = tf.pad(U[:, :, :], [[0, 0], [0, 0], [1, 1]], "SYMMETRIC")
#     Uj = tf.pad(U[:, :, :], [[0, 0], [1, 1], [0, 0]], "SYMMETRIC")
#     Uk = tf.pad(U[:, :, :], [[1, 1], [0, 0], [0, 0]], "SYMMETRIC")

#     Vi = tf.pad(V[:, :, :], [[0, 0], [0, 0], [1, 1]], "SYMMETRIC")
#     Vj = tf.pad(V[:, :, :], [[0, 0], [1, 1], [0, 0]], "SYMMETRIC")
#     Vk = tf.pad(V[:, :, :], [[1, 1], [0, 0], [0, 0]], "SYMMETRIC")

#     DZ2 = tf.concat([dz[0:1], dz[:-1] + dz[1:], dz[-1:]], axis=0)

#     Exx = (Ui[:, :, 2:] - Ui[:, :, :-2]) / (2 * dx)
#     Eyy = (Vj[:, 2:, :] - Vj[:, :-2, :]) / (2 * dx)
#     Ezz = -Exx - Eyy

#     Exy = 0.5 * (Vi[:, :, 2:] - Vi[:, :, :-2]) / (2 * dx) + 0.5 * (
#         Uj[:, 2:, :] - Uj[:, :-2, :]
#     ) / (2 * dx)
#     Exz = 0.5 * (Uk[2:, :, :] - Uk[:-2, :, :]) / tf.maximum(DZ2, thr)
#     Eyz = 0.5 * (Vk[2:, :, :] - Vk[:-2, :, :]) / tf.maximum(DZ2, thr)

#     Exx = tf.where(DZ2 > 1, Exx, 0.0)
#     Eyy = tf.where(DZ2 > 1, Eyy, 0.0)
#     Ezz = tf.where(DZ2 > 1, Ezz, 0.0)
#     Exy = tf.where(DZ2 > 1, Exy, 0.0)
#     Exz = tf.where(DZ2 > 1, Exz, 0.0)
#     Eyz = tf.where(DZ2 > 1, Eyz, 0.0)

#     return Exx, Eyy, Ezz, Exy, Exz, Eyz


@tf.function()
def compute_strainratetensor_tf(U, V, dx, dz, thr):
    """
    Compute strain-rate tensor components using 4th-order central differences.
    U: velocity in x-direction (tensor [nz, ny, nx] or [batch,...])
    V: velocity in y-direction
    dx: grid spacing (assumed isotropic in x,y)
    dz: vertical grid spacing (1D tensor along z)
    thr: threshold to avoid division by very small dz
    """

    def deriv4(f, axis, d):
        """
        4th-order central difference along given axis
        f: tensor [nz, ny, nx]
        axis: 0 (z), 1 (y), 2 (x)
        d: spacing (scalar or tensor)
        """

        # Pad with 2 points on each side
        paddings = [[0, 0]] * len(f.shape)
        paddings[axis] = [2, 2]
        fpad = tf.pad(f, paddings, mode="SYMMETRIC")

        # Promote to [batch, depth, height, width, channels]
        fpad5 = fpad[None, ..., None]  # shape [1, nz+4, ny+4, nx+4, 1]

        # Stencil coefficients
        coeffs = tf.constant([-1, 8, 0, -8, 1], dtype=f.dtype) / (12.0 * d)

        # Build kernel for 3D conv
        kernel_shape = [1, 1, 1]
        kernel_shape[axis] = 5
        kernel = tf.reshape(coeffs, kernel_shape + [1, 1])  # [kz, ky, kx, inC, outC]

        # Convolution
        deriv5 = tf.nn.conv3d(fpad5, kernel, strides=[1, 1, 1, 1, 1], padding="VALID")

        # Back to [nz, ny, nx]
        return deriv5[0, ..., 0]

    DZ2 = tf.concat([dz[0:1], dz[:-1] + dz[1:], dz[-1:]], axis=0)

    Uk = tf.pad(U[:, :, :], [[1, 1], [0, 0], [0, 0]], "SYMMETRIC")
    Vk = tf.pad(V[:, :, :], [[1, 1], [0, 0], [0, 0]], "SYMMETRIC")

    # Derivatives
    Exx = deriv4(U, axis=2, d=dx)  # du/dx
    Eyy = deriv4(V, axis=1, d=dx)  # dv/dy

    # Incompressibility assumption for vertical strain rate
    Ezz = -Exx - Eyy

    # Shear components
    Exy = 0.5 * (deriv4(U, axis=1, d=dx) + deriv4(V, axis=2, d=dx))
    Exz = 0.5 * (Uk[2:, :, :] - Uk[:-2, :, :]) / tf.maximum(DZ2, thr)
    Eyz = 0.5 * (Vk[2:, :, :] - Vk[:-2, :, :]) / tf.maximum(DZ2, thr)

    # # Mask out layers with invalid dz
    # mask = tf.cast(tf.greater(dz, 1.0), tf.dtype)  # shape [nz]
    # mask = tf.reshape(mask, [-1,1,1])             # broadcast to 3D
    # Exx *= mask
    # Eyy *= mask
    # Ezz *= mask
    # Exy *= mask
    # Exz *= mask
    # Eyz *= mask

    Exx = tf.where(DZ2 > 1, Exx, 0.0)
    Eyy = tf.where(DZ2 > 1, Eyy, 0.0)
    Ezz = tf.where(DZ2 > 1, Ezz, 0.0)
    Exy = tf.where(DZ2 > 1, Exy, 0.0)
    Exz = tf.where(DZ2 > 1, Exz, 0.0)
    Eyz = tf.where(DZ2 > 1, Eyz, 0.0)

    return Exx, Eyy, Ezz, Exy, Exz, Eyz
