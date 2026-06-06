#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file


import tensorflow as tf

from igm.utils.grad.grad import grad_xy
from igm.utils.grad.compute_divflux import compute_divflux
from igm.processes.iceflow.utils.vertical_discretization import compute_levels
from igm.processes.iceflow.vertical_velocity.vertical_velocity_legendre import (
    compute_vertical_velocity_legendre,
)
from igm.processes.iceflow.vertical_velocity.vertical_velocity_v2 import (
    compute_vertical_velocity_twolayers,
)


def compute_vertical_velocity_v1(cfg, state):

    basis_vertical = cfg.processes.iceflow.numerics.basis_vertical.lower()

    if basis_vertical == "lagrange":
        method = cfg.processes.iceflow.vertical_velocity.method.lower()
        if method == "kinematic":
            return compute_vertical_velocity_kinematic_v1(cfg, state)
        elif method == "incompressibility":
            return compute_vertical_velocity_incompressibility_v1(cfg, state)
        else:
            raise ValueError(f"❌ Unknown method: <{method}>.")

    elif basis_vertical == "legendre":
        return compute_vertical_velocity_legendre(cfg, state)

    elif basis_vertical == "molho":
        return compute_vertical_velocity_twolayers(cfg, state)

    else:
        raise ValueError(f"❌ Unsupported vertical basis: <{basis_vertical}>.")


def compute_vertical_velocity_kinematic_v1(cfg, state):

    # implementation GJ

    # use the formula w = u dot \nabla l + \nable \cdot (u l)

    # get the vertical thickness layers    # )

    levels = compute_levels(
        cfg.processes.iceflow.numerics.Nz, cfg.processes.iceflow.numerics.vert_spacing
    )

    temd = levels[1:] - levels[:-1]
    dz = tf.stack([state.thk * z for z in temd], axis=0)

    sloptopgx, sloptopgy = grad_xy(state.topg, state.dX, state.dX, False, "extrapolate")

    sloplayx = [sloptopgx]
    sloplayy = [sloptopgy]
    divfl = [tf.zeros_like(state.thk)]

    for l in range(1, state.U.shape[0]):

        cumdz = tf.reduce_sum(dz[:l], axis=0)

        sx, sy = grad_xy(state.topg + cumdz, state.dx, state.dx, False, "extrapolate")

        sloplayx.append(sx)
        sloplayy.append(sy)

        ub = tf.reduce_sum(state.vert_weight[:l] * state.U[:l], axis=0) / tf.reduce_sum(
            state.vert_weight[:l], axis=0
        )
        vb = tf.reduce_sum(state.vert_weight[:l] * state.V[:l], axis=0) / tf.reduce_sum(
            state.vert_weight[:l], axis=0
        )
        div = compute_divflux(ub, vb, cumdz, state.dx, state.dx, method="centered")

        divfl.append(div)

    sloplayx = tf.stack(sloplayx, axis=0)
    sloplayy = tf.stack(sloplayy, axis=0)
    divfl = tf.stack(divfl, axis=0)

    W = state.U * sloplayx + state.V * sloplayy - divfl

    return W


def compute_vertical_velocity_incompressibility_v1(cfg, state):

    # implementation GJ

    # Compute horinzontal derivatives
    dUdx = (state.U[:, :, 2:] - state.U[:, :, :-2]) / (2 * state.dX[0, 0])
    dVdy = (state.V[:, 2:, :] - state.V[:, :-2, :]) / (2 * state.dX[0, 0])

    dUdx = tf.pad(dUdx, [[0, 0], [0, 0], [1, 1]], "CONSTANT")
    dVdy = tf.pad(dVdy, [[0, 0], [1, 1], [0, 0]], "CONSTANT")

    dUdx = (dUdx[1:] + dUdx[:-1]) / 2  # compute between the layers
    dVdy = (dVdy[1:] + dVdy[:-1]) / 2  # compute between the layers

    # get dVdz from impcrompressibility condition
    dVdz = -dUdx - dVdy

    # get the basal vertical velocities
    sloptopgx, sloptopgy = grad_xy(state.topg, state.dX, state.dX, False, "extrapolate")
    wvelbase = state.U[0] * sloptopgx + state.V[0] * sloptopgy

    # get the vertical thickness layers
    levels = compute_levels(
        cfg.processes.iceflow.numerics.Nz, cfg.processes.iceflow.numerics.vert_spacing
    )

    temd = levels[1:] - levels[:-1]
    dz = tf.stack([state.thk * z for z in temd], axis=0)

    W = []
    W.append(wvelbase)
    for l in range(dVdz.shape[0]):
        W.append(W[-1] + dVdz[l] * dz[l])
    W = tf.stack(W)

    return W
