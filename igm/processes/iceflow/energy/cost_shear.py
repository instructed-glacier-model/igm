#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from igm.processes.iceflow.energy.utils import stag4h, stag2v, psia, psiap
from igm.utils.gradient.compute_gradient import compute_gradient

def cost_shear(cfg, U, V, W, P, fieldin, vert_disc, staggered_grid):

    thk, usurf, arrhenius, slidingco, dX = fieldin
    zeta, dzeta, Leg_P, Leg_dPdz = vert_disc

    exp_glen = cfg.processes.iceflow.physics.exp_glen
    regu_glen = cfg.processes.iceflow.physics.regu_glen
    thr_ice_thk = cfg.processes.iceflow.physics.thr_ice_thk
    min_sr = cfg.processes.iceflow.physics.min_sr
    max_sr = cfg.processes.iceflow.physics.max_sr
    vert_basis = cfg.processes.iceflow.numerics.vert_basis

    return _cost_shear(U, V, W, thk, usurf, arrhenius, slidingco, dX, zeta, dzeta, Leg_P, Leg_dPdz,
                       exp_glen, regu_glen, thr_ice_thk, min_sr, max_sr,  staggered_grid, vert_basis)

@tf.function()
def compute_horizontal_derivatives(U, V, W, dx, staggered_grid):

    if staggered_grid:

        dUdx = (U[..., :, :, 1:] - U[..., :, :, :-1]) / dx
        dVdx = (V[..., :, :, 1:] - V[..., :, :, :-1]) / dx
        dWdx = (W[..., :, :, 1:] - W[..., :, :, :-1]) / dx
        dUdy = (U[..., :, 1:, :] - U[..., :, :-1, :]) / dx
        dVdy = (V[..., :, 1:, :] - V[..., :, :-1, :]) / dx
        dWdy = (W[..., :, 1:, :] - W[..., :, :-1, :]) / dx

        dUdx = (dUdx[..., :, :-1, :] + dUdx[..., :, 1:, :]) / 2
        dVdx = (dVdx[..., :, :-1, :] + dVdx[..., :, 1:, :]) / 2
        dWdx = (dWdx[..., :, :-1, :] + dWdx[..., :, 1:, :]) / 2
        dUdy = (dUdy[..., :, :, :-1] + dUdy[..., :, :, 1:]) / 2
        dVdy = (dVdy[..., :, :, :-1] + dVdy[..., :, :, 1:]) / 2
        dWdy = (dWdy[..., :, :, :-1] + dWdy[..., :, :, 1:]) / 2
    
    else:

        paddings = [[0, 0]] * (len(U.shape) - 2) + [[1, 1], [1, 1]]
        U = tf.pad(U, paddings, mode="SYMMETRIC")
        V = tf.pad(V, paddings, mode="SYMMETRIC")
        W = tf.pad(W, paddings, mode="SYMMETRIC")

        dUdx = (U[..., :, 1:-1, 2:] - U[..., :, 1:-1, :-2]) / (2 * dx)
        dVdx = (V[..., :, 1:-1, 2:] - V[..., :, 1:-1, :-2]) / (2 * dx)
        dWdx = (W[..., :, 1:-1, 2:] - W[..., :, 1:-1, :-2]) / (2 * dx)
        dUdy = (U[..., :, 2:, 1:-1] - U[..., :, :-2, 1:-1]) / (2 * dx)
        dVdy = (V[..., :, 2:, 1:-1] - V[..., :, :-2, 1:-1]) / (2 * dx)
        dWdy = (W[..., :, 2:, 1:-1] - W[..., :, :-2, 1:-1]) / (2 * dx)

    return dUdx, dVdx, dWdx, dUdy, dVdy, dWdy

@tf.function()
def compute_srx(dUdx, dVdx, dWdx, dUdy, dVdy, dWdy, dUdz, dVdz, dWdz):

    Exx = dUdx
    Eyy = dVdy
    Ezz = dWdz
    Exy = 0.5 * (dUdy + dVdx)
    Exz = 0.5 * (dUdz + dWdx)
    Eyz = 0.5 * (dVdz + dWdy)
    
    return 0.5 * ( Exx**2 + Exy**2 + Exz**2 + Exy**2 + Eyy**2 + Eyz**2 + Exz**2 + Eyz**2 + Ezz**2 )

@tf.function()
def compute_vertical_derivatives(U, V, W, thk, dzeta, thr):
     
    if U.shape[-3] > 1:  
        dUdz = (U[:, 1:, :, :] - U[:, :-1, :, :]) \
            / (dzeta[None, :, None, None] * tf.maximum(thk, thr))
        dVdz = (V[:, 1:, :, :] - V[:, :-1, :, :]) \
            / (dzeta[None, :, None, None] * tf.maximum(thk, thr))
        dWdz = (W[:, 1:, :, :] - W[:, :-1, :, :]) \
            / (dzeta[None, :, None, None] * tf.maximum(thk, thr))
    else: 
        dUdz = tf.zeros_like(U)
        dVdz = tf.zeros_like(V)
        dWdz = tf.zeros_like(W)

    return dUdz, dVdz, dWdz

@tf.function()
def compute_vertical_derivatives_1(W, thk, dzeta, thr):

    if W.shape[-3] > 1:
        dWdz = (W[:, 1:, :, :] - W[:, :-1, :, :]) \
            / (dzeta[None, :, None, None] * tf.maximum(thk, thr))
    else:
        dWdz = tf.zeros_like(W)

    return dWdz

def dampen_vertical_derivatives_where_floating(dUdz, dVdz, dWdz, slidingco, sc=0.01):

    dUdz = tf.where(slidingco[:, None, :, :] > 0, dUdz, sc * dUdz)
    dVdz = tf.where(slidingco[:, None, :, :] > 0, dVdz, sc * dVdz)
    dWdz = tf.where(slidingco[:, None, :, :] > 0, dWdz, sc * dWdz)

    return dUdz, dVdz, dWdz

@tf.function()
def correct_for_change_of_coordinate(dUdx, dVdx, dUdy, dVdy, dUdz, dVdz, sloptopgx, sloptopgy):
    # This correct for the change of coordinate z -> z - b

    dUdx = dUdx - dUdz * sloptopgx[:, None, :, :]
    dUdy = dUdy - dUdz * sloptopgy[:, None, :, :]
    dVdx = dVdx - dVdz * sloptopgx[:, None, :, :]
    dVdy = dVdy - dVdz * sloptopgy[:, None, :, :]

    return dUdx, dVdx, dUdy, dVdy
 
@tf.function()
def _cost_shear(U, V, W, thk, usurf, arrhenius, slidingco, dX, zeta, dzeta, Leg_P, Leg_dPdz,
                exp_glen, regu_glen, thr_ice_thk, min_sr, max_sr, staggered_grid, vert_basis):
    
    # B has Unit Mpa y^(1/n)
    B = 2.0 * arrhenius ** (-1.0 / exp_glen)
    if len(B.shape) == 3:
        B = B[:,None, :, :]
    p = 1.0 + 1.0 / exp_glen

    dUdx, dVdx, dWdx, dUdy, dVdy, dWdy = compute_horizontal_derivatives(U, V, W, dX[0,0,0], staggered_grid) 

    # TODO : sloptopgx, sloptopgy must be the elevaion of layers! not the bedrock, little effects?
    # sloptopgx, sloptopgy = compute_gradient(usurf - thk, dX, dX, staggered_grid) 

    # compute the horizontal average, these quantitites will be used for vertical derivatives
    if staggered_grid:
        U = stag4h(U)
        V = stag4h(V)
        W = stag4h(W)
        slidingco = stag4h(slidingco)
        thk = stag4h(thk)
        B = stag4h(B)

    if vert_basis == "Lagrange":

        dUdx = stag2v(dUdx) 
        dVdx = stag2v(dVdx) 
        dWdx = stag2v(dWdx)
        dUdy = stag2v(dUdy) 
        dVdy = stag2v(dVdy)
        dWdy = stag2v(dWdy)
        B    = stag2v(B)   

        dUdz, dVdz, dWdz = compute_vertical_derivatives(U, V, W, thk, dzeta, thr=thr_ice_thk) 

#        dUdz, dVdz, dWdz = dampen_vertical_derivatives_where_floating(dUdz, dVdz, dWdz, slidingco)  

#        dUdx, dVdx, dUdy, dVdy = correct_for_change_of_coordinate(dUdx, dVdx, dUdy, dVdy, dUdz, dVdz,
#                                                                  sloptopgx, sloptopgy)  

    elif vert_basis == "Legendre":
        raise ValueError(f"Unknown vertical basis: {vert_basis}")
    
    elif vert_basis == "SIA":
        raise ValueError(f"Unknown vertical basis: {vert_basis}")
    
    else:
        raise ValueError(f"Unknown vertical basis: {vert_basis}")
    
    # print("dUdz", dUdz.shape, "dVdz", dVdz.shape, "dWdz", dWdz.shape)
    # print("dUdx", dUdx.shape, "dVdx", dVdx.shape, "dUdy", dUdy.shape, "dVdy", dVdy.shape)
    # print("dWdx", dWdx.shape, "dWdy", dWdy.shape)

    sr2 = compute_srx(dUdx, dVdx, dWdx, dUdy, dVdy, dWdy, dUdz, dVdz, dWdz)

    sr2capped = tf.clip_by_value(sr2, min_sr**2, max_sr**2)

#    sr2 = tf.where(thk[:, None, :, :]>0, sr2, 0.0) 

    p_term = ((sr2capped + regu_glen**2) ** ((p-2) / 2)) * sr2 / p 
 
    # C_shear is unit  Mpa y^(1/n) y^(-1-1/n) * m = Mpa m/y
    return thk * tf.reduce_sum( B * dzeta[None, :, None, None] * p_term, axis=1)  
 
