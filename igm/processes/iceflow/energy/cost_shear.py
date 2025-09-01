#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from igm.processes.iceflow.energy.utils import stag4h, stag2v, psia, psiap
from igm.utils.gradient.compute_gradient import compute_gradient

def cost_shear(cfg, U, V, fieldin, vert_disc, staggered_grid):

    thk, usurf, arrhenius, slidingco, dX = fieldin
    zeta, dzeta, Leg_P, Leg_dPdz = vert_disc

    exp_glen = cfg.processes.iceflow.physics.exp_glen
    regu_glen = cfg.processes.iceflow.physics.regu_glen
    thr_ice_thk = cfg.processes.iceflow.physics.thr_ice_thk
    min_sr = cfg.processes.iceflow.physics.min_sr
    max_sr = cfg.processes.iceflow.physics.max_sr
    vert_basis = cfg.processes.iceflow.numerics.vert_basis

    return _cost_shear(U, V, thk, usurf, arrhenius, slidingco, dX, zeta, dzeta, Leg_P, Leg_dPdz,
                       exp_glen, regu_glen, thr_ice_thk, min_sr, max_sr,  staggered_grid, vert_basis)

@tf.function()
def compute_horizontal_derivatives(U, V, dx, staggered_grid):

    if staggered_grid:

        dUdx = (U[..., :, :, 1:] - U[..., :, :, :-1]) / dx
        dVdx = (V[..., :, :, 1:] - V[..., :, :, :-1]) / dx
        dUdy = (U[..., :, 1:, :] - U[..., :, :-1, :]) / dx
        dVdy = (V[..., :, 1:, :] - V[..., :, :-1, :]) / dx

        dUdx = (dUdx[..., :, :-1, :] + dUdx[..., :, 1:, :]) / 2
        dVdx = (dVdx[..., :, :-1, :] + dVdx[..., :, 1:, :]) / 2
        dUdy = (dUdy[..., :, :, :-1] + dUdy[..., :, :, 1:]) / 2
        dVdy = (dVdy[..., :, :, :-1] + dVdy[..., :, :, 1:]) / 2
    
    else:

        paddings = [[0, 0]] * (len(U.shape) - 2) + [[1, 1], [1, 1]]
        U = tf.pad(U, paddings, mode="SYMMETRIC")
        V = tf.pad(V, paddings, mode="SYMMETRIC")

        dUdx = (U[..., :, 1:-1, 2:] - U[..., :, 1:-1, :-2]) / (2 * dx)
        dVdx = (V[..., :, 1:-1, 2:] - V[..., :, 1:-1, :-2]) / (2 * dx)
        dUdy = (U[..., :, 2:, 1:-1] - U[..., :, :-2, 1:-1]) / (2 * dx)
        dVdy = (V[..., :, 2:, 1:-1] - V[..., :, :-2, 1:-1]) / (2 * dx)

    return dUdx, dVdx, dUdy, dVdy

@tf.function()
def compute_srxy2(dUdx, dVdx, dUdy, dVdy):

    Exx = dUdx
    Eyy = dVdy
    Ezz = -dUdx - dVdy
    Exy = 0.5 * dVdx + 0.5 * dUdy
    
    return 0.5 * ( Exx**2 + Exy**2 + Exy**2 + Eyy**2 + Ezz**2 )

@tf.function()
def compute_srz2(dUdz, dVdz):
 
    Exz = 0.5 * dUdz
    Eyz = 0.5 * dVdz
    
    return 0.5 * ( Exz**2 + Eyz**2 + Exz**2 + Eyz**2 )

@tf.function()
def compute_vertical_derivatives(U, V, thk, dzeta, thr):
     
    if U.shape[-3] > 1:  
        dUdz = (U[:, 1:, :, :] - U[:, :-1, :, :]) \
            / (dzeta[None, :, None, None] * tf.maximum(thk, thr))
        dVdz = (V[:, 1:, :, :] - V[:, :-1, :, :]) \
            / (dzeta[None, :, None, None] * tf.maximum(thk, thr))
    else: 
        dUdz = tf.zeros_like(U)
        dVdz = tf.zeros_like(V)
    
    return dUdz, dVdz

def dampen_vertical_derivatives_where_floating(dUdz, dVdz, slidingco, sc=0.01):
     
    dUdz = tf.where(slidingco[:, None, :, :] > 0, dUdz, sc * dUdz)
    dVdz = tf.where(slidingco[:, None, :, :] > 0, dVdz, sc * dVdz)

    return dUdz, dVdz

@tf.function()
def correct_for_change_of_coordinate(dUdx, dVdx, dUdy, dVdy, dUdz, dVdz, sloptopgx, sloptopgy):
    # This correct for the change of coordinate z -> z - b

    dUdx = dUdx - dUdz * sloptopgx[:, None, :, :]
    dUdy = dUdy - dUdz * sloptopgy[:, None, :, :]
    dVdx = dVdx - dVdz * sloptopgx[:, None, :, :]
    dVdy = dVdy - dVdz * sloptopgy[:, None, :, :]

    return dUdx, dVdx, dUdy, dVdy
 
@tf.function()
def _cost_shear(U, V, thk, usurf, arrhenius, slidingco, dX, zeta, dzeta, Leg_P, Leg_dPdz,
                exp_glen, regu_glen, thr_ice_thk, min_sr, max_sr, staggered_grid, vert_basis):
    
    # B has Unit Mpa y^(1/n)
    B = 2.0 * arrhenius ** (-1.0 / exp_glen)
    if len(B.shape) == 3:
        B = B[:,None, :, :]
    p = 1.0 + 1.0 / exp_glen

    dUdx, dVdx, dUdy, dVdy = compute_horizontal_derivatives(U, V, dX[0,0,0], staggered_grid) 

    # TODO : sloptopgx, sloptopgy must be the elevaion of layers! not the bedrock, little effects?
    sloptopgx, sloptopgy = compute_gradient(usurf - thk, dX, dX, staggered_grid) 

    # compute the horizontal average, these quantitites will be used for vertical derivatives
    if staggered_grid:
        U = stag4h(U)
        V = stag4h(V)
        slidingco = stag4h(slidingco)
        thk = stag4h(thk)
        B = stag4h(B)

    if vert_basis == "Lagrange":

        dUdx = stag2v(dUdx) 
        dVdx = stag2v(dVdx) 
        dUdy = stag2v(dUdy) 
        dVdy = stag2v(dVdy) 
        B    = stag2v(B)   

        dUdz, dVdz = compute_vertical_derivatives(U, V, thk, dzeta, thr=thr_ice_thk) 

        dUdz, dVdz = dampen_vertical_derivatives_where_floating(dUdz, dVdz, slidingco)  

        dUdx, dVdx, dUdy, dVdy = correct_for_change_of_coordinate(dUdx, dVdx, dUdy, dVdy, dUdz, dVdz,
                                                                  sloptopgx, sloptopgy)  

    elif vert_basis == "Legendre":
 
        dUdx = tf.einsum('ij,bjkl->bikl', Leg_P, dUdx) 
        dVdx = tf.einsum('ij,bjkl->bikl', Leg_P, dVdx)
        dUdy = tf.einsum('ij,bjkl->bikl', Leg_P, dUdy)
        dVdy = tf.einsum('ij,bjkl->bikl', Leg_P, dVdy)

        dUdz = tf.einsum('ij,bjkl->bikl', Leg_dPdz, U) / tf.maximum(thk[:, None, :, :], thr_ice_thk) 
        dVdz = tf.einsum('ij,bjkl->bikl', Leg_dPdz, V) / tf.maximum(thk[:, None, :, :], thr_ice_thk)
 
    elif vert_basis == "SIA":

        dUdx = dUdx[:, 0:1, :, :] + (dUdx[:, -1:, :, :] - dUdx[:, 0:1, :, :]) \
                                  * psia(zeta[None, :, None, None], exp_glen)
        dVdy = dVdy[:, 0:1, :, :] + (dVdy[:, -1:, :, :] - dVdy[:, 0:1, :, :]) \
                                  * psia(zeta[None, :, None, None], exp_glen)
        dUdy = dUdy[:, 0:1, :, :] + (dUdy[:, -1:, :, :] - dUdy[:, 0:1, :, :]) \
                                  * psia(zeta[None, :, None, None], exp_glen)
        dVdx = dVdx[:, 0:1, :, :] + (dVdx[:, -1:, :, :] - dVdx[:, 0:1, :, :]) \
                                  * psia(zeta[None, :, None, None], exp_glen)

        dUdz = (U[:, -1:, :, :] - U[:, 0:1, :, :]) \
             * psiap(zeta[None, :, None, None], exp_glen) / tf.maximum(thk[:, None, :, :], thr_ice_thk)
        dVdz = (V[:, -1:, :, :] - V[:, 0:1, :, :]) \
             * psiap(zeta[None, :, None, None], exp_glen) / tf.maximum(thk[:, None, :, :], thr_ice_thk)

    else:
        raise ValueError(f"Unknown vertical basis: {vert_basis}")

    sr2 = compute_srxy2(dUdx, dVdx, dUdy, dVdy) + compute_srz2(dUdz, dVdz)  

    sr2capped = tf.clip_by_value(sr2, min_sr**2, max_sr**2)

#    sr2 = tf.where(thk[:, None, :, :]>0, sr2, 0.0) 

    p_term = ((sr2capped + regu_glen**2) ** ((p-2) / 2)) * sr2 / p 
 
    # C_shear is unit  Mpa y^(1/n) y^(-1-1/n) * m = Mpa m/y
    return thk * tf.reduce_sum( B * dzeta[None, :, None, None] * p_term, axis=1)  
 
