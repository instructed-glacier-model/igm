#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from igm.processes.iceflow.energy.utils import stag4h, stag2v, psia, psiap
from igm.utils.gradient.compute_gradient import compute_gradient

def cost_shear(cfg, U, V, thk, usurf, arrhenius, slidingco, dX, zeta, dzeta):

    exp_glen = cfg.processes.iceflow.physics.exp_glen
    regu_glen = cfg.processes.iceflow.physics.regu_glen
    thr_ice_thk = cfg.processes.iceflow.physics.thr_ice_thk
    min_sr = cfg.processes.iceflow.physics.min_sr
    max_sr = cfg.processes.iceflow.physics.max_sr
    Nz = cfg.processes.iceflow.numerics.Nz
    staggered_grid = cfg.processes.iceflow.numerics.staggered_grid
    vert_basis = cfg.processes.iceflow.numerics.vert_basis

    return _cost_shear(U, V, thk, usurf, arrhenius, slidingco, dX, zeta, dzeta, 
                       exp_glen, regu_glen, thr_ice_thk, min_sr, max_sr, Nz, staggered_grid, vert_basis)

@tf.function()
def compute_horizontal_derivatives(U, V, dX, staggered_grid):

    if staggered_grid:

        dUdx = (U[:, :, :, 1:] - U[:, :, :, :-1]) / dX[0, 0, 0]
        dVdx = (V[:, :, :, 1:] - V[:, :, :, :-1]) / dX[0, 0, 0]
        dUdy = (U[:, :, 1:, :] - U[:, :, :-1, :]) / dX[0, 0, 0]
        dVdy = (V[:, :, 1:, :] - V[:, :, :-1, :]) / dX[0, 0, 0]

        dUdx = (dUdx[:, :, :-1, :] + dUdx[:, :, 1:, :]) / 2
        dVdx = (dVdx[:, :, :-1, :] + dVdx[:, :, 1:, :]) / 2
        dUdy = (dUdy[:, :, :, :-1] + dUdy[:, :, :, 1:]) / 2
        dVdy = (dVdy[:, :, :, :-1] + dVdy[:, :, :, 1:]) / 2
    
    else:

        UU = tf.pad(U, [[0, 0], [0, 0], [1, 1], [1, 1]], "SYMMETRIC")
        VV = tf.pad(V, [[0, 0], [0, 0], [1, 1], [1, 1]], "SYMMETRIC")

        dUdx = (UU[:, :, 1:-1, 2:] - UU[:, :, 1:-1, :-2]) / (2 * dX[0, 0, 0])
        dVdx = (VV[:, :, 1:-1, 2:] - VV[:, :, 1:-1, :-2]) / (2 * dX[0, 0, 0])
        dUdy = (UU[:, :, 2:, 1:-1] - UU[:, :, :-2, 1:-1]) / (2 * dX[0, 0, 0])
        dVdy = (VV[:, :, 2:, 1:-1] - VV[:, :, :-2, 1:-1]) / (2 * dX[0, 0, 0])

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
def compute_vertical_derivatives(Um, Vm, thk, dzeta, thr):
     
    if Um.shape[-3] > 1:  
        dUdz = (Um[:, 1:, :, :] - Um[:, :-1, :, :]) / (dzeta * tf.maximum(thk, thr))
        dVdz = (Vm[:, 1:, :, :] - Vm[:, :-1, :, :]) / (dzeta * tf.maximum(thk, thr))
    else: 
        dUdz = tf.zeros_like(Um)
        dVdz = tf.zeros_like(Vm)
    
    return dUdz, dVdz

def dampen_vertical_derivatives_where_floating(dUdz, dVdz, slidingco):
    
    slc = tf.expand_dims(slidingco, axis=1)
    dUdz = tf.where(slc > 0, dUdz, 0.01 * dUdz)
    dVdz = tf.where(slc > 0, dVdz, 0.01 * dVdz)

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
def compute_horizontal_derivatives_2layers(dUdx, dVdx, dUdy, dVdy, zeta, exp_glen):
        
    dUdx = tf.expand_dims(dUdx[:, 0, :, :],1) \
        + tf.expand_dims(dUdx[:, -1, :, :]-dUdx[:, 0, :, :],1) * psia(zeta,exp_glen)
    dVdy = tf.expand_dims(dVdy[:, 0, :, :],1) \
        + tf.expand_dims(dVdy[:, -1, :, :]-dVdy[:, 0, :, :],1) * psia(zeta,exp_glen)
    dUdy = tf.expand_dims(dUdy[:, 0, :, :],1) \
        + tf.expand_dims(dUdy[:, -1, :, :]-dUdy[:, 0, :, :],1) * psia(zeta,exp_glen)
    dVdx = tf.expand_dims(dVdx[:, 0, :, :],1) \
        + tf.expand_dims(dVdx[:, -1, :, :]-dVdx[:, 0, :, :],1) * psia(zeta,exp_glen)
    
    return dUdx, dVdx, dUdy, dVdy
    
@tf.function()
def compute_vertical_derivatives_2layers(U, V, thk, zeta, exp_glen):
    
    dUdz = tf.expand_dims(U[:, -1, :, :]-U[:, 0, :, :],1) \
        * psiap(zeta,exp_glen) / tf.maximum( thk , 1)
    dVdz = tf.expand_dims(V[:, -1, :, :]-V[:, 0, :, :],1) \
        * psiap(zeta,exp_glen) / tf.maximum( thk , 1)
        
    return dUdz, dVdz
 
@tf.function()
def _cost_shear(U, V, thk, usurf, arrhenius, slidingco, dX, zeta, dzeta, 
                exp_glen, regu_glen, thr_ice_thk, min_sr, max_sr, Nz, staggered_grid, vert_basis):
    
    # B has Unit Mpa y^(1/n)
    B = 2.0 * arrhenius ** (-1.0 / exp_glen)
    if len(B.shape) == 3:
        B = B[:,None, :, :]
    p = 1.0 + 1.0 / exp_glen

    dUdx, dVdx, dUdy, dVdy = compute_horizontal_derivatives(U, V, dX, staggered_grid) 

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

        print("Warning: Legendre basis is not implemented for shear stress cost function, using Lagrange instead.") 
    
    elif vert_basis == "SIA":

        dUdx, dVdx, dUdy, dVdy = compute_horizontal_derivatives_2layers(dUdx, dVdx, dUdy, dVdy, zeta, exp_glen)
        
        dUdz, dVdz = compute_vertical_derivatives_2layers(U, V, thk, zeta, exp_glen)

    else:
        raise ValueError(f"Unknown vertical basis: {vert_basis}")

    sr2 = compute_srxy2(dUdx, dVdx, dUdy, dVdy) + compute_srz2(dUdz, dVdz)  

    sr2 = tf.clip_by_value(sr2, min_sr**2, max_sr**2)

#    sr2 = tf.where(thk[:, None, :, :]>0, sr2, 0.0) 

    p_term = ((sr2 + regu_glen**2) ** ((p-2) / 2)) * sr2 / p 
 
    # C_shear is unit  Mpa y^(1/n) y^(-1-1/n) * m = Mpa m/y
    return thk * tf.reduce_sum( B * dzeta * p_term, axis=1)  
 
