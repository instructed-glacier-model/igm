#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from igm.processes.iceflow.energy.utils import stag4, stag8, psia, psiap
from igm.utils.gradient.compute_gradient_stag import compute_gradient_stag

def cost_shear(cfg, U, V, thk, usurf, arrhenius, slidingco, dX, zeta, dzeta):

    exp_glen = cfg.processes.iceflow.physics.exp_glen
    regu_glen = cfg.processes.iceflow.physics.regu_glen
    thr_ice_thk = cfg.processes.iceflow.physics.thr_ice_thk
    min_sr = cfg.processes.iceflow.physics.min_sr
    max_sr = cfg.processes.iceflow.physics.max_sr
    Nz = cfg.processes.iceflow.numerics.Nz

    return _cost_shear(U, V, thk, usurf, arrhenius, slidingco, dX, zeta, dzeta, 
                        exp_glen, regu_glen, thr_ice_thk, min_sr, max_sr, Nz)

@tf.function()
def compute_horizontal_derivatives(U, V, dX):

    dUdx = (U[:, :, :, 1:] - U[:, :, :, :-1]) / dX[0, 0, 0]
    dVdx = (V[:, :, :, 1:] - V[:, :, :, :-1]) / dX[0, 0, 0]
    dUdy = (U[:, :, 1:, :] - U[:, :, :-1, :]) / dX[0, 0, 0]
    dVdy = (V[:, :, 1:, :] - V[:, :, :-1, :]) / dX[0, 0, 0]

    dUdx = (dUdx[:, :, :-1, :] + dUdx[:, :, 1:, :]) / 2
    dVdx = (dVdx[:, :, :-1, :] + dVdx[:, :, 1:, :]) / 2
    dUdy = (dUdy[:, :, :, :-1] + dUdy[:, :, :, 1:]) / 2
    dVdy = (dVdy[:, :, :, :-1] + dVdy[:, :, :, 1:]) / 2

    return dUdx, dVdx, dUdy, dVdy

@tf.function()
def average_vertically(dUdx, dVdx, dUdy, dVdy):

    dUdx = (dUdx[:, :-1, :, :] + dUdx[:, 1:, :, :]) / 2
    dVdx = (dVdx[:, :-1, :, :] + dVdx[:, 1:, :, :]) / 2
    dUdy = (dUdy[:, :-1, :, :] + dUdy[:, 1:, :, :]) / 2
    dVdy = (dVdy[:, :-1, :, :] + dVdy[:, 1:, :, :]) / 2

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
     
    if Um.shape[1] > 1:  
        dUdz = (Um[:, 1:, :, :] - Um[:, :-1, :, :]) / (dzeta * tf.maximum(stag4(thk), thr))
        dVdz = (Vm[:, 1:, :, :] - Vm[:, :-1, :, :]) / (dzeta * tf.maximum(stag4(thk), thr))
    else: 
        dUdz = 0.0
        dVdz = 0.0
    
    return dUdz, dVdz

def dampen_vertical_derivatives_where_floating(dUdz, dVdz, slidingco):
    
    slc = tf.expand_dims(stag4(slidingco), axis=1)
    dUdz = tf.where(slc > 0, dUdz, 0.01 * dUdz)
    dVdz = tf.where(slc > 0, dVdz, 0.01 * dVdz)

    return dUdz, dVdz

@tf.function()
def correct_for_change_of_coordinate(dUdx, dVdx, dUdy, dVdy, dUdz, dVdz, thk, dX, usurf):
     
    sloptopgx, sloptopgy = compute_gradient_stag(usurf - thk, dX, dX)
    sloptopgx = tf.expand_dims(sloptopgx, axis=1)
    sloptopgy = tf.expand_dims(sloptopgy, axis=1)
    # TODO : sloptopgx, sloptopgy must be the elevaion of layers! not the bedrock,
    #  this probably has very little effects.

    # This correct for the change of coordinate z -> z - b
    dUdx = dUdx - dUdz * sloptopgx
    dUdy = dUdy - dUdz * sloptopgy
    dVdx = dVdx - dVdz * sloptopgx
    dVdy = dVdy - dVdz * sloptopgy

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
def compute_vertical_derivatives_2layers(Um, Vm, thk, zeta, exp_glen):
    
    dUdz = tf.expand_dims(Um[:, -1, :, :]-Um[:, 0, :, :],1) \
        * psiap(zeta,exp_glen) / tf.maximum( stag4(thk) , 1)
    dVdz = tf.expand_dims(Vm[:, -1, :, :]-Vm[:, 0, :, :],1) \
        * psiap(zeta,exp_glen) / tf.maximum( stag4(thk) , 1)
        
    return dUdz, dVdz
 
@tf.function()
def _cost_shear(U, V, thk, usurf, arrhenius, slidingco, dX, zeta, dzeta, 
                    exp_glen, regu_glen, thr_ice_thk, min_sr, max_sr, Nz):
    
    # B has Unit Mpa y^(1/n)
    B = 2.0 * arrhenius ** (-1.0 / exp_glen)
    if len(B.shape) == 3:
        B = B[:,None, :, :]
    p = 1.0 + 1.0 / exp_glen

    dUdx, dVdx, dUdy, dVdy = compute_horizontal_derivatives(U, V, dX)

    # compute the horizontal average, these quantitites will be used for vertical derivatives
    Um = stag4(U)
    Vm = stag4(V)

    if not (Nz == 2):

        if (Nz > 2):
           dUdx, dVdx, dUdy, dVdy = average_vertically(dUdx, dVdx, dUdy, dVdy)

        dUdz, dVdz =  compute_vertical_derivatives(Um, Vm, thk, dzeta, thr=thr_ice_thk)

        dUdz, dVdz = dampen_vertical_derivatives_where_floating(dUdz, dVdz, slidingco)

        dUdx, dVdx, dUdy, dVdy = correct_for_change_of_coordinate(dUdx, dVdx, dUdy, dVdy, dUdz, dVdz, thk, dX, usurf)

    else: 
        dUdx, dVdx, dUdy, dVdy = compute_horizontal_derivatives_2layers(dUdx, dVdx, dUdy, dVdy, zeta, exp_glen)
        
        dUdz, dVdz = compute_vertical_derivatives_2layers(Um, Vm, thk, zeta, exp_glen)

    sr2 = compute_srxy2(dUdx, dVdx, dUdy, dVdy) + compute_srz2(dUdz, dVdz)

    sr2 = tf.clip_by_value(sr2, min_sr**2, max_sr**2)

#    sr2 = tf.where(tf.expand_dims(stag4(thk)>0, axis=1), sr2, 0.0)

    p_term = ((sr2 + regu_glen**2) ** ((p-2) / 2)) * sr2 / p 
 
    # C_shear is unit  Mpa y^(1/n) y^(-1-1/n) * m = Mpa m/y
    return stag4(thk) * tf.reduce_sum( stag8(B) * dzeta * p_term, axis=1)  
 
