#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from igm.processes.iceflow.energy.utils import stag4, stag8
from igm.processes.iceflow.energy.utils import compute_gradient_stag
from igm.processes.iceflow.energy.utils import gauss_points_and_weights
from igm.processes.iceflow.energy.utils import psia, psiap

def cost_shear(cfg, U, V, thk, usurf, arrhenius, slidingco, dX, dz):

    exp_glen = cfg.processes.iceflow.physics.exp_glen
    regu_glen = cfg.processes.iceflow.physics.regu_glen

    if cfg.processes.iceflow.numerics.Nz == 2:
  
        return _cost_shear_2layers(thk, arrhenius, U, V, dX, exp_glen, regu_glen)

    else:
   
        thr_ice_thk = cfg.processes.iceflow.physics.thr_ice_thk
        min_sr = cfg.processes.iceflow.physics.min_sr
        max_sr = cfg.processes.iceflow.physics.max_sr
 
        return _cost_shear(U, V, thk, usurf, arrhenius, slidingco, dX, dz, 
                            exp_glen, regu_glen, thr_ice_thk, min_sr, max_sr)

##########################

@tf.function()
def compute_horizontal_strainrate_Glen_tf(U, V, dX):

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
def compute_strainrate_Glen_tf(U, V, thk, slidingco, dX, ddz, sloptopgx, sloptopgy, thr):
    
    dUdx, dVdx, dUdy, dVdy = compute_horizontal_strainrate_Glen_tf(U, V, dX)

    # compute the horizontal average, these quantitites will be used for vertical derivatives
    Um = stag4(U)
    Vm = stag4(V)

    # homgenize sizes in the vertical plan on the stagerred grid
    if U.shape[1] > 1:
        dUdx = (dUdx[:, :-1, :, :] + dUdx[:, 1:, :, :]) / 2
        dVdx = (dVdx[:, :-1, :, :] + dVdx[:, 1:, :, :]) / 2
        dUdy = (dUdy[:, :-1, :, :] + dUdy[:, 1:, :, :]) / 2
        dVdy = (dVdy[:, :-1, :, :] + dVdy[:, 1:, :, :]) / 2
 
        # vertical derivative if there is at least two layears
        dUdz = (Um[:, 1:, :, :] - Um[:, :-1, :, :]) / tf.maximum(ddz, thr)
        dVdz = (Vm[:, 1:, :, :] - Vm[:, :-1, :, :]) / tf.maximum(ddz, thr)
        slc = tf.expand_dims(stag4(slidingco), axis=1)
        dUdz = tf.where(slc > 0, dUdz, 0.01 * dUdz)
        dVdz = tf.where(slc > 0, dVdz, 0.01 * dVdz)
    else:
        # zero otherwise
        dUdz = 0.0
        dVdz = 0.0

    # This correct for the change of coordinate z -> z - b
    dUdx = dUdx - dUdz * sloptopgx
    dUdy = dUdy - dUdz * sloptopgy
    dVdx = dVdx - dVdz * sloptopgx
    dVdy = dVdy - dVdz * sloptopgy

    Exx = dUdx
    Eyy = dVdy
    Ezz = -dUdx - dVdy
    Exy = 0.5 * dVdx + 0.5 * dUdy
    Exz = 0.5 * dUdz
    Eyz = 0.5 * dVdz
    
    return 0.5 * ( Exx**2 + Exy**2 + Exy**2 + Eyy**2 + Ezz**2 + Exz**2 + Eyz**2 + Exz**2 + Eyz**2 )
 

@tf.function()
def _cost_shear(U, V, thk, usurf, arrhenius, slidingco, dX, dz, 
                    exp_glen, regu_glen, thr_ice_thk, min_sr, max_sr):
    
    COND = ( (thk[:, 1:, 1:] > 0) & (thk[:, 1:, :-1] > 0)
            & (thk[:, :-1, 1:] > 0) & (thk[:, :-1, :-1] > 0) )
    COND = tf.expand_dims(COND, axis=1)
  
    # B has Unit Mpa y^(1/n)
    B = 2.0 * arrhenius ** (-1.0 / exp_glen)

    sloptopgx, sloptopgy = compute_gradient_stag(usurf - thk, dX, dX)
    sloptopgx = tf.expand_dims(sloptopgx, axis=1)
    sloptopgy = tf.expand_dims(sloptopgy, axis=1)
    # TODO : sloptopgx, sloptopgy must be the elevaion of layers! not the bedrock,
    #  this probably has very little effects.

    p = 1.0 + 1.0 / exp_glen

    # sr has unit y^(-1)
    sr = compute_strainrate_Glen_tf(
        U, V, thk, slidingco, dX, dz, sloptopgx, sloptopgy, thr=thr_ice_thk
    )
     
    sr = tf.where(COND, sr, 0.0)
    
    srcapped = tf.clip_by_value(sr, min_sr**2, max_sr**2)

    srcapped = tf.where(COND, srcapped, 0.0)
  
    # C_shear is unit  Mpa y^(1/n) y^(-1-1/n) * m = Mpa m/y
    if len(B.shape) == 3:
        C_shear = stag4(B) * tf.reduce_sum(dz * ((srcapped + regu_glen**2) ** ((p-2) / 2)) * sr, axis=1 ) / p
    else:
        C_shear = tf.reduce_sum( stag8(B) * dz * ((srcapped + regu_glen**2) ** ((p-2) / 2)) * sr, axis=1 ) / p

    return C_shear


# In the case of a 2 layers model, we assume a velcity profile is a SIA-like profile
@tf.function()
def _cost_shear_2layers(thk, arrhenius, U, V, dX, exp_glen, regu_glen):

    n, w = gauss_points_and_weights(ord_gauss=3)

    dUdx, dVdx, dUdy, dVdy = compute_horizontal_strainrate_Glen_tf(U, V, dX)

    Um = stag4(U)
    Vm = stag4(V)

    # B has Unit Mpa y^(1/n)
    B = 2.0 * arrhenius ** (-1.0 / exp_glen)
    p = 1.0 + 1.0 / exp_glen


    def p_term(zeta):
  
        UDX = (dUdx[:, 0, :, :] + (dUdx[:, -1, :, :]-dUdx[:, 0, :, :]) * psia(zeta,exp_glen))
        VDY = (dVdy[:, 0, :, :] + (dVdy[:, -1, :, :]-dVdy[:, 0, :, :]) * psia(zeta,exp_glen))
        UDY = (dUdy[:, 0, :, :] + (dUdy[:, -1, :, :]-dUdy[:, 0, :, :]) * psia(zeta,exp_glen))
        VDX = (dVdx[:, 0, :, :] + (dVdx[:, -1, :, :]-dVdx[:, 0, :, :]) * psia(zeta,exp_glen))
        UDZ = (Um[:, -1, :, :]-Um[:, 0, :, :]) * psiap(zeta,exp_glen) / tf.maximum( stag4(thk) , 1)
        VDZ = (Vm[:, -1, :, :]-Vm[:, 0, :, :]) * psiap(zeta,exp_glen) / tf.maximum( stag4(thk) , 1)
        
        Exx = UDX
        Eyy = VDY
        Ezz = - UDX - VDY
        Exy = 0.5 * VDX + 0.5 * UDY
        Exz = 0.5 * UDZ
        Eyz = 0.5 * VDZ
    
        sr2 = 0.5 * ( Exx**2 + Exy**2 + Exy**2 + Eyy**2 + Ezz**2 + Exz**2 + Eyz**2 + Exz**2 + Eyz**2 )

        return (sr2 + regu_glen**2) ** (p / 2) / p
  
    # C_shear is unit  Mpa y^(1/n) y^(-1-1/n) * m = Mpa m/y
    C_shear = stag4(B) * stag4(thk) * sum(w[i] * p_term(n[i]) for i in range(len(n)))

    return C_shear