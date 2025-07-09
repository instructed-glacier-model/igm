#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from igm.processes.iceflow.energy.utils import stag4, stag8, psia
from igm.utils.gradient.compute_gradient_stag import compute_gradient_stag

def cost_gravity(cfg, U, V, thk, usurf, arrhenius, slidingco, dX, zeta, dzeta):

    exp_glen = cfg.processes.iceflow.physics.exp_glen
    ice_density = cfg.processes.iceflow.physics.ice_density
    gravity_cst = cfg.processes.iceflow.physics.gravity_cst
    fnge = cfg.processes.iceflow.physics.force_negative_gravitational_energy
    Nz = cfg.processes.iceflow.numerics.Nz

    return _cost_gravity(U, V, usurf, dX, zeta, dzeta, thk, Nz, ice_density, gravity_cst, fnge, exp_glen)

@tf.function()
def _cost_gravity(U, V, usurf, dX, zeta, dzeta, thk, Nz, ice_density, gravity_cst, fnge, exp_glen):
    
    slopsurfx, slopsurfy = compute_gradient_stag(usurf, dX, dX)

    if not (Nz == 2):
    
        slopsurfx = tf.expand_dims(slopsurfx, axis=1)
        slopsurfy = tf.expand_dims(slopsurfy, axis=1)
    
        uds = stag8(U) * slopsurfx + stag8(V) * slopsurfy 

        COND = ( (thk[:, 1:, 1:] > 0) & (thk[:, 1:, :-1] > 0)
               & (thk[:, :-1, 1:] > 0) & (thk[:, :-1, :-1] > 0) )
        COND = tf.expand_dims(COND, axis=1)

        uds = tf.where(COND, uds, 0.0)
 
    else:
     
        Um = stag4(U)
        Vm = stag4(V)

        uds = ( tf.expand_dims(Um[:, 0, :, :],1) \
            + tf.expand_dims(Um[:, -1, :, :]-Um[:, 0, :, :],1) * psia(zeta,exp_glen) ) \
            * tf.expand_dims(slopsurfx,1) \
            + ( tf.expand_dims(Vm[:, 0, :, :],1) \
            + tf.expand_dims(Vm[:, -1, :, :]-Vm[:, 0, :, :],1) * psia(zeta,exp_glen) ) \
            * tf.expand_dims(slopsurfy,1)
        
    if fnge:
        uds = tf.minimum(uds, 0.0) # force non-postiveness

    # C_slid is unit Mpa m^-1 m/y m = Mpa m/y
    return (
        ice_density
        * gravity_cst
        * 10 ** (-6)
        * stag4(thk) 
        * tf.reduce_sum(dzeta * uds, axis=1)
    ) 
 