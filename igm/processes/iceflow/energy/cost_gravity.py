#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from igm.processes.iceflow.energy.utils import stag4h, stag2v, psia, legendre_basis
from igm.utils.gradient.compute_gradient import compute_gradient

def cost_gravity(cfg, U, V, W, P, fieldin, vert_disc, staggered_grid):

    thk, usurf, arrhenius, slidingco, dX = fieldin
    zeta, dzeta, Leg_P, Leg_dPdz = vert_disc

    exp_glen = cfg.processes.iceflow.physics.exp_glen
    ice_density = cfg.processes.iceflow.physics.ice_density
    gravity_cst = cfg.processes.iceflow.physics.gravity_cst
    fnge = cfg.processes.iceflow.physics.force_negative_gravitational_energy
    vert_basis = cfg.processes.iceflow.numerics.vert_basis

    return _cost_gravity(U, V, W, usurf, dX, zeta, dzeta, thk, Leg_P,
                         ice_density, gravity_cst, fnge, exp_glen, staggered_grid, vert_basis)

@tf.function()
def _cost_gravity(U, V, W, usurf, dX, zeta, dzeta, thk, Leg_P,
                  ice_density, gravity_cst, fnge, exp_glen, staggered_grid, vert_basis):
     
#    slopsurfx, slopsurfy = compute_gradient(usurf, dX, dX, staggered_grid)  
 
    if staggered_grid:
        U = stag4h(U)
        V = stag4h(V)
        W = stag4h(W)
        thk = stag4h(thk)

    if vert_basis == "Lagrange":
        U = stag2v(U)
        V = stag2v(V)
        W = stag2v(W)

    elif vert_basis == "Legendre":
         raise ValueError(f"Unknown vertical basis: {vert_basis}")
    
    elif vert_basis == "SIA":
        raise ValueError(f"Unknown vertical basis: {vert_basis}") 
    
    else:
        raise ValueError(f"Unknown vertical basis: {vert_basis}")
    
    uds = W
  
    # if fnge:
    #     uds = tf.minimum(uds, 0.0) # force non-postiveness

#    uds = tf.where(thk[:, None, :, :]>0, uds, 0.0)

    # C_slid is unit Mpa m^-1 m/y m = Mpa m/y
    return (
        ice_density
        * gravity_cst
        * 10 ** (-6)
        * thk
        * tf.reduce_sum(dzeta[None, :, None, None] * uds, axis=1)
    ) 
 