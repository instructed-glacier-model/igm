#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from igm.processes.iceflow.energy.utils import stag4h, stag2v, psia, legendre_basis
from igm.utils.gradient.compute_gradient import compute_gradient

def cost_gravity(cfg, U, V, fieldin, vert_disc, staggered_grid):

    thk, usurf, arrhenius, slidingco, dX = fieldin
    zeta, dzeta, Leg_P, Leg_dPdz = vert_disc

    exp_glen = cfg.processes.iceflow.physics.exp_glen
    ice_density = cfg.processes.iceflow.physics.ice_density
    gravity_cst = cfg.processes.iceflow.physics.gravity_cst
    fnge = cfg.processes.iceflow.physics.force_negative_gravitational_energy
    vert_basis = cfg.processes.iceflow.numerics.vert_basis

    return _cost_gravity(U, V, usurf, dX, zeta, dzeta, thk, Leg_P,
                         ice_density, gravity_cst, fnge, exp_glen, staggered_grid, vert_basis)

@tf.function()
def _cost_gravity(U, V, usurf, dX, zeta, dzeta, thk, Leg_P,
                  ice_density, gravity_cst, fnge, exp_glen, staggered_grid, vert_basis):
     
    slopsurfx, slopsurfy = compute_gradient(usurf, dX, dX, staggered_grid)  
 
    if staggered_grid:
        U = stag4h(U)
        V = stag4h(V)
        thk = stag4h(thk)

    if vert_basis == "Lagrange":

        U = stag2v(U)
        V = stag2v(V)

    elif vert_basis == "Legendre":
 
        U = tf.einsum('ij,bjkl->bikl', Leg_P, U)
        V = tf.einsum('ij,bjkl->bikl', Leg_P, V)
    
    elif vert_basis == "SIA":
 
        U = U[:, 0:1, :, :] + (U[:, -1:, :, :] - U[:, 0:1, :, :]) \
                            * psia(zeta[None, :, None, None], exp_glen)
        V = V[:, 0:1, :, :] + (V[:, -1:, :, :] - V[:, 0:1, :, :]) \
                            * psia(zeta[None, :, None, None], exp_glen)
 
    else:
        raise ValueError(f"Unknown vertical basis: {vert_basis}")
    
    uds = U * slopsurfx[:, None, :, :] + V * slopsurfy[:, None, :, :] 
  
    if fnge:
        uds = tf.minimum(uds, 0.0) # force non-postiveness

#    uds = tf.where(thk[:, None, :, :]>0, uds, 0.0)

    # C_slid is unit Mpa m^-1 m/y m = Mpa m/y
    return (
        ice_density
        * gravity_cst
        * 10 ** (-6)
        * thk
        * tf.reduce_sum(dzeta[None, :, None, None] * uds, axis=1)
    ) 
 