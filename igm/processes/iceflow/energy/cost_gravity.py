#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from igm.processes.iceflow.energy.utils import stag4, stag8
from igm.processes.iceflow.energy.utils import compute_gradient_stag 
from igm.processes.iceflow.energy.utils import gauss_points_and_weights

def cost_gravity(cfg, U, V, thk, usurf, arrhenius, slidingco, dX, dz):

    exp_glen = cfg.processes.iceflow.physics.exp_glen
    ice_density = cfg.processes.iceflow.physics.ice_density
    gravity_cst = cfg.processes.iceflow.physics.gravity_cst
    force_negative_gravitational_energy = cfg.processes.iceflow.physics.force_negative_gravitational_energy
    Nz = cfg.processes.iceflow.numerics.Nz

    if cfg.processes.iceflow.numerics.Nz == 2:
        return _cost_gravity_2layers(U, V, thk, usurf, dX, exp_glen, ice_density, gravity_cst)
    
    else:

        return _cost_gravity(U, V, usurf, dX, dz, thk, Nz, ice_density, gravity_cst, 
                   force_negative_gravitational_energy)


@tf.function()
def _cost_gravity(U, V, usurf, dX, dz, thk, Nz, ice_density, gravity_cst, 
                   force_negative_gravitational_energy):
    
    
    COND = ( (thk[:, 1:, 1:] > 0) & (thk[:, 1:, :-1] > 0)
            & (thk[:, :-1, 1:] > 0) & (thk[:, :-1, :-1] > 0) )
    COND = tf.expand_dims(COND, axis=1)

    slopsurfx, slopsurfy = compute_gradient_stag(usurf, dX, dX)
    slopsurfx = tf.expand_dims(slopsurfx, axis=1)
    slopsurfy = tf.expand_dims(slopsurfy, axis=1)
 
    if Nz > 1:
        uds = stag8(U) * slopsurfx + stag8(V) * slopsurfy
    else:
        uds = stag4(U) * slopsurfx + stag4(V) * slopsurfy  

    if force_negative_gravitational_energy:
        uds = tf.minimum(uds, 0.0) # force non-postiveness

    uds = tf.where(COND, uds, 0.0)

    # C_slid is unit Mpa m^-1 m/y m = Mpa m/y
    C_grav = (
        ice_density
        * gravity_cst
        * 10 ** (-6)
        * tf.reduce_sum(dz * uds, axis=1)
    )

    return C_grav


@tf.function()
def _cost_gravity_2layers(U, V, thk, usurf, dX, exp_glen, ice_density, gravity_cst):

    n, w = gauss_points_and_weights(ord_gauss=3)

    slopsurfx, slopsurfy = compute_gradient_stag(usurf, dX, dX)
 
    Um = stag4(U)
    Vm = stag4(V)

#    slopsurfx = tf.clip_by_value( slopsurfx , -0.25, 0.25)
#    slopsurfy = tf.clip_by_value( slopsurfy , -0.25, 0.25)

    def f(zeta):
        return ( 1 - (1 - zeta) ** (exp_glen + 1) )

    def uds(zeta):

        return (Um[:, 0, :, :] + (Um[:, -1, :, :]-Um[:, 0, :, :]) * f(zeta)) * slopsurfx \
             + (Vm[:, 0, :, :] + (Vm[:, -1, :, :]-Vm[:, 0, :, :]) * f(zeta)) * slopsurfy
 
    # C_slid is unit Mpa m^-1 m/y m = Mpa m/y
    C_grav = (
        ice_density
        * gravity_cst
        * 10 ** (-6)
        * stag4(thk) 
        * sum(w[i] * uds(n[i]) for i in range(len(n)))
    )

    return C_grav 