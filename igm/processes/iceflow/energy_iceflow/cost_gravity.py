#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from igm.processes.iceflow.energy_iceflow.utils import stag4, stag8
from igm.processes.iceflow.energy_iceflow.utils import compute_gradient_stag

@tf.function()
def cost_gravity(U, V, usurf, dX, dz, COND, Nz, ice_density, gravity_cst, 
                   force_negative_gravitational_energy):

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
