#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from igm.processes.iceflow.energy_iceflow.utils import stag4, compute_gradient_stag, compute_average_velocity_twolayers_tf

@tf.function()
def cost_gravity_2layers(U, V, thk, usurf, dX, exp_glen, ice_density, gravity_cst, w, n):

    slopsurfx, slopsurfy = compute_gradient_stag(usurf, dX, dX)

    Um, Vm = compute_average_velocity_twolayers_tf(U, V)

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