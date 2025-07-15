#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from igm.processes.iceflow.energy.utils import stag4h
from igm.utils.gradient.compute_gradient import compute_gradient
from igm.processes.iceflow.utils import get_velbase

def cost_sliding_weertman(cfg, U, V, fieldin, vert_disc, staggered_grid):

    thk, usurf, arrhenius, slidingco, dX = fieldin
    zeta, dzeta, Leg_P, Leg_dPdz = vert_disc

    exp_weertman = cfg.processes.iceflow.physics.exp_weertman
    regu_weertman = cfg.processes.iceflow.physics.regu_weertman
    vert_basis = cfg.processes.iceflow.numerics.vert_basis

    return _cost_sliding(U, V, thk, usurf, slidingco, dX, zeta, dzeta,
                         exp_weertman, regu_weertman, staggered_grid, vert_basis)

@tf.function()
def _cost_sliding(U, V, thk, usurf, slidingco, dX, zeta, dzeta, \
                  exp_weertman, regu_weertman, staggered_grid, vert_basis):
 
    C = 1.0 * slidingco  # C has unit Mpa y^m m^(-m) 
 
    s = 1.0 + 1.0 / exp_weertman
  
    sloptopgx, sloptopgy = compute_gradient(usurf - thk, dX, dX, staggered_grid) 

    if staggered_grid:
        U = stag4h(U)
        V = stag4h(V)
        C = stag4h(C)

    uvelbase, vvelbase = get_velbase(U, V, vert_basis)

    N = ( (uvelbase ** 2 + vvelbase ** 2) + regu_weertman**2 \
        + (uvelbase * sloptopgx + vvelbase * sloptopgy) ** 2 )

    return C * N ** (s / 2) / s # C_slid is unit Mpa y^m m^(-m) * m^(1+m) * y^(-1-m)  = Mpa  m/y