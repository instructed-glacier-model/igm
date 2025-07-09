#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from igm.processes.iceflow.energy.utils import stag4
from igm.utils.gradient.compute_gradient_stag import compute_gradient_stag

def cost_sliding_weertman(cfg, U, V, thk, usurf, arrhenius, slidingco, dX, dz):

    exp_weertman = cfg.processes.iceflow.physics.exp_weertman
    regu_weertman = cfg.processes.iceflow.physics.regu_weertman

    return _cost_sliding(U, V, thk, usurf, slidingco, dX,
                         exp_weertman, regu_weertman)

@tf.function()
def _cost_sliding(U, V, thk, usurf, slidingco, dX, exp_weertman, regu_weertman):
 
    C = 1.0 * slidingco  # C has unit Mpa y^m m^(-m) 
 
    s = 1.0 + 1.0 / exp_weertman
  
    sloptopgx, sloptopgy = compute_gradient_stag(usurf - thk, dX, dX)

    # C_slid is unit Mpa y^m m^(-m) * m^(1+m) * y^(-1-m)  = Mpa  m/y
    N = (
        stag4(U[:, 0, :, :] ** 2 + V[:, 0, :, :] ** 2)
        + regu_weertman**2
        + (stag4(U[:, 0, :, :]) * sloptopgx + stag4(V[:, 0, :, :]) * sloptopgy) ** 2
    )
    C_slid = stag4(C) * N ** (s / 2) / s

    return C_slid