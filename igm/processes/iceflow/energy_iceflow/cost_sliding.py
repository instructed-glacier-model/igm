#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import tensorflow as tf
from igm.processes.iceflow.energy_iceflow.utils import stag4, compute_gradient_stag
from igm.processes.iceflow.energy_iceflow.utils import compute_gradient_stag

@tf.function()
def cost_sliding(U, V, thk, usurf, slidingco, dX, exp_weertman, regu_weertman, new_friction_param):
 
    if new_friction_param:
        C = 1.0 * slidingco  # C has unit Mpa y^m m^(-m) 
    else:
        if exp_weertman == 1:
            # C has unit Mpa y^m m^(-m)
            C = 1.0 * slidingco
        else:
            C = (slidingco + 10 ** (-12)) ** -(1.0 / exp_weertman)
 
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