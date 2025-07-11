#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from igm.processes.iceflow.energy.utils import stag4h
from igm.utils.gradient.compute_gradient import compute_gradient
  
def weertman(U, V, thk, usurf, slidingco, dX, exp_weertman, regu_weertman, staggered_grid):
  
    C = 1.0 * slidingco  # C has unit Mpa y^m m^(-m) 
 
    s = 1.0 + 1.0 / exp_weertman

    sloptopgx, sloptopgy = compute_gradient(usurf - thk, dX, dX, staggered_grid) 
 
    if staggered_grid:
        U = stag4h(U)
        V = stag4h(V)
        C = stag4h(C)

    U_basal = U[:, 0, :, :]
    V_basal = V[:, 0, :, :]

    N = (U_basal ** 2 + V_basal ** 2) + regu_weertman**2 \
      + (U_basal * sloptopgx + V_basal * sloptopgy) ** 2
      
    basis_vectors = [U_basal, V_basal]

    sliding_shear_stress = [ C * N ** ((s - 2)/2) * U_basal,
                             C * N ** ((s - 2)/2) * V_basal ]
    
    return basis_vectors, sliding_shear_stress