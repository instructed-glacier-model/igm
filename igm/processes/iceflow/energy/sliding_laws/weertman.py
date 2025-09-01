#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from igm.processes.iceflow.energy.utils import stag4h
from igm.utils.gradient.compute_gradient import compute_gradient
from igm.processes.iceflow.utils import get_velbase
  
def weertman(U, V, thk, usurf, slidingco, dX, exp_weertman, regu_weertman, staggered_grid, vert_basis):
  
    C = 1.0 * slidingco  # C has unit Mpa y^m m^(-m) 
 
    s = 1.0 + 1.0 / exp_weertman

    sloptopgx, sloptopgy = compute_gradient(usurf - thk, dX, dX, staggered_grid) 
 
    if staggered_grid:
        U = stag4h(U)
        V = stag4h(V)
        C = stag4h(C)

    uvelbase, vvelbase = get_velbase(U, V, vert_basis)

    N = (uvelbase ** 2 + vvelbase ** 2) + regu_weertman**2 \
      + (uvelbase * sloptopgx + vvelbase * sloptopgy) ** 2
      
    basis_vectors = [uvelbase, vvelbase]

    sliding_shear_stress = [ C * N ** ((s - 2)/2) * uvelbase,
                             C * N ** ((s - 2)/2) * vvelbase ]
    
    return basis_vectors, sliding_shear_stress