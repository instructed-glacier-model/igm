#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np 
import tensorflow as tf  

from .weertman import weertman, SlidingLaw
from igm.processes.iceflow.utils import X_to_fieldin, Y_to_UV  

# def sliding_law(cfg, U, V, fieldin): # -> OLD SETUP

#     thk, usurf, arrhenius, slidingco, dX = fieldin

#     exp_weertman = cfg.processes.iceflow.physics.exp_weertman
#     regu_weertman = cfg.processes.iceflow.physics.regu_weertman
#     staggered_grid = cfg.processes.iceflow.numerics.staggered_grid
#     vert_basis = cfg.processes.iceflow.numerics.vert_basis
 
#     if cfg.processes.iceflow.physics.sliding_law == "weertman":
#         return weertman(U, V, thk, usurf, slidingco, dX, exp_weertman, regu_weertman, staggered_grid, vert_basis)
#     else:
#         raise ValueError(f"Unknown sliding law: {cfg.processes.iceflow.physics.sliding_law}")


    
@tf.function(jit_compile=True)
def sliding_law_XY(X, Y, Nz, fieldin_list, dim_arrhenius, sliding_law: SlidingLaw):

    U, V = Y_to_UV(Nz, Y)

    fieldin = X_to_fieldin(X, fieldin_list, dim_arrhenius, Nz)
    
    return sliding_law(U, V, fieldin)
