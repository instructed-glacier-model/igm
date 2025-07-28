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


    

def sliding_law_XY(cfg, X, Y, sliding_law: SlidingLaw):

    U, V = Y_to_UV(cfg.processes.iceflow.numerics.Nz, Y)

    fieldin = X_to_fieldin(X,
                           cfg.processes.iceflow.emulator.fieldin,
                           cfg.processes.iceflow.physics.dim_arrhenius,
                           cfg.processes.iceflow.numerics.Nz)
    
    return sliding_law(U, V, fieldin)
