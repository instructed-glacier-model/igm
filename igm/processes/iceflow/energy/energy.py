#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np 
import tensorflow as tf  

from igm.processes.iceflow.energy.utils import stag4
from igm.processes.iceflow.vert_disc import compute_levels, compute_dzeta, compute_dz
from igm.processes.iceflow.utils import X_to_fieldin, Y_to_UV 
import igm.processes.iceflow.energy as energy

def iceflow_energy(cfg, U, V, fieldin):

    thk, usurf, arrhenius, slidingco, dX = fieldin
 
    levels = compute_levels(cfg.processes.iceflow.numerics.Nz, 
                            cfg.processes.iceflow.numerics.vert_spacing)

    dz =  compute_dz(stag4(thk), levels)
    #dzeta =  compute_dzeta(levels)

    energy_list = []
    for component in cfg.processes.iceflow.physics.energy_components:
        func = getattr(energy, f"cost_{component}")
        energy_list.append(func(cfg, U, V, thk, usurf, arrhenius, slidingco, dX, dz))

    return energy_list

def iceflow_energy_XY(cfg, X, Y):
    
    U, V = Y_to_UV(cfg, Y)

    fieldin = X_to_fieldin(cfg, X)

    return iceflow_energy(cfg, U, V, fieldin)
