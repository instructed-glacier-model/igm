#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np 
import tensorflow as tf  

from igm.processes.iceflow.utils import X_to_fieldin, Y_to_UV 
import igm.processes.iceflow.energy as energy

def iceflow_energy(cfg, U, V, fieldin, vert_disc):

    energy_list = []
    for component in cfg.processes.iceflow.physics.energy_components:
        func = getattr(energy, f"cost_{component}")
        if cfg.processes.iceflow.numerics.staggered_grid in [1,2]:
            energy_list.append(func(cfg, U, V, fieldin, vert_disc, 1))
        if cfg.processes.iceflow.numerics.staggered_grid in [0,2]:
            energy_list.append(func(cfg, U, V, fieldin, vert_disc, 0))

    return energy_list

def iceflow_energy_XY(cfg, X, Y, vert_disc):
    
    U, V = Y_to_UV(cfg, Y)

    fieldin = X_to_fieldin(cfg, X)

    return iceflow_energy(cfg, U, V, fieldin, vert_disc)
