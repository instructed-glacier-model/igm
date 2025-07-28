#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np 
import tensorflow as tf  

from igm.processes.iceflow.utils import X_to_fieldin, Y_to_UV 
import igm.processes.iceflow.energy as energy

def iceflow_energy(cfg, U, V, fieldin, vert_disc):

    energy_tensor = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    for i, component in enumerate(cfg.processes.iceflow.physics.energy_components):
        func = getattr(energy, f"cost_{component}")
        if cfg.processes.iceflow.numerics.staggered_grid in [1,2]:
            output = func(cfg, U, V, fieldin, vert_disc, 1)
            energy_tensor = energy_tensor.write(i, output)
        if cfg.processes.iceflow.numerics.staggered_grid in [0,2]:
            output = func(cfg, U, V, fieldin, vert_disc, 0)
            energy_tensor = energy_tensor.write(i, output)
    
    energy_tensor = energy_tensor.stack()
    
    return energy_tensor

def iceflow_energy_XY(cfg, X, Y, vert_disc):
    
    U, V = Y_to_UV(cfg.processes.iceflow.numerics.Nz, Y)

    # fieldin = X_to_fieldin(cfg, X)
    fieldin = X_to_fieldin(X=X,
                           fieldin_names=cfg.processes.iceflow.emulator.fieldin,
                           dim_arrhenius=cfg.processes.iceflow.physics.dim_arrhenius,
                           Nz=cfg.processes.iceflow.numerics.Nz)

    return iceflow_energy(cfg, U, V, fieldin, vert_disc)
