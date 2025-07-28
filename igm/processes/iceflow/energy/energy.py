#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np 
import tensorflow as tf  

from igm.processes.iceflow.utils import X_to_fieldin, Y_to_UV 
import igm.processes.iceflow.energy as energy

def iceflow_energy(staggered_grid, U, V, fieldin, vert_disc, energy_components):

    energy_tensor = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    # state.iceflow.energy_components
    i = 0
    for component in energy_components:
        if staggered_grid in [1,2]:
            output = component.cost(U, V, fieldin, vert_disc, 1)
        if staggered_grid in [0,2]:
            output = component.cost(U, V, fieldin, vert_disc, 0)
        energy_tensor = energy_tensor.write(i, output)
        i += 1
    # for i, component in enumerate(cfg.processes.iceflow.physics.energy_components):
    #     func = getattr(energy, f"cost_{component}") # using energy_components object instead of this...
    #     energy_params = energy_params_dict[component]
        
    #     if cfg.processes.iceflow.numerics.staggered_grid in [1,2]:
    #         output = func(U, V, fieldin, vert_disc, 1, energy_params)
    #         energy_tensor = energy_tensor.write(i, output)
    #     if cfg.processes.iceflow.numerics.staggered_grid in [0,2]:
    #         output = func(U, V, fieldin, vert_disc, 0, energy_params)
    #         energy_tensor = energy_tensor.write(i, output)
    
    energy_tensor = energy_tensor.stack()
    
    return energy_tensor

@tf.function(jit_compile=True)
def iceflow_energy_XY(Nz, dim_arrhenius, staggered_grid, fieldin_names, X, Y, vert_disc, energy_components):
    
    U, V = Y_to_UV(Nz, Y)

    # fieldin = X_to_fieldin(cfg, X)
    fieldin = X_to_fieldin(X=X,
                           fieldin_names=fieldin_names,
                           dim_arrhenius=dim_arrhenius,
                           Nz=Nz)

    return iceflow_energy(staggered_grid, U, V, fieldin, vert_disc, energy_components)
