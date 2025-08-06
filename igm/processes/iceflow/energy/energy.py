#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np 
import tensorflow as tf  

from igm.processes.iceflow.utils import X_to_fieldin, Y_to_UV 
import igm.processes.iceflow.energy as energy

def iceflow_energy(U, V, fieldin, vert_disc, energy_components, staggered_grid):

    if staggered_grid == 2:
        energy_tensor_length = 2 * len(energy_components)
    else:
        energy_tensor_length = len(energy_components)

    staggered_shape = (1, U.shape[2] - 1, U.shape[3] - 1)
    nonstaggered_shape = (1, U.shape[2], U.shape[3])
    
    energy_tensor_staggered = tf.TensorArray(dtype=tf.float32, size=energy_tensor_length, element_shape=staggered_shape)
    energy_tensor_nonstaggered = tf.TensorArray(dtype=tf.float32, size=energy_tensor_length, element_shape=nonstaggered_shape) # do not make this dynamic for some reason... (slice dimension issue with XLA)

    i = 0
    for component in energy_components:
        if staggered_grid in [1,2]:
            output = component.cost(U, V, fieldin, vert_disc, 1)
            energy_tensor_staggered = energy_tensor_staggered.write(i, output)
            i += 1
        if staggered_grid in [0,2]:
            output = component.cost(U, V, fieldin, vert_disc, 0)
            energy_tensor_nonstaggered = energy_tensor_nonstaggered.write(i, output)
            i += 1
    
    energy_tensor_nonstaggered = energy_tensor_nonstaggered.stack()
    energy_tensor_staggered = energy_tensor_staggered.stack()
    
    return energy_tensor_nonstaggered, energy_tensor_staggered

@tf.function()
def iceflow_energy_XY(Nz, dim_arrhenius, staggered_grid, fieldin_names, X, Y, vert_disc, energy_components):
    
    U, V = Y_to_UV(Nz, Y)
    fieldin = X_to_fieldin(X=X,
                           fieldin_names=fieldin_names,
                           dim_arrhenius=dim_arrhenius,
                           Nz=Nz)


    return iceflow_energy(U, V, fieldin, vert_disc, energy_components, staggered_grid)
