#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf

from igm.processes.iceflow.utils.data_preprocessing import X_to_fieldin, Y_to_UV


def iceflow_energy(
    U, V, fieldin, vert_disc, energy_components, staggered_grid, batch_size, Ny, Nx
):
    if staggered_grid == 2:
        energy_tensor_length = 2 * len(energy_components)
    else:
        energy_tensor_length = len(energy_components)

    # Define element shapes for TensorArray with static values
    staggered_shape = (batch_size, Ny - 1, Nx - 1)
    nonstaggered_shape = (batch_size, Ny, Nx)

    energy_tensor_staggered = tf.TensorArray(
        dtype=tf.float32, size=energy_tensor_length, element_shape=staggered_shape
    )
    energy_tensor_nonstaggered = tf.TensorArray(
        dtype=tf.float32, size=energy_tensor_length, element_shape=nonstaggered_shape
    )  # do not make this dynamic for some reason... (slice dimension issue with XLA)

    i = 0
    for component in energy_components:
        if staggered_grid in [1, 2]:
            output = component.cost(U, V, fieldin, vert_disc, 1)
            energy_tensor_staggered = energy_tensor_staggered.write(i, output)
            i += 1
        if staggered_grid in [0, 2]:
            output = component.cost(U, V, fieldin, vert_disc, 0)
            energy_tensor_nonstaggered = energy_tensor_nonstaggered.write(i, output)
            i += 1

    energy_tensor_nonstaggered = energy_tensor_nonstaggered.stack()
    energy_tensor_staggered = energy_tensor_staggered.stack()

    return energy_tensor_nonstaggered, energy_tensor_staggered


@tf.function()
def iceflow_energy_XY(
    Nz,
    dim_arrhenius,
    staggered_grid,
    fieldin_names,
    X,
    Y,
    vert_disc,
    energy_components,
    batch_size,
    Ny,
    Nx,
):

    U, V = Y_to_UV(Nz, Y)
    fieldin = X_to_fieldin(
        X=X, fieldin_names=fieldin_names, dim_arrhenius=dim_arrhenius, Nz=Nz
    )

    return iceflow_energy(
        U, V, fieldin, vert_disc, energy_components, staggered_grid, batch_size, Ny, Nx
    )


@tf.function()
def iceflow_energy_UV(
    Nz,
    dim_arrhenius,
    staggered_grid,
    inputs_names,
    inputs,
    U,
    V,
    vert_disc,
    energy_components,
):

    fieldin = X_to_fieldin(
        X=inputs, fieldin_names=inputs_names, dim_arrhenius=dim_arrhenius, Nz=Nz
    )

    Ny = inputs.shape[1]
    Nx = inputs.shape[2]
    batch_size = inputs.shape[0]

    return iceflow_energy(
        U, V, fieldin, vert_disc, energy_components, staggered_grid, batch_size, Ny, Nx
    )
