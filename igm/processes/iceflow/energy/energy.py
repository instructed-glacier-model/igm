#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Dict, List

from .components import EnergyComponent
from igm.processes.iceflow.horizontal import HorizontalDiscr
from igm.processes.iceflow.vertical import VerticalDiscr
from igm.processes.iceflow.utils.data_preprocessing import X_to_fieldin, Y_to_UV


def iceflow_energy(
    U: tf.Tensor,
    V: tf.Tensor,
    fieldin: Dict[str, tf.Tensor],
    discr_h: HorizontalDiscr,
    discr_v: VerticalDiscr,
    energy_components: List[EnergyComponent],
    batch_size: int,
    Ny: int,
    Nx: int,
) -> tf.TensorArray:

    dtype = U.dtype
    element_shape = (batch_size, Ny - 1, Nx - 1)
    size = len(energy_components)

    energies = tf.TensorArray(dtype=dtype, size=size, element_shape=element_shape)

    for i, component in enumerate(energy_components):
        output = component.cost(U, V, fieldin, discr_h, discr_v)
        energies = energies.write(i, output)

    energies = energies.stack()

    return energies


@tf.function()
def iceflow_energy_XY(
    Nz: int,
    fieldin_names: List[str],
    X: tf.Tensor,
    Y: tf.Tensor,
    discr_h: HorizontalDiscr,
    discr_v: VerticalDiscr,
    energy_components: List[EnergyComponent],
    batch_size: int,
    Ny: int,
    Nx: int,
) -> tf.TensorArray:

    U, V = Y_to_UV(Nz, Y)
    fieldin = X_to_fieldin(X=X, fieldin_names=fieldin_names)

    return iceflow_energy(
        U, V, fieldin, discr_h, discr_v, energy_components, batch_size, Ny, Nx
    )


@tf.function()
def iceflow_energy_UV(
    inputs_names: List[str],
    inputs: tf.Tensor,
    U: tf.Tensor,
    V: tf.Tensor,
    discr_h: HorizontalDiscr,
    discr_v: VerticalDiscr,
    energy_components: List[EnergyComponent],
) -> tf.TensorArray:

    inputs = tf.cast(inputs, U.dtype)
    fieldin = X_to_fieldin(X=inputs, fieldin_names=inputs_names)

    Ny = inputs.shape[1]
    Nx = inputs.shape[2]
    batch_size = inputs.shape[0]

    return iceflow_energy(
        U, V, fieldin, discr_h, discr_v, energy_components, batch_size, Ny, Nx
    )
