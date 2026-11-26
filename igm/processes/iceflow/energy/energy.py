#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from typing import Dict, List

import tensorflow as tf

from igm.processes.iceflow.discretization import HorizontalDiscr, VerticalDiscr

from .components import EnergyComponent


@tf.function(jit_compile=True)
def X_to_fieldin(X: tf.Tensor, fieldin_names: List) -> Dict[str, tf.Tensor]:
    """Converts the input tensor X to a dictionary of fieldin variables.

    Name-agnostic. The dual-name handling for the friction field is on
    the consumer side, in `sliding.get_friction_field`.
    """
    channels = tf.unstack(X, num=len(fieldin_names), axis=-1)
    return dict(zip(fieldin_names, channels))


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
) -> tf.Tensor:

    return tf.stack(
        [
            component.cost(U, V, fieldin, discr_h, discr_v)
            for component in energy_components
        ],
        axis=0,
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

    fieldin = X_to_fieldin(X=inputs, fieldin_names=inputs_names)

    Ny = inputs.shape[1]
    Nx = inputs.shape[2]
    batch_size = inputs.shape[0]

    return iceflow_energy(
        U, V, fieldin, discr_h, discr_v, energy_components, batch_size, Ny, Nx
    )
