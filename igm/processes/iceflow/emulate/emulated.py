#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from omegaconf import DictConfig
from typing import Dict, Any
import tensorflow as tf

from igm.common import State
from igm.processes.iceflow.utils.data_preprocessing import (
    Y_to_UV,
    prepare_X,
    fieldin_state_to_X,
)
from igm.processes.iceflow.utils.velocities import (
    get_velbase,
    get_velsurf,
    get_velbar,
    clip_max_velbar,
)


class EmulatedParams(tf.experimental.ExtensionType):
    Nz: int
    exclude_borders: int
    multiple_window_size: int
    force_max_velbar: float


def get_emulated_params_args(cfg: DictConfig) -> Dict[str, Any]:

    cfg_emulator = cfg.processes.iceflow.emulator
    cfg_numerics = cfg.processes.iceflow.numerics

    return {
        "Nz": cfg_numerics.Nz,
        "exclude_borders": cfg_emulator.exclude_borders,
        "multiple_window_size": cfg_emulator.network.multiple_window_size,
        "force_max_velbar": cfg.processes.iceflow.force_max_velbar,
    }


def get_emulated_bag(state: State) -> Dict[str, Any]:

    return {
        "thk": state.thk,
        "PAD": state.PAD,
        "V_b": state.iceflow.discr_v.V_b,
        "V_s": state.iceflow.discr_v.V_s,
        "V_bar": state.iceflow.discr_v.V_bar,
        "iceflow_model_inference": state.iceflow_model_inference,
    }


def update_iceflow_emulated(cfg: DictConfig, state: State) -> None:

    fieldin = fieldin_state_to_X(state, cfg.processes.iceflow.emulator.fieldin)
    X = prepare_X(cfg, fieldin, pertubate=False, split_into_patches=False)  # name-agnostic from here
    bag = get_emulated_bag(state)
    updated_variable_dict = update_emulated(bag, X, state.iceflow.emulated_params)

    for key, value in updated_variable_dict.items():
        setattr(state, key, value)


@tf.function(jit_compile=True)
def update_emulated(
    bag: Dict, X: tf.Tensor, parameters: EmulatedParams
) -> Dict[str, tf.Tensor]:

    if parameters.exclude_borders > 0:
        iz = parameters.exclude_borders
        X = tf.pad(X, [[0, 0], [iz, iz], [iz, iz], [0, 0]], "SYMMETRIC")

    if parameters.multiple_window_size > 0:
        Ny, Nx = bag["thk"].shape
        X = (tf.pad(X, bag["PAD"], "CONSTANT"))[:, :Ny, :Nx, :]

    # Compute output of neural network: Y
    Y = bag["iceflow_model_inference"](X)

    # Post-processing of output of neural network
    if parameters.exclude_borders > 0:
        iz = parameters.exclude_borders
        Y = Y[:, iz:-iz, iz:-iz, :]

    # Compute velocity fields: U, V
    U, V = Y_to_UV(parameters.Nz, Y)
    U = U[0]
    V = V[0]

    # Post-processing of velocity fields
    U = tf.where(bag["thk"] > 0.0, U, 0.0)
    V = tf.where(bag["thk"] > 0.0, V, 0.0)

    if parameters.force_max_velbar > 0.0:
        U, V = clip_max_velbar(
            U,
            V,
            bag["V_bar"],
            parameters.force_max_velbar,
        )

    # Retrieve derived quantities from velocity fields
    uvelbase, vvelbase = get_velbase(U, V, bag["V_b"])
    uvelsurf, vvelsurf = get_velsurf(U, V, bag["V_s"])
    ubar, vbar = get_velbar(U, V, bag["V_bar"])

    return {
        "U": U,
        "V": V,
        "uvelbase": uvelbase,
        "vvelbase": vvelbase,
        "uvelsurf": uvelsurf,
        "vvelsurf": vvelsurf,
        "ubar": ubar,
        "vbar": vbar,
    }
