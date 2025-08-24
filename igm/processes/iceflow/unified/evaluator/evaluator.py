#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from typing import Any, Dict, Tuple
import tensorflow as tf
import os
import warnings
import igm
from ..mappings import Mappings

from igm.processes.iceflow.utils.data_preprocessing import match_fieldin_dimensions
from igm.processes.iceflow.energy import (
    EnergyComponents,
    EnergyParams,
    get_energy_params_args,
)
from igm.processes.iceflow.utils.data_preprocessing import (
    fieldin_to_X_2d,
    fieldin_to_X_3d,
    Y_to_UV,
)
from igm.processes.iceflow.utils.velocities import (
    get_velbase,
    get_velsurf,
    get_velbar,
    clip_max_velbar,
)


class EvaluatorParams(tf.experimental.ExtensionType):
    Nz: int
    force_max_velbar: float
    vertical_basis: str
    dim_arrhenius: int


def get_evaluator_params_args(cfg) -> Dict[str, Any]:

    cfg_numerics = cfg.processes.iceflow.numerics
    cfg_physics = cfg.processes.iceflow.physics

    return {
        "dim_arrhenius": cfg_physics.dim_arrhenius,
        "Nz": cfg_numerics.Nz,
        "force_max_velbar": cfg.processes.iceflow.force_max_velbar,
        "vertical_basis": cfg_numerics.vert_basis,
    }


def get_data_from_state(state) -> Dict[str, Any]:

    return {
        "thk": state.thk,
        "vert_weight": state.vert_weight,
        "mapping": state.mapping,
    }


def get_inputs_from_state(cfg, state):

    cfg_physics = cfg.processes.iceflow.physics
    cfg_unified = cfg.processes.iceflow.unified

    if len(cfg_unified.inputs) != len(cfg_unified.inputs_scales):
        raise ValueError("❌ The inputs and input scales should have the same size.")

    inputs = [vars(state)[input] for input in cfg_unified.inputs]

    inputs = [input / scale for input, scale in zip(inputs, cfg_unified.inputs_scales)]

    if cfg_physics.dim_arrhenius == 3:
        inputs = match_fieldin_dimensions(inputs)
        inputs = fieldin_to_X_3d(cfg_physics.dim_arrhenius, inputs)
    elif cfg_physics.dim_arrhenius == 2:
        inputs = tf.stack(inputs, axis=-1)
        inputs = fieldin_to_X_2d(inputs)

    return inputs


@tf.function(jit_compile=True)
def evaluator_iceflow(
    inputs: tf.Tensor, data: Dict, parameters: EvaluatorParams
) -> Dict[str, tf.Tensor]:

    U, V = data["mapping"].get_UV(inputs)
    U, V = U[0], V[0]

    # Post-processing of velocity fields
    U = tf.where(data["thk"] > 0.0, U, 0.0)
    V = tf.where(data["thk"] > 0.0, V, 0.0)

    if parameters.force_max_velbar > 0.0:
        U, V = clip_max_velbar(
            U,
            V,
            parameters.force_max_velbar,
            parameters.vertical_basis,
            data["vert_weight"],
        )

    # Retrieve derived quantities from velocity fields
    uvelbase, vvelbase = get_velbase(U, V, parameters.vertical_basis)
    uvelsurf, vvelsurf = get_velsurf(U, V, parameters.vertical_basis)
    ubar, vbar = get_velbar(U, V, data["vert_weight"], parameters.vertical_basis)

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


def evaluate_iceflow(cfg, state):

    # Get inputs for mapping
    inputs = get_inputs_from_state(cfg, state)

    # Get data for evaluator
    data = get_data_from_state(state)

    # Evaluate ice-flow model
    evaluator_params = state.iceflow.evaluator_params
    update = evaluator_iceflow(inputs, data, evaluator_params)

    # Update velocity state
    for key, value in update.items():
        setattr(state, key, value)
