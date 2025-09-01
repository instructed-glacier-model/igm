#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Any, Dict
from omegaconf import DictConfig

from igm.common.core import State
from igm.processes.iceflow.utils.data_preprocessing import (
    fieldin_to_X_2d,
    fieldin_to_X_3d,
    match_fieldin_dimensions,
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


def get_evaluator_params_args(cfg: DictConfig) -> Dict[str, Any]:

    cfg_numerics = cfg.processes.iceflow.numerics
    cfg_physics = cfg.processes.iceflow.physics

    return {
        "dim_arrhenius": cfg_physics.dim_arrhenius,
        "Nz": cfg_numerics.Nz,
        "force_max_velbar": cfg.processes.iceflow.force_max_velbar,
        "vertical_basis": cfg_numerics.vert_basis,
    }


def get_kwargs_from_state(state: State) -> Dict[str, Any]:

    return {
        "thk": state.thk,
        "vert_weight": state.vert_weight,
        "mapping": state.iceflow.mapping,
    }


def get_inputs_from_state(cfg: DictConfig, state: State) -> tf.Tensor:

    cfg_physics = cfg.processes.iceflow.physics
    cfg_unified = cfg.processes.iceflow.unified

    inputs = [vars(state)[input] for input in cfg_unified.inputs]

    if cfg_physics.dim_arrhenius == 3:
        inputs = match_fieldin_dimensions(inputs)
        inputs = fieldin_to_X_3d(cfg_physics.dim_arrhenius, inputs)
    elif cfg_physics.dim_arrhenius == 2:
        inputs = tf.stack(inputs, axis=-1)
        inputs = fieldin_to_X_2d(inputs)

    return inputs


@tf.function(jit_compile=True)
def evaluator_iceflow(
    inputs: tf.Tensor, parameters: EvaluatorParams, **kwargs: Dict[str, Any]
) -> Dict[str, tf.Tensor]:

    # Compute velocity from mapping
    U, V = kwargs["mapping"].get_UV(inputs)
    U, V = U[0], V[0]

    # Post-processing of velocity fields
    U = tf.where(kwargs["thk"] > 0.0, U, 0.0)
    V = tf.where(kwargs["thk"] > 0.0, V, 0.0)

    if parameters.force_max_velbar > 0.0:
        U, V = clip_max_velbar(
            U,
            V,
            parameters.force_max_velbar,
            parameters.vertical_basis,
            kwargs["vert_weight"],
        )

    # Retrieve derived quantities from velocity fields
    uvelbase, vvelbase = get_velbase(U, V, parameters.vertical_basis)
    uvelsurf, vvelsurf = get_velsurf(U, V, parameters.vertical_basis)
    ubar, vbar = get_velbar(U, V, kwargs["vert_weight"], parameters.vertical_basis)

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


def evaluate_iceflow(cfg: DictConfig, state: State) -> None:

    # Get inputs for mapping
    inputs = get_inputs_from_state(cfg, state)

    # Get kwargs for evaluator
    kwargs = get_kwargs_from_state(state)

    # Evaluate ice-flow model
    evaluator_params = state.iceflow.evaluator_params
    update = evaluator_iceflow(inputs, evaluator_params, **kwargs)

    # Update velocity state
    for key, value in update.items():
        setattr(state, key, value)
