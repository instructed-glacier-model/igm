#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Any, Dict
from omegaconf import DictConfig

from igm.common import State
from igm.utils.math.precision import normalize_precision

from igm.processes.iceflow.utils.data_preprocessing import fieldin_to_X_2d
from igm.processes.iceflow.utils.velocities import (
    get_velbase,
    get_velsurf,
    get_velbar,
    clip_max_velbar,
)


class EvaluatorParams(tf.experimental.ExtensionType):
    """Parameters for ice flow evaluator."""

    Nz: int
    force_max_velbar: float


def get_evaluator_params_args(cfg: DictConfig) -> Dict[str, Any]:
    """Extract evaluator parameters from configuration."""

    cfg_numerics = cfg.processes.iceflow.numerics

    return {
        "Nz": cfg_numerics.Nz,
        "force_max_velbar": cfg.processes.iceflow.force_max_velbar,
    }


def get_kwargs_from_state(state: State) -> Dict[str, Any]:
    """Extract keyword arguments needed for evaluation from state."""

    return {
        "thk": state.thk,
        "mapping": state.iceflow.mapping,
        "V_bar": state.iceflow.discr_v.V_bar,
        "V_b": state.iceflow.discr_v.V_b,
        "V_s": state.iceflow.discr_v.V_s,
    }


def get_evaluator_inputs_from_state(cfg: DictConfig, state: State) -> tf.Tensor:
    """Prepare input tensor from state variables for ice flow evaluation."""

    cfg_unified = cfg.processes.iceflow.unified

    inputs = [vars(state)[input] for input in cfg_unified.inputs]

    inputs = tf.stack(inputs, axis=-1)
    inputs = fieldin_to_X_2d(inputs)

    # not sure if this is needed but need to verify double precision in other places...
    dtype = normalize_precision(cfg.processes.iceflow.numerics.precision)
    inputs = tf.cast(inputs, dtype)

    return inputs


@tf.function(jit_compile=True)
def evaluator_iceflow(
    inputs: tf.Tensor, parameters: EvaluatorParams, **kwargs: Dict[str, Any]
) -> Dict[str, tf.Tensor]:
    """Evaluate ice flow model to compute velocity fields and derived quantities."""

    # Compute velocity from mapping
    U, V = kwargs["mapping"].get_UV(inputs)
    U, V = U[0], V[0]

    # Post-processing of velocity fields
    U = tf.where(kwargs["thk"] > 0.0, U, 0.0)
    V = tf.where(kwargs["thk"] > 0.0, V, 0.0)

    if parameters.force_max_velbar > 0.0:
        U, V = clip_max_velbar(U, V, kwargs["V_bar"], parameters.force_max_velbar)

    # Retrieve derived quantities from velocity fields
    uvelbase, vvelbase = get_velbase(U, V, kwargs["V_b"])
    uvelsurf, vvelsurf = get_velsurf(U, V, kwargs["V_s"])
    ubar, vbar = get_velbar(U, V, kwargs["V_bar"])

    result = {
        "U": U,
        "V": V,
        "uvelbase": uvelbase,
        "vvelbase": vvelbase,
        "uvelsurf": uvelsurf,
        "vvelsurf": vvelsurf,
        "ubar": ubar,
        "vbar": vbar,
    }
    return {k: tf.cast(v, tf.float32) for k, v in result.items()}


def evaluate_iceflow(cfg: DictConfig, state: State) -> None:
    """Evaluate ice flow model and update velocity state."""

    # Get inputs for mapping
    inputs = get_evaluator_inputs_from_state(cfg, state)

    # Get kwargs for evaluator
    kwargs = get_kwargs_from_state(state)

    # Evaluate ice-flow model
    evaluator_params = state.iceflow.evaluator_params
    update = evaluator_iceflow(inputs, evaluator_params, **kwargs)

    # Update velocity state
    for key, value in update.items():
        setattr(state, key, value)
