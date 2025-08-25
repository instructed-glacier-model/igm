#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from typing import Any, Dict, Tuple
import tensorflow as tf
import os
import warnings
import igm
from .mappings import Mappings
from .optimizers import Optimizers
from igm.processes.iceflow.emulate.utils.misc import get_pretrained_emulator_path
from igm.processes.iceflow.energy import (
    EnergyComponents,
    EnergyParams,
    get_energy_params_args,
)
from igm.processes.iceflow.energy.energy import iceflow_energy_UV

from .evaluator import EvaluatorParams, get_evaluator_params_args, evaluate_iceflow
from .solver import solve_iceflow


def initialize_iceflow_unified(cfg, state):

    cfg_unified = cfg.processes.iceflow.unified
    cfg_physics = cfg.processes.iceflow.physics

    # Temporary for now -- should be in optimizer init
    dir_path = get_pretrained_emulator_path(cfg, state)

    inputs = []
    fid = open(os.path.join(dir_path, "fieldin.dat"), "r")
    for fileline in fid:
        part = fileline.split()
        inputs.append(part[0])
    fid.close()
    assert cfg_unified.inputs == inputs
    state.iceflow_model = tf.keras.models.load_model(
        os.path.join(dir_path, "model.h5"), compile=False
    )
    state.iceflow_model.compile(jit_compile=True)

    # Initialize mapping
    mapping_type = cfg_unified.mapping
    mapping = Mappings[mapping_type](
        network=state.iceflow_model,
        Nz=cfg.processes.iceflow.numerics.Nz,
        output_scale=cfg_unified.network.output_scale,
    )
    state.mapping = mapping

    # Initialize energy components
    state.iceflow.energy_components = []
    for component in cfg_physics.energy_components:
        if component not in EnergyComponents:
            raise ValueError(f"❌ Unknown energy component: <{component}>.")

        # Get component and params class
        if component == "sliding":
            law = cfg_physics.sliding.law
            component_class = EnergyComponents[component][law]
            params_class = EnergyParams[component][law]
        else:
            component_class = EnergyComponents[component]
            params_class = EnergyParams[component]

        # Get args extractor
        get_params_args = get_energy_params_args[component]

        # Instantiate params and component classes
        params_args = get_params_args(cfg)
        params = params_class(**params_args)
        component_obj = component_class(params)

        # Add component to the list of components
        state.iceflow.energy_components.append(component_obj)

    # Initialize optimizer
    cfg_numerics = cfg.processes.iceflow.numerics

    def cost_fn(U: tf.Tensor, V: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        nonstaggered_energy, staggered_energy = iceflow_energy_UV(
            Nz=cfg_numerics.Nz,
            dim_arrhenius=cfg_physics.dim_arrhenius,
            staggered_grid=cfg_numerics.staggered_grid,
            inputs_names=tuple(cfg_unified.inputs),
            inputs=input,
            U=U,
            V=V,
            vert_disc=state.vert_disc,
            energy_components=state.iceflow.energy_components,
        )

        energy_mean_staggered = tf.reduce_mean(staggered_energy, axis=[1, 2, 3])
        energy_mean_nonstaggered = tf.reduce_mean(nonstaggered_energy, axis=[1, 2, 3])

        total_energy = tf.reduce_sum(energy_mean_nonstaggered, axis=0) + tf.reduce_sum(
            energy_mean_staggered, axis=0
        )

        return total_energy

    optimizer = Optimizers[cfg_unified.optimizer](
        cost_fn=cost_fn, map=state.mapping, lr=cfg_unified.lr
    )
    state.optimizer = optimizer

    # Evaluator params
    evaluator_params_args = get_evaluator_params_args(cfg)
    evaluator_params = EvaluatorParams(**evaluator_params_args)
    state.iceflow.evaluator_params = evaluator_params

    # Solve once
    solve_iceflow(cfg, state, init=True)

    # Evaluate once
    evaluate_iceflow(cfg, state)


def update_iceflow_unified(cfg, state):

    # Solve ice flow
    solve_iceflow(cfg, state)

    # Evalute ice flow
    evaluate_iceflow(cfg, state)
