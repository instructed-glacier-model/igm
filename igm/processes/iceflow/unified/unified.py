#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from omegaconf import DictConfig

from igm.common.core import State
from .mappings import Mappings, InterfaceMappings
from .optimizers import Optimizers, InterfaceOptimizers
from .evaluator import EvaluatorParams, get_evaluator_params_args, evaluate_iceflow
from .solver import solve_iceflow
from .utils import get_cost_fn

from igm.processes.iceflow.data_preparation.data_preprocessing import (
    PreparationParams,
    calculate_expected_dimensions,
    get_input_params_args,
)

from igm.processes.iceflow.data_preparation.patching import OverlapPatching


def initialize_iceflow_unified(cfg: DictConfig, state: State) -> None:

    # Initialize training set
    preparation_params_args = get_input_params_args(cfg)
    preparation_params = PreparationParams(**preparation_params_args)

    state.iceflow.preparation_params = preparation_params

    state.iceflow.patching = OverlapPatching(
        patch_size=preparation_params.patch_size, overlap=preparation_params.overlap
    )

    # Initialize mapping
    mapping_name = cfg.processes.iceflow.unified.mapping
    mapping_args = InterfaceMappings[mapping_name].get_mapping_args(cfg, state)
    mapping = Mappings[mapping_name](**mapping_args)
    state.iceflow.mapping = mapping

    # Initialize optimizer
    optimizer_name = cfg.processes.iceflow.unified.optimizer
    optimizer_args = InterfaceOptimizers[optimizer_name].get_optimizer_args(
        cfg=cfg, cost_fn=get_cost_fn(cfg, state), map=mapping
    )
    optimizer = Optimizers[optimizer_name](**optimizer_args)
    state.iceflow.optimizer = optimizer

    # Evaluator params
    evaluator_params_args = get_evaluator_params_args(cfg)
    evaluator_params = EvaluatorParams(**evaluator_params_args)
    state.iceflow.evaluator_params = evaluator_params

    # Solve once
    solve_iceflow(cfg, state, init=True)

    # Evaluate once
    evaluate_iceflow(cfg, state)


def update_iceflow_unified(cfg: DictConfig, state: State) -> None:

    # Solve ice flow
    solve_iceflow(cfg, state)

    # Evalute ice flow
    evaluate_iceflow(cfg, state)
