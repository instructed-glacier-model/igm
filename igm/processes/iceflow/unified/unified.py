#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from omegaconf import DictConfig

from igm.common import State, print_model_with_inputs_detailed
from .mappings import Mappings, InterfaceMappings
from .optimizers import Optimizers, InterfaceOptimizers
from .evaluator import EvaluatorParams, get_evaluator_params_args, evaluate_iceflow
from .solver import solve_iceflow
from .utils import get_cost_fn
from ..utils.data_preprocessing import fieldin_state_to_X, X_to_fieldin


def initialize_iceflow_unified(cfg: DictConfig, state: State) -> None:
    """Initialize iceflow module in unified mode."""

    cfg_unified = cfg.processes.iceflow.unified

    # Initialize mapping
    mapping_name = cfg_unified.mapping
    mapping_args = InterfaceMappings[mapping_name].get_mapping_args(cfg, state)
    mapping = Mappings[mapping_name](**mapping_args)
    state.iceflow.mapping = mapping

    # Initialize optimizer
    optimizer_name = cfg_unified.optimizer
    optimizer_args = InterfaceOptimizers[optimizer_name].get_optimizer_args(
        cfg=cfg, cost_fn=get_cost_fn(cfg, state), map=mapping
    )
    # Clamp batch_size to actual number of patches
    if "batch_size" in optimizer_args:
        framesizemax = cfg_unified.data_preparation.framesizemax
        X = fieldin_state_to_X(cfg, state)
        ny, nx = int(X.shape[0]), int(X.shape[1])
        num_patches = (ny // framesizemax + 1) * (nx // framesizemax + 1)
        optimizer_args["batch_size"] = min(optimizer_args["batch_size"], num_patches)
    optimizer = Optimizers[optimizer_name](**optimizer_args)
    state.iceflow.optimizer = optimizer

    # Evaluator params
    evaluator_params_args = get_evaluator_params_args(cfg)
    evaluator_params = EvaluatorParams(**evaluator_params_args)
    state.iceflow.evaluator_params = evaluator_params

    if cfg_unified.mapping == "network" and cfg_unified.network.print_summary:
        X = fieldin_state_to_X(cfg, state)
        fieldin_dict = X_to_fieldin(X, fieldin_names=cfg_unified.inputs)
        print_model_with_inputs_detailed(
            model=state.iceflow_model,
            input_data=fieldin_dict,
            cfg_inputs=cfg_unified.inputs,
            normalization_method=cfg_unified.normalization.method,
        )

    # Solve once
    solve_iceflow(cfg, state, init=True)

    # Evaluate once
    evaluate_iceflow(cfg, state)


def update_iceflow_unified(cfg: DictConfig, state: State) -> None:
    """Update iceflow module in unified mode."""

    # Solve ice flow
    solve_iceflow(cfg, state)

    # Evalute ice flow
    evaluate_iceflow(cfg, state)
