#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file


import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State, print_model_with_inputs_detailed
from igm.processes.iceflow.discretization import HorizontalDiscrs, VerticalDiscrs
from igm.processes.iceflow.discretization.vertical.levels import define_vertical_weight
from igm.processes.iceflow.energy.cost import get_cost_fn
from igm.processes.iceflow.error_estimator import (
    ErrorEstimator,
    InterfaceErrorEstimator,
)
from igm.processes.iceflow.evaluator import (
    EvaluatorParams,
    evaluate_iceflow,
    get_evaluator_params_args,
)
from igm.processes.iceflow.init_stop import InitStop
from igm.processes.iceflow.mappings import InterfaceMappings, Mappings
from igm.processes.iceflow.optimizers import InterfaceOptimizers, Optimizers
from igm.processes.iceflow.solver import solve_iceflow
from igm.processes.iceflow.solver.solver import should_normalize
from igm.processes.iceflow.utils.fields import initialize_iceflow_fields
from igm.processes.iceflow.utils.inputs import patch_grid
from igm.processes.iceflow.vertical_velocity.vertical_velocity import (
    update as update_vertical_velocity,
)
from igm.utils.schedule import check_schedule, is_count, is_positive, parse_schedule


class Iceflow:
    """Container for the iceflow related states."""

    pass


def check_iceflow_method(cfg: DictConfig) -> None:
    """Only the unified method remains; refuse the removed ones explicitly."""
    method = str(cfg.processes.iceflow.method).lower()
    if method != "unified":
        raise ValueError(
            f"❌ processes.iceflow.method='{method}' is no longer available: the "
            "emulated, solved and diagnostic methods have been removed, and "
            "'unified' is the only ice-flow method.\n"
            "Set processes.iceflow.method: unified (or drop the key) and configure "
            "the solver under processes.iceflow.unified (mapping, optimizer, nbit, "
            "network, ...)."
        )


def initialize(cfg: DictConfig, state: State) -> None:
    """Initialize the iceflow module."""

    check_iceflow_method(cfg)

    # Make sure this function is only called once
    if getattr(state, "iceflow_initialized", False):
        return
    state.iceflow_initialized = True

    # Create ice flow object
    state.iceflow = Iceflow()

    # Initialize discretization
    cfg_numerics = cfg.processes.iceflow.numerics

    basis_v = cfg_numerics.basis_vertical.lower()
    discr_v = VerticalDiscrs[basis_v](cfg)
    state.iceflow.discr_v = discr_v

    basis_h = cfg_numerics.basis_horizontal.lower()
    discr_h = HorizontalDiscrs[basis_h](cfg)
    state.iceflow.discr_h = discr_h

    state.vert_weight = define_vertical_weight(
        cfg_numerics.Nz, cfg_numerics.vert_spacing
    )

    if cfg.get("assimilations") and "pretraining" in cfg.assimilations:
        print("Iceflow pretraining mode activated. Skipping iceflow initialization.")
        return

    # Initialize ice-flow fields: U, V, tau_ref, arrhenius
    initialize_iceflow_fields(cfg, state)

    cfg_unified = cfg.processes.iceflow.unified

    # Parameters that may be schedules over model time
    check_schedule(cfg_unified.nbit, "nbit", is_count, "non-negative integers")
    if cfg_unified.optimizer == "sequential":
        for cfg_stage in cfg_unified.sequential.stages:
            if "nbit" in cfg_stage:
                check_schedule(
                    cfg_stage.nbit, "sequential nbit", is_count, "non-negative integers"
                )
    check_schedule(
        cfg_unified.retrain_freq, "retrain_freq", is_count, "non-negative integers"
    )
    check_schedule(cfg_unified.adam.lr, "adam.lr", is_positive, "positive numbers")

    # Parsed once, as it is read at every time step
    state.iceflow.retrain_freq = parse_schedule(cfg_unified.retrain_freq)

    # Drift detection needs running input statistics; otherwise it is never computed.
    cfg_at = getattr(cfg_unified, "adaptive_time", None)
    at_method = (
        str(getattr(cfg_at, "method", "none")).lower() if cfg_at is not None else "none"
    )
    if at_method == "shift_distribution" and not should_normalize(cfg):
        raise ValueError(
            "adaptive_time.method=shift_distribution needs running input statistics "
            "(normalization.method=adaptive with an untrained network mapping); with the "
            "current settings the distribution shift is never computed."
        )

    # Initialize mapping
    mapping_name = cfg_unified.mapping
    mapping_args = InterfaceMappings[mapping_name].get_mapping_args(cfg, state)
    mapping = Mappings[mapping_name](**mapping_args)
    state.iceflow.mapping = mapping

    # Initialize optimizer
    cost_fn = get_cost_fn(cfg, state)
    optimizer_name = cfg_unified.optimizer
    optimizer_args = InterfaceOptimizers[optimizer_name].get_optimizer_args(
        cfg=cfg,
        cost_fn=cost_fn,
        map=mapping,
    )
    if "batch_size" in optimizer_args:
        framesizemax = cfg_unified.data_preparation.framesizemax
        ny, nx = int(state.thk.shape[0]), int(state.thk.shape[1])
        sy, sx = patch_grid(ny, nx, framesizemax)
        num_patches = sy * sx
        if num_patches > 1 and mapping_name != "network":
            raise ValueError(
                f"❌ Multiple batching requires mapping=network, "
                f"but mapping={mapping_name!r} was specified "
                f"(domain produces {num_patches} patches with framesizemax={framesizemax})."
            )
        optimizer_args["batch_size"] = min(optimizer_args["batch_size"], num_patches)
    optimizer = Optimizers[optimizer_name](**optimizer_args)
    state.iceflow.optimizer = optimizer

    # Error estimator (optional)
    if InterfaceErrorEstimator.enabled(cfg):
        error_estimator_args = InterfaceErrorEstimator.get_error_estimator_args(
            cfg, state, cost_fn, mapping
        )
        error_estimator = ErrorEstimator(**error_estimator_args)
        state.iceflow.error_estimator = error_estimator
        # Sequential schedules: attach to the stages, not to the wrapper.
        for stage in getattr(optimizer, "optimizers", None) or [optimizer]:
            stage.attach_error_estimator(error_estimator)

    # Evaluator params
    evaluator_params_args = get_evaluator_params_args(cfg)
    evaluator_params = EvaluatorParams(**evaluator_params_args)
    state.iceflow.evaluator_params = evaluator_params

    if mapping_name == "network" and cfg_unified.network.print_summary:
        fieldin_dict = {
            name: tf.convert_to_tensor(getattr(state, name))
            for name in cfg_unified.inputs
        }
        print_model_with_inputs_detailed(
            model=state.iceflow_model,
            input_data=fieldin_dict,
            cfg_inputs=cfg_unified.inputs,
            normalization_method=cfg_unified.normalization.method,
        )

    # Stop the initial training on its error against a direct solve (optional)
    if cfg_unified.init_stop.enabled:
        init_stop = InitStop.from_cfg(cfg, state)

    # Solve once
    solve_iceflow(cfg, state, init=True)

    if cfg_unified.init_stop.enabled:
        init_stop.finish()

    # Evaluate once
    evaluate_iceflow(cfg, state)


def update(cfg: DictConfig, state: State) -> None:
    """Update the iceflow module."""
    if cfg.get("assimilations") and "pretraining" in cfg.assimilations:
        return

    # Logger
    if hasattr(state, "logger"):
        state.logger.info("Update ICEFLOW at iteration : " + str(state.it))

    # Solve ice flow
    solve_iceflow(cfg, state)

    # Evaluate ice flow
    evaluate_iceflow(cfg, state)

    if cfg.processes.iceflow.vertical_velocity.enabled:
        update_vertical_velocity(cfg, state)


def finalize(cfg: DictConfig, state: State) -> None:
    """Finalize the iceflow module."""
    pass
