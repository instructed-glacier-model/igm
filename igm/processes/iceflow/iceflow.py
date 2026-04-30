#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file


from omegaconf import DictConfig

from igm.common import State
from igm.processes.iceflow.emulate.emulator import (
    initialize_iceflow_emulator,
    update_iceflow_emulator,
)
from igm.processes.iceflow.solve.solve import (
    initialize_iceflow_solver,
    update_iceflow_solved,
)
from igm.processes.iceflow.diagnostic.diagnostic import (
    initialize_iceflow_diagnostic,
    update_iceflow_diagnostic,
)
from igm.processes.iceflow.unified.unified import (
    initialize_iceflow_unified,
    update_iceflow_unified,
)
from igm.processes.iceflow.emulate.utils import save_iceflow_model
from igm.processes.iceflow.utils.fields import initialize_iceflow_fields
from igm.processes.iceflow.utils.vertical_discretization import define_vertical_weight
from igm.processes.iceflow.vertical import VerticalDiscrs
from igm.processes.iceflow.horizontal import HorizontalDiscrs


class Iceflow:
    """Container for the iceflow related states."""

    pass


def initialize(cfg: DictConfig, state: State) -> None:
    """Initialize the iceflow module."""

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

    if "pretraining" in cfg.processes.keys():
        print("Iceflow pretraining mode activated. Skipping iceflow initialization.")
        return

    # Initialize ice-flow fields: U, V, slidingco, arrhenius
    initialize_iceflow_fields(cfg, state)

    # Initialize ice-flow method
    iceflow_method = cfg.processes.iceflow.method.lower()

    if iceflow_method == "emulated":
        initialize_iceflow = initialize_iceflow_emulator
    elif iceflow_method == "solved":
        initialize_iceflow = initialize_iceflow_solver
    elif iceflow_method == "diagnostic":
        initialize_iceflow = initialize_iceflow_diagnostic
    elif iceflow_method == "unified":
        initialize_iceflow = initialize_iceflow_unified
    else:
        raise ValueError(f"❌ Unknown ice flow method: <{iceflow_method}>.")

    initialize_iceflow(cfg, state)


def update(cfg: DictConfig, state: State) -> None:
    """Update the iceflow module."""
    if "pretraining" in cfg.processes.keys():
        return

    # Logger
    if hasattr(state, "logger"):
        state.logger.info("Update ICEFLOW at iteration : " + str(state.it))

    # Update ice-flow method
    iceflow_method = cfg.processes.iceflow.method.lower()

    if iceflow_method == "emulated":
        update_iceflow = update_iceflow_emulator
    elif iceflow_method == "solved":
        update_iceflow = update_iceflow_solved
    elif iceflow_method == "diagnostic":
        update_iceflow = update_iceflow_diagnostic
    elif iceflow_method == "unified":
        update_iceflow = update_iceflow_unified
    else:
        raise ValueError(f"❌ Unknown ice flow method: <{iceflow_method}>.")

    update_iceflow(cfg, state)


def finalize(cfg: DictConfig, state: State) -> None:
    """Finalize the iceflow module."""
    if "pretraining" in cfg.processes.keys():
        return

    # Save emulated model
    iceflow_method = cfg.processes.iceflow.method.lower()
    if iceflow_method == "emulated":
        if cfg.processes.iceflow.emulator.save_model:
            save_iceflow_model(cfg, state)
