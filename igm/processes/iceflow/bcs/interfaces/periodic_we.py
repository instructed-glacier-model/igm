#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from omegaconf import DictConfig
from typing import Any, Dict

from .interface import InterfaceBoundaryCondition
from igm.common import State


class InterfacePeriodicWE(InterfaceBoundaryCondition):
    """Interface for periodic boundary condition in west-east direction."""

    @staticmethod
    def get_bc_args(cfg: DictConfig, state: State) -> Dict[str, Any]:
        """Return empty arguments for periodic west-east boundary condition."""
        return {}


class InterfacePeriodicWEGlobal(InterfaceBoundaryCondition):
    """Interface for global periodic boundary condition in west-east direction."""

    @staticmethod
    def get_bc_args(cfg: DictConfig, state: State) -> Dict[str, Any]:
        """Extract grid dimensions from configuration and state for global periodic west-east condition."""
        Nx = state.thk.shape[1]
        Ny = state.thk.shape[0]
        Nz = cfg.processes.iceflow.numerics.Nz

        return {"Nx": Nx, "Ny": Ny, "Nz": Nz}
