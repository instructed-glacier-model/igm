#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from typing import Any, Dict

from omegaconf import DictConfig

from igm.common import State

from .interface import InterfaceBoundaryCondition


class InterfacePeriodicNS(InterfaceBoundaryCondition):
    """Interface for periodic boundary condition in north-south direction."""

    @staticmethod
    def get_bc_args(cfg: DictConfig, state: State) -> Dict[str, Any]:
        """Return empty arguments for periodic north-south boundary condition."""
        return {}


class InterfacePeriodicNSGlobal(InterfaceBoundaryCondition):
    """Interface for global periodic boundary condition in north-south direction."""

    @staticmethod
    def get_bc_args(cfg: DictConfig, state: State) -> Dict[str, Any]:
        """Extract grid dimensions from configuration and state for global periodic north-south condition."""
        Nx = state.thk.shape[1]
        Ny = state.thk.shape[0]
        Nz = cfg.processes.iceflow.numerics.Nz

        return {"Nx": Nx, "Ny": Ny, "Nz": Nz}
