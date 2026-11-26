#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from typing import Any, Dict

from omegaconf import DictConfig

from igm.common.core import State

from .interface import InterfaceBoundaryCondition


class InterfaceNoInflow(InterfaceBoundaryCondition):
    """Interface for no-inflow boundary condition."""

    @staticmethod
    def get_bc_args(cfg: DictConfig, state: State) -> Dict[str, Any]:
        """Return empty arguments for no-inflow boundary condition."""
        basis_vertical = cfg.processes.iceflow.numerics.basis_vertical.lower()
        allowed_bases = ["lagrange", "molho", "ssa"]

        if basis_vertical not in allowed_bases:
            raise ValueError(
                f"No-inflow boundary condition is incompatible with basis_vertical='{basis_vertical}'. "
                f"Supported vertical bases are: {', '.join(allowed_bases)}. "
                f"This boundary condition may cause numerical issues with other discretizations."
            )

        return {}
