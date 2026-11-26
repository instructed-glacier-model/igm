#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from typing import Any, Dict

from omegaconf import DictConfig

from igm.common import State

from .interface import InterfaceBoundaryCondition


class InterfaceFrozenBed(InterfaceBoundaryCondition):
    """Interface for frozen bed boundary condition."""

    @staticmethod
    def get_bc_args(cfg: DictConfig, state: State) -> Dict[str, Any]:
        """Extract vertical discretization weights from state for frozen bed condition."""
        return {"V_b": state.iceflow.discr_v.V_b}
