#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from omegaconf import DictConfig
from typing import List

from . import (
    BoundaryCondition,
    BoundaryConditions,
    InterfaceBoundaryConditions,
)
from igm.common import State


def init_bcs(
    cfg: DictConfig, state: State, bc_names: List[str]
) -> List[BoundaryCondition]:
    """Initialize and instantiate boundary conditions from configuration and state."""
    bcs = []

    for bc_name in bc_names:
        bc_args = InterfaceBoundaryConditions[bc_name].get_bc_args(cfg, state)
        bc = BoundaryConditions[bc_name](**bc_args)
        bcs.append(bc)

    return bcs
