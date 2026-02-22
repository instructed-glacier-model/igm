#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from omegaconf import DictConfig
from typing import Any, Dict

from .interface import InterfaceBoundaryCondition
from igm.common import State


class InterfaceZeroLeft(InterfaceBoundaryCondition):

    @staticmethod
    def get_bc_args(cfg: DictConfig, state: State) -> Dict[str, Any]:
        return {}
