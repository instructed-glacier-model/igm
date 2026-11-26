#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from omegaconf import DictConfig
from abc import ABC, abstractmethod
from typing import Any, Dict

from igm.common import State


class InterfaceBoundaryCondition(ABC):
    """Abstract base class for boundary condition interfaces."""

    @staticmethod
    @abstractmethod
    def get_bc_args(cfg: DictConfig, state: State) -> Dict[str, Any]:
        """Extract boundary condition arguments from configuration and state."""
        raise NotImplementedError(
            "❌ The get_bc_args static method is not implemented."
        )
