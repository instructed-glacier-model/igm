from omegaconf import DictConfig
from typing import Any, Dict, Optional
from .interface import InterfaceBoundaryCondition
from igm.common import State

class InterfaceDirichletBoundary(InterfaceBoundaryCondition):
    """Interface for Dirichlet boundary condition on specified edges."""

    @staticmethod
    def get_bc_args(cfg: DictConfig, state: State) -> Dict[str, Any]:
        """Extract boundary values from config.
        
        Expected config fields (all optional, omit or set to null to skip):
            bc.left   : float
            bc.right  : float
            bc.top    : float
            bc.bottom : float
        """
        return {
            "left":   cfg.processes.iceflow.unified.bc.dirichlet.get("left",   None),
            "right":  cfg.processes.iceflow.unified.bc.dirichlet.get("right",  None),
            "top":    cfg.processes.iceflow.unified.bc.dirichlet.get("top",    None),
            "bottom": cfg.processes.iceflow.unified.bc.dirichlet.get("bottom", None),
        }