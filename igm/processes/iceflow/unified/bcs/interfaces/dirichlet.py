from omegaconf import DictConfig
from typing import Any, Dict

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
        basis_vertical = cfg.processes.iceflow.numerics.basis_vertical.lower()
        allowed_bases = ["lagrange", "molho", "ssa"]

        cfg_dirichlet = cfg.processes.iceflow.unified.bc.dirichlet
        values = {
            "left": cfg_dirichlet.get("left", None),
            "right": cfg_dirichlet.get("right", None),
            "top": cfg_dirichlet.get("top", None),
            "bottom": cfg_dirichlet.get("bottom", None),
        }

        if basis_vertical not in allowed_bases:
            nonzero = [k for k, v in values.items() if v is not None and v != 0.0]
            if nonzero:
                raise ValueError(
                    f"Dirichlet boundary condition with non-zero values ({', '.join(nonzero)}) "
                    f"is incompatible with basis_vertical='{basis_vertical}'. "
                    f"The boundary values set the degrees of freedom of the velocity representation, "
                    f"which do not directly correspond to the velocity field for non-nodal bases. "
                    f"Supported vertical bases for non-zero Dirichlet conditions are: {', '.join(allowed_bases)}."
                )

        return values
