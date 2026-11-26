from omegaconf import DictConfig
from typing import Any, Dict

from .interface import InterfaceBoundaryCondition
from igm.common import State


def _plain(value: Any) -> Any:
    """A config edge value as a float, a list of floats, or None."""
    if value is None or isinstance(value, (int, float)):
        return value
    return [float(v) for v in value]


class InterfaceDirichletBoundary(InterfaceBoundaryCondition):
    """Interface for Dirichlet boundary condition on specified edges."""

    @staticmethod
    def get_bc_args(cfg: DictConfig, state: State) -> Dict[str, Any]:
        """Extract boundary values from config.

        Expected config fields (all optional, omit or set to null to skip):
            bc.left   : float, or [u, v]
            bc.right  : float, or [u, v]
            bc.top    : float, or [u, v]
            bc.bottom : float, or [u, v]
        """
        basis_vertical = cfg.processes.iceflow.numerics.basis_vertical.lower()
        allowed_bases = ["lagrange", "molho", "ssa"]

        cfg_dirichlet = cfg.processes.iceflow.unified.bc.dirichlet
        values = {
            side: _plain(cfg_dirichlet.get(side, None))
            for side in ("left", "right", "top", "bottom")
        }

        if basis_vertical not in allowed_bases:
            nonzero = [
                k
                for k, v in values.items()
                if v is not None
                and any(c != 0.0 for c in (v if isinstance(v, list) else [v]))
            ]
            if nonzero:
                raise ValueError(
                    f"Dirichlet boundary condition with non-zero values ({', '.join(nonzero)}) "
                    f"is incompatible with basis_vertical='{basis_vertical}'. "
                    f"The boundary values set the degrees of freedom of the velocity representation, "
                    f"which do not directly correspond to the velocity field for non-nodal bases. "
                    f"Supported vertical bases for non-zero Dirichlet conditions are: {', '.join(allowed_bases)}."
                )

        return values
