"""Preconditioners used by the unified Newton-CG optimizer."""

from .barotropic_multigrid import (
    BarotropicMultigrid,
    GridTransfer,
    barotropic_mode,
)
from .preconditioner import (
    BarotropicMultigridPreconditioner,
    ComponentBlockJacobiPreconditioner,
    Preconditioner,
    SSABlockJacobiPreconditioner,
    build_preconditioner,
    invert_spd_4x4,
)
