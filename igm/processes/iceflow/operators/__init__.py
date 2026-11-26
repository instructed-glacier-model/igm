"""Hessian operators used by the unified ice-flow optimizers."""

from .energy_operator import (
    ADOperator,
    BandedADOperator,
    MOLHOBandedADOperator,
    Operator,
    SSABandedADOperator,
)
from .molho_banded import supports_compact_molho
from .ssa_banded import supports_compact_ssa
from .tridiag1d import Tridiag1DADOperator, supports_tridiag1d
from .tridiag1d_analytic import Tridiag1DAnalyticOperator
from .gauss_newton import GaussNewtonOperator
from .factory import build_energy_operator
