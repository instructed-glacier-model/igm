from .optimizer import Optimizer
from .adam import OptimizerAdam
from .adam_DA import OptimizerAdamDataAssimilation
from .cg import OptimizerCG
from .cg_newton import OptimizerCGNewton
from .gauss_newton import OptimizerGaussNewton
from .lbfgs import OptimizerLBFGS
from .lbfgs_bounds import OptimizerLBFGSBounds
from .lbfgs_DA import OptimizerLBFGSBoundsDA
from .muon import OptimizerMuon
from .newton import OptimizerNewton
from .sequential import OptimizerSequential
from .soap import OptimizerSOAP
from .ss_esoap import OptimizerSSESOAP
from .trust_region import OptimizerTrustRegion
from .spectral_projected_gradient import OptimizerSpectralProjectedGradient
from .tridiag_newton import OptimizerTridiagNewton

Optimizers = {
    "adam": OptimizerAdam,
    "adam_da": OptimizerAdamDataAssimilation,
    "cg": OptimizerCG,
    "cg_newton": OptimizerCGNewton,
    "gauss_newton": OptimizerGaussNewton,
    "lbfgs": OptimizerLBFGS,
    "lbfgs_bounds": OptimizerLBFGSBounds,
    "lbfgs_da": OptimizerLBFGSBoundsDA,
    "muon": OptimizerMuon,
    "newton": OptimizerNewton,
    "sequential": OptimizerSequential,
    "soap": OptimizerSOAP,
    "ss_esoap": OptimizerSSESOAP,
    "trust_region": OptimizerTrustRegion,
    "spg": OptimizerSpectralProjectedGradient,
    "tridiag_newton": OptimizerTridiagNewton,
}

from .interfaces import InterfaceOptimizer, InterfaceOptimizers, Status
from .utils import SyntheticCosts
