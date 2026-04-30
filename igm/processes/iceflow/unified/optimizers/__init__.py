from .optimizer import Optimizer
from .adam import OptimizerAdam
from .adam_DA import OptimizerAdamDataAssimilation
from .cg import OptimizerCG
from .cg_newton import OptimizerCGNewton
from .lbfgs import OptimizerLBFGS
from .lbfgs_bounds import OptimizerLBFGSBounds
from .lbfgs_DA import OptimizerLBFGSBoundsDA
from .muon import OptimizerMuon
from .newton import OptimizerNewton
from .sequential import OptimizerSequential
from .soap import OptimizerSOAP
from .trust_region import OptimizerTrustRegion

Optimizers = {
    "adam": OptimizerAdam,
    "adam_da": OptimizerAdamDataAssimilation,
    "cg": OptimizerCG,
    "cg_newton": OptimizerCGNewton,
    "lbfgs": OptimizerLBFGS,
    "lbfgs_bounds": OptimizerLBFGSBounds,
    "lbfgs_da": OptimizerLBFGSBoundsDA,
    "muon": OptimizerMuon,
    "newton": OptimizerNewton,
    "sequential": OptimizerSequential,
    "soap": OptimizerSOAP,
    "trust_region": OptimizerTrustRegion,
}

from .interfaces import InterfaceOptimizer, InterfaceOptimizers, Status
from .utils import SyntheticCosts
