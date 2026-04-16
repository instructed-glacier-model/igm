from .optimizer import Optimizer
from .adam import OptimizerAdam
from .adam_DA import OptimizerAdamDataAssimilation
from .lbfgs import OptimizerLBFGS
from .lbfgs_bounds import OptimizerLBFGSBounds
from .lbfgs_DA import OptimizerLBFGSDataAssimilation
from .cg import OptimizerCG
from .sequential import OptimizerSequential
from .composite import OptimizerComposite
from .muon import OptimizerMuon
from .soap import OptimizerSOAP

Optimizers = {
    "adam": OptimizerAdam,
    "adam_da": OptimizerAdamDataAssimilation,
    "lbfgs": OptimizerLBFGS,
    "lbfgs_bounds": OptimizerLBFGSBounds,
    "lbfgs_da": OptimizerLBFGSDataAssimilation,
    "cg": OptimizerCG,
    "sequential": OptimizerSequential,
    "composite": OptimizerComposite,
    "muon": OptimizerMuon,
    "soap": OptimizerSOAP,
}

from .interfaces import InterfaceOptimizer, InterfaceOptimizers, Status, get_save_args
