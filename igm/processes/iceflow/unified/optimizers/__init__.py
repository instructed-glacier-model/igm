from .optimizer import Optimizer
from .adam import OptimizerAdam
from .adam_DA import OptimizerAdamDataAssimilation
from .lbfgs import OptimizerLBFGS
from .lbfgs_bounds import OptimizerLBFGSBounds
from .lbfgs_DA import OptimizerLBFGSDataAssimilation
from .cg import OptimizerCG
from .sequential import OptimizerSequential
from .composite import OptimizerComposite

Optimizers = {
    "adam": OptimizerAdam,
    "adam_da": OptimizerAdamDataAssimilation,
    "lbfgs": OptimizerLBFGS,
    "lbfgs_bounds": OptimizerLBFGSBounds,
    "lbfgs_da": OptimizerLBFGSDataAssimilation,
    "cg": OptimizerCG,
    "sequential": OptimizerSequential,
    "composite": OptimizerComposite,
}

from .interfaces import InterfaceOptimizer, InterfaceOptimizers, Status
