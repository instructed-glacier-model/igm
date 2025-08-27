from .optimizer import Optimizer
from .optimizer_adam import OptimizerAdam
from .optimizer_lbfgs import OptimizerLBFGS
from .interface import InterfaceOptimizer, Status
from .interface_adam import InterfaceAdam
from .interface_lbfgs import InterfaceLBFGS

Optimizers = {
    "adam": OptimizerAdam,
    "lbfgs": OptimizerLBFGS,
}

InterfaceOptimizers = {
    "adam": InterfaceAdam,
    "lbfgs": InterfaceLBFGS,
}
