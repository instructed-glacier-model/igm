from .optimizer import Optimizer
from .optimizer_adam import OptimizerAdam
from .interface import InterfaceOptimizer, Status
from .interface_adam import InterfaceAdam

Optimizers = {
    "adam": OptimizerAdam,
}

InterfaceOptimizers = {
    "adam": InterfaceAdam,
}
