from .optimizer import Optimizer
from .optimizer_adam import OptimizerAdam
from .interface import Interface, Status
from .interface_adam import InterfaceAdam

Optimizers = {
    "adam": OptimizerAdam,
}

Interfaces = {
    "adam": InterfaceAdam,
}
