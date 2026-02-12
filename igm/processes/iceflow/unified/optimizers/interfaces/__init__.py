from .interface import InterfaceOptimizer, Status
from .adam import InterfaceAdam
from .lbfgs import InterfaceLBFGS
from .cg import InterfaceCG
from .sequential import InterfaceSequential
from .composite import InterfaceComposite

InterfaceOptimizers = {
    "adam": InterfaceAdam,
    "adam_da": InterfaceAdam,
    "lbfgs": InterfaceLBFGS,
    "lbfgs_bounds": InterfaceLBFGS,
    "lbfgs_da": InterfaceLBFGS,
    "cg": InterfaceCG,
    "sequential": InterfaceSequential,
    "composite": InterfaceComposite,
}
