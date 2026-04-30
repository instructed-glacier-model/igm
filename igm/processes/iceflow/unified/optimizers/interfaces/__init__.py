from .interface import InterfaceOptimizer, Status
from .adam import InterfaceAdam
from .cg import InterfaceCG
from .cg_newton import InterfaceCGNewton
from .lbfgs import InterfaceLBFGS
from .muon import InterfaceMuon
from .newton import InterfaceNewton
from .sequential import InterfaceSequential
from .soap import InterfaceSOAP
from .trust_region import InterfaceTrustRegion

InterfaceOptimizers = {
    "adam": InterfaceAdam,
    "adam_da": InterfaceAdam,
    "cg": InterfaceCG,
    "cg_newton": InterfaceCGNewton,
    "lbfgs": InterfaceLBFGS,
    "lbfgs_bounds": InterfaceLBFGS,
    "lbfgs_da": InterfaceLBFGS,
    "muon": InterfaceMuon,
    "newton": InterfaceNewton,
    "sequential": InterfaceSequential,
    "soap": InterfaceSOAP,
    "trust_region": InterfaceTrustRegion,
}
