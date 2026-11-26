from .interface import InterfaceOptimizer, Status
from .adam import InterfaceAdam
from .cg import InterfaceCG
from .cg_newton import InterfaceCGNewton
from .gauss_newton import InterfaceGaussNewton
from .lbfgs import InterfaceLBFGS
from .muon import InterfaceMuon
from .newton import InterfaceNewton
from .sequential import InterfaceSequential
from .soap import InterfaceSOAP
from .ss_esoap import InterfaceSSESOAP
from .trust_region import InterfaceTrustRegion
from .spg import InterfaceSPG
from .tridiag_newton import InterfaceTridiagNewton

InterfaceOptimizers = {
    "adam": InterfaceAdam,
    "adam_da": InterfaceAdam,
    "cg": InterfaceCG,
    "cg_newton": InterfaceCGNewton,
    "gauss_newton": InterfaceGaussNewton,
    "lbfgs": InterfaceLBFGS,
    "lbfgs_bounds": InterfaceLBFGS,
    "lbfgs_da": InterfaceLBFGS,
    "muon": InterfaceMuon,
    "newton": InterfaceNewton,
    "sequential": InterfaceSequential,
    "soap": InterfaceSOAP,
    "ss_esoap": InterfaceSSESOAP,
    "trust_region": InterfaceTrustRegion,
    "spg": InterfaceSPG,
    "tridiag_newton": InterfaceTridiagNewton,
}
