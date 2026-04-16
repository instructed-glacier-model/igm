from .interface import InterfaceBoundaryCondition
from .frozen_bed import InterfaceFrozenBed
from .periodic_ns import InterfacePeriodicNS, InterfacePeriodicNSGlobal
from .periodic_we import InterfacePeriodicWE, InterfacePeriodicWEGlobal
from .dirichlet import InterfaceDirichletBoundary
from .zero_left import InterfaceZeroLeft

InterfaceBoundaryConditions = {
    "frozen_bed": InterfaceFrozenBed,
    "periodic_ns": InterfacePeriodicNS,
    "periodic_we": InterfacePeriodicWE,
    "periodic_ns_global": InterfacePeriodicNSGlobal,
    "periodic_we_global": InterfacePeriodicWEGlobal,
    "dirichlet": InterfaceDirichletBoundary,
    "zero_left": InterfaceZeroLeft,
}
