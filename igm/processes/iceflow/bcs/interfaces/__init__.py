from .dirichlet import InterfaceDirichletBoundary
from .frozen_bed import InterfaceFrozenBed
from .interface import InterfaceBoundaryCondition
from .no_inflow import InterfaceNoInflow
from .periodic_ns import InterfacePeriodicNS, InterfacePeriodicNSGlobal
from .periodic_we import InterfacePeriodicWE, InterfacePeriodicWEGlobal

InterfaceBoundaryConditions = {
    "frozen_bed": InterfaceFrozenBed,
    "periodic_ns": InterfacePeriodicNS,
    "periodic_we": InterfacePeriodicWE,
    "periodic_ns_global": InterfacePeriodicNSGlobal,
    "periodic_we_global": InterfacePeriodicWEGlobal,
    "no_inflow": InterfaceNoInflow,
    "dirichlet": InterfaceDirichletBoundary,
}
