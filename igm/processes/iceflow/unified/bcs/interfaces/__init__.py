from .interface import InterfaceBoundaryCondition
from .frozen_bed import InterfaceFrozenBed
from .periodic_ns import InterfacePeriodicNS, InterfacePeriodicNSGlobal
from .periodic_we import InterfacePeriodicWE, InterfacePeriodicWEGlobal
from .no_inflow import InterfaceNoInflow
from .dirichlet import InterfaceDirichletBoundary

InterfaceBoundaryConditions = {
    "frozen_bed": InterfaceFrozenBed,
    "periodic_ns": InterfacePeriodicNS,
    "periodic_we": InterfacePeriodicWE,
    "periodic_ns_global": InterfacePeriodicNSGlobal,
    "periodic_we_global": InterfacePeriodicWEGlobal,
    "no_inflow": InterfaceNoInflow,
    "dirichlet": InterfaceDirichletBoundary
}
