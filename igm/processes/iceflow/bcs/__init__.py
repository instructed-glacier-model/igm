from .bc import BoundaryCondition
from .dirichlet import DirichletBoundary
from .frozen_bed import FrozenBed
from .no_inflow import NoInflow
from .periodic_ns import PeriodicNS, PeriodicNSGlobal
from .periodic_we import PeriodicWE, PeriodicWEGlobal

BoundaryConditions = {
    "frozen_bed": FrozenBed,
    "periodic_ns": PeriodicNS,
    "periodic_we": PeriodicWE,
    "periodic_ns_global": PeriodicNSGlobal,
    "periodic_we_global": PeriodicWEGlobal,
    "no_inflow": NoInflow,
    "dirichlet": DirichletBoundary,
}

from .interfaces import InterfaceBoundaryCondition, InterfaceBoundaryConditions
