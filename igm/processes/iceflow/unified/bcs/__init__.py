from .bc import BoundaryCondition
from .frozen_bed import FrozenBed
from .periodic_ns import PeriodicNS, PeriodicNSGlobal
from .periodic_we import PeriodicWE, PeriodicWEGlobal
from .zero_left import ZeroLeft

BoundaryConditions = {
    "frozen_bed": FrozenBed,
    "periodic_ns": PeriodicNS,
    "periodic_we": PeriodicWE,
    "periodic_ns_global": PeriodicNSGlobal,
    "periodic_we_global": PeriodicWEGlobal,
    "zero_left": ZeroLeft,
}

from .interfaces import InterfaceBoundaryCondition, InterfaceBoundaryConditions
