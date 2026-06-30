from .laws import (
    Budd,
    BuddParams,
    ReguCoulomb,
    ReguCoulombParams,
    MohrCoulomb,
    MohrCoulombParams,
    Weertman,
    WeertmanParams,
)
from .sliding import get_sliding_params_args

SlidingComponents = {
    "budd": Budd,
    "regu_coulomb": ReguCoulomb,
    "mohr_coulomb": MohrCoulomb,
    "weertman": Weertman,
}

SlidingParams = {
    "budd": BuddParams,
    "regu_coulomb": ReguCoulombParams,
    "mohr_coulomb": MohrCoulombParams,
    "weertman": WeertmanParams,
}
