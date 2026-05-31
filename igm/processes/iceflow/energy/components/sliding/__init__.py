from .laws import (
    Budd,
    BuddParams,
    Coulomb,
    CoulombParams,
    MohrCoulomb,
    MohrCoulombParams,
    Weertman,
    WeertmanParams,
)
from .sliding import get_sliding_params_args

SlidingComponents = {
    "budd": Budd,
    "coulomb": Coulomb,
    "mohr_coulomb": MohrCoulomb,
    "weertman": Weertman,
}

SlidingParams = {
    "budd": BuddParams,
    "coulomb": CoulombParams,
    "mohr_coulomb": MohrCoulombParams,
    "weertman": WeertmanParams,
}
