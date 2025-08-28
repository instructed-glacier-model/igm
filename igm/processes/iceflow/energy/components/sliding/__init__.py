from .laws import Budd, BuddParams, Coulomb, CoulombParams, Weertman, WeertmanParams
from .sliding import get_sliding_params_args

SlidingComponents = {
    "budd": Budd,
    "coulomb": Coulomb,
    "weertman": Weertman,
}

SlidingParams = {
    "budd": BuddParams,
    "coulomb": CoulombParams,
    "weertman": WeertmanParams,
}
