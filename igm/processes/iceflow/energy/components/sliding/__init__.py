from .laws import Weertman, WeertmanParams
from .sliding import get_sliding_params_args

SlidingComponents = {
    "weertman": Weertman,
}

SlidingParams = {
    "weertman": WeertmanParams,
}
