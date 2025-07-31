from .sliding_law import sliding_law_XY

from .laws import (
    Budd,
    BuddParams,
    Coulomb,
    CoulombParams,
    Weertman,
    WeertmanParams
)

SlidingLaws = {
	"budd": Budd,
	"coulomb": Coulomb,
	"weertman": Weertman,
}

SlidingParams = {
	"budd": BuddParams,
	"coulomb": CoulombParams,
	"weertman": WeertmanParams,
}