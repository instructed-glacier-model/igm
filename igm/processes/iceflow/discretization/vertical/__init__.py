from .lagrange import LagrangeDiscr
from .legendre import LegendreDiscr
from .molho import MOLHODiscr
from .ssa import SSADiscr
from .vertical import VerticalDiscr

VerticalDiscrs = {
    "lagrange": LagrangeDiscr,
    "legendre": LegendreDiscr,
    "molho": MOLHODiscr,
    "ssa": SSADiscr,
}
