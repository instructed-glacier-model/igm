from .central import CentralDiscr
from .horizontal import HorizontalDiscr
from .mac import MACDiscr
from .p1 import P1Discr
from .q1 import Q1Discr

HorizontalDiscrs = {
    "central": CentralDiscr,
    "q1": Q1Discr,
    "p1": P1Discr,
    "mac": MACDiscr,
}
