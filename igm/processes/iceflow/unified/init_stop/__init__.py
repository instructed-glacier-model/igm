"""Stop the initial training of the network on its error against a direct solve."""

from .display import InitStopDisplay
from .init_stop import InitStop, InitStopCriterion, StopReason
from .reference import ReferenceSolve, get_reference_cfg, solve_reference
