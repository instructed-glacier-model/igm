"""Cheap a-posteriori velocity-error estimation for the unified solver."""

from .error_estimator import ErrorEstimator
from .interface import InterfaceErrorEstimator
from .metrics import (
    masked_median,
    masked_rmse,
    relative_error_percent,
    summarize_velocity_error,
)
from .reference import load_reference_surface_velocity

__all__ = [
    "ErrorEstimator",
    "InterfaceErrorEstimator",
    "masked_median",
    "masked_rmse",
    "relative_error_percent",
    "summarize_velocity_error",
    "load_reference_surface_velocity",
]
