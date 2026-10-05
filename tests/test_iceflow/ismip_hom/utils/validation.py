#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
from typing import Tuple, Optional
from .config import load_test_config, get_tolerance


def validate_results(
    x_ref: np.ndarray,
    v_ref: np.ndarray,
    x_igm: np.ndarray,
    v_igm: np.ndarray,
    experiment: str,
    mapping: Optional[str] = None,
    optimizer: Optional[str] = None,
) -> Tuple[bool, Optional[float]]:
    """Validate simulation results against reference data."""
    config = load_test_config()
    validation_config = config["validation"]

    if validation_config["mode"] == "run_only":
        return True, None

    # Interpolate IGM results to reference grid
    v_igm_interp = np.interp(x_ref, x_igm, v_igm)

    # Compute error based on type
    error_type = validation_config["tolerance"]["type"]

    if error_type == "relative_l2":
        # Relative L2 error: ||v_igm - v_ref||_2 / ||v_ref||_2
        norm_diff = np.linalg.norm(v_igm_interp - v_ref)
        norm_ref = np.linalg.norm(v_ref)
        error = norm_diff / norm_ref if norm_ref > 1e-10 else norm_diff

    elif error_type == "absolute_l2":
        # Absolute L2 error: ||v_igm - v_ref||_2
        error = np.linalg.norm(v_igm_interp - v_ref)

    elif error_type == "relative_max":
        # Relative max error: max(|v_igm - v_ref| / |v_ref|)
        v_ref_safe = np.where(np.abs(v_ref) < 1e-10, 1e-10, v_ref)
        error = np.max(np.abs(v_igm_interp - v_ref) / np.abs(v_ref_safe))

    elif error_type == "absolute_max":
        # Absolute max error: max(|v_igm - v_ref|)
        error = np.max(np.abs(v_igm_interp - v_ref))

    else:
        raise ValueError(f"Unknown error type: {error_type}")

    tolerance = get_tolerance(experiment, mapping, optimizer)
    is_valid = error <= tolerance

    return is_valid, error
