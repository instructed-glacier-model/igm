#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Physical validation for the Hewitt & Schoof (2017) ice cap test.

Three mandatory checks (required for the test to pass):
  1. No superheating: T ≤ T_pmp = 273.15 K (beta=0 condition from the paper).
  2. Temperature ≥ T_surface - 1 K (physically reasonable lower bound).
  3. Non-negative water content: omega ≥ 0 everywhere.
"""

import numpy as np
from typing import Dict, Any, List, Tuple


def validate_results(
    results: Dict[str, Any], cfg, T_s: float = -10.0
) -> Tuple[bool, List[str]]:
    """
    Validate simulation results against physical expectations.

    Returns:
        (is_valid, errors): True and empty list if all checks pass;
                            False and list of failure messages otherwise.
    """
    T_final = results["T"][-1]  # (Nz, Ny, Nx)
    omega_final = results["omega"][-1]  # (Nz, Ny, Nx)

    T_pmp = 273.15
    T_surface_K = T_s + 273.15

    errors = []

    max_T = float(np.max(T_final))
    if max_T - T_pmp >= 0.1:
        errors.append(f"Superheating: max T = {max_T:.3f} K > T_pmp = {T_pmp} K")

    min_T = float(np.min(T_final))
    if min_T < T_surface_K - 1.0:
        errors.append(
            f"Temperature too cold: min T = {min_T:.3f} K < {T_surface_K - 1.0:.3f} K"
        )

    min_omega = float(np.min(omega_final))
    if min_omega < -1e-6:
        errors.append(f"Negative water content: min omega = {min_omega:.2e}")

    return not errors, errors
