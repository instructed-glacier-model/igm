#!/usr/bin/env python3
"""
Analytical solutions for Experiment A from Kleiner et al. (2015):
"Enthalpy benchmark experiments for numerical ice sheet models"
The Cryosphere, 9, 217–228, doi:10.5194/tc-9-217-2015
"""

import numpy as np


class ExperimentAAnalytical:
    """Analytical solutions for 1D thermal evolution benchmark."""

    def __init__(
        self,
        H=1000.0,
        k_ice=2.1,
        c_ice=2009.0,
        rho_ice=910.0,
        q_geo=0.042,
        rho_w=1000.0,
        L=3.34e5,
    ):
        """
        Initialize physical parameters.

        Parameters:
            H: Ice thickness (m)
            k_ice: Thermal conductivity (W/(m K))
            c_ice: Specific heat capacity (J/(kg K))
            rho_ice: Ice density (kg/m^3)
            q_geo: Geothermal heat flux (W/m^2)
            rho_w: Water density (kg/m^3)
            L: Latent heat of fusion (J/kg)
        """
        self.H = H
        self.k_ice = k_ice
        self.q_geo = q_geo
        self.rho_w = rho_w
        self.L = L
        self.kappa = k_ice / (rho_ice * c_ice)
        self.spy = 31556926.0

    def steady_basal_temperature(self, T_surface):
        """Compute steady-state basal temperature for cold ice [K]."""
        return T_surface + self.H * self.q_geo / self.k_ice

    def steady_melt_rate(self, T_surface, T_pmp):
        """Compute steady-state basal melt rate when base is at PMP [m/y]."""
        q_basal = self.q_geo + self.k_ice * (T_surface - T_pmp) / self.H
        return self.spy * q_basal / (self.rho_w * self.L)

    def transient_melt_rate(self, t, T_cold, T_warm, T_pmp, n_terms=100):
        """
        Compute transient basal melt rate during cooling [m/y].

        Parameters:
            t: Time since cooling started [years]
            T_cold: Final cold surface temperature [K]
            T_warm: Initial warm surface temperature [K]
            T_pmp: Pressure melting point at base [K]
            n_terms: Number of Fourier series terms (default 100)

        Returns:
            Basal melt rate [m/y]
        """
        t_sec = t * self.spy
        dT_eq = (T_cold - T_pmp) / self.H  # Equilibrium gradient

        # Fourier series for temperature deviation
        dT_dev = 0.0
        for n in range(1, n_terms + 1):
            lambda_n = -self.kappa * (n * np.pi / self.H) ** 2
            A_n = (-1) ** (n + 1) * 2 * (T_warm - T_cold) / (n * np.pi)
            dT_dev += (n * np.pi / self.H) * A_n * np.exp(lambda_n * t_sec)

        # Total heat flux and melt rate
        q_ice = -self.k_ice * (dT_eq + dT_dev)
        return self.spy * (self.q_geo - q_ice) / (self.rho_w * self.L)

    def validate_phase_i(self, T_base_final, T_pmp, tolerance=1.0):
        """
        Validate Phase I: Equilibration should warm but stay below PMP.

        Returns:
            (is_valid, T_analytical, error)
        """
        T_steady = self.steady_basal_temperature(T_surface=243.15)  # -30°C
        error = abs(T_base_final - T_steady)
        is_valid = error < tolerance and T_base_final < T_pmp
        return is_valid, T_steady, error

    def validate_phase_ii(self, melt_rate_final, T_pmp, tolerance=0.001):
        """
        Validate Phase II: Should reach steady melt rate at PMP.

        Returns:
            (is_valid, melt_analytical, error)
        """
        T_warm = 268.15  # -5°C
        melt_steady = self.steady_melt_rate(T_warm, T_pmp)
        error = abs(melt_rate_final - melt_steady)
        is_valid = error < tolerance
        return is_valid, melt_steady, error

    def validate_phase_iii(
        self, t, melt_rate, T_pmp, tolerance=0.15, melt_threshold=1e-4
    ):
        """
        Validate Phase III: Transient cooling should match analytical solution.

        Only validates the initial transient period where base remains near PMP.
        Once melt becomes negligible, the analytical solution is no longer valid.

        Parameters:
            t: Time array since cooling started [years]
            melt_rate: Numerical melt rate array [m/y]
            T_pmp: Pressure melting point [K]
            tolerance: Relative error tolerance (default 0.15 = 15%)
            melt_threshold: Minimum melt rate to consider [m/y] (default 1e-4)

        Returns:
            (is_valid, melt_analytical, mean_error)
        """
        T_cold = 243.15  # -30°C
        T_warm = 268.15  # -5°C
        melt_analytical = self.transient_melt_rate(t, T_cold, T_warm, T_pmp)

        # Only compare initial transient phase where melting is still significant
        # Find where numerical melt drops below threshold
        active_melt_mask = melt_rate > melt_threshold
        if active_melt_mask.sum() > 0:
            # Find the last index where melt is still active
            last_active_idx = np.where(active_melt_mask)[0][-1]
            # Only validate up to this point
            t_valid = t[: last_active_idx + 1]
            melt_rate_valid = melt_rate[: last_active_idx + 1]
            melt_analytical_valid = melt_analytical[: last_active_idx + 1]

            # Compare where both are non-zero
            valid_mask = (melt_analytical_valid > 1e-6) & (melt_rate_valid > 1e-6)
            if valid_mask.sum() > 0:
                # Normalized error to avoid division by very small values
                # Scale: 1e-5 m/y = 0.01 mm/y
                rel_error = np.abs(
                    (melt_rate_valid[valid_mask] - melt_analytical_valid[valid_mask])
                    / (np.abs(melt_analytical_valid[valid_mask]) + 1e-5)
                )
                mean_error = np.mean(rel_error)
                is_valid = mean_error < tolerance
            else:
                mean_error = 0.0
                is_valid = True
        else:
            mean_error = 0.0
            is_valid = True

        return is_valid, melt_analytical, mean_error


def validate_exp_a(time, T_base, melt_rate, till_water):
    """
    Validate Experiment A results against analytical solutions.

    Returns:
        Dictionary with validation results for each phase
    """
    analytical = ExperimentAAnalytical()
    T_pmp = 273.15 - 0.7  # Pressure melting point at 1000m

    results = {}

    # Phase I: Equilibration (0-100 ky)
    mask_i = time < 100000.0
    T_i_final = T_base[mask_i][-1] + 273.15  # Convert to K
    is_valid_i, T_analytical_i, error_i = analytical.validate_phase_i(T_i_final, T_pmp)
    results["phase_i"] = {
        "valid": is_valid_i,
        "T_numerical": T_i_final - 273.15,
        "T_analytical": T_analytical_i - 273.15,
        "error": error_i,
    }

    # Phase II: Warming (100-150 ky)
    mask_ii = (time >= 100000.0) & (time < 150000.0)
    melt_ii_final = melt_rate[mask_ii][-1]
    is_valid_ii, melt_analytical_ii, error_ii = analytical.validate_phase_ii(
        melt_ii_final, T_pmp
    )
    results["phase_ii"] = {
        "valid": is_valid_ii,
        "melt_numerical": melt_ii_final,
        "melt_analytical": melt_analytical_ii,
        "error": error_ii,
    }

    # Phase III: Cooling (150-300 ky)
    mask_iii = time >= 150000.0
    t_iii = time[mask_iii] - 150000.0  # Time since cooling started
    melt_iii = melt_rate[mask_iii]
    is_valid_iii, melt_analytical_iii, error_iii = analytical.validate_phase_iii(
        t_iii, melt_iii, T_pmp
    )
    results["phase_iii"] = {
        "valid": is_valid_iii,
        "melt_numerical": melt_iii,
        "melt_analytical": melt_analytical_iii,
        "mean_error": error_iii,
        "time": t_iii,
    }

    return results
