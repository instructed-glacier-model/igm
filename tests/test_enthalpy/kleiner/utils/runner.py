#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Generic test runner for Kleiner et al. (2015) benchmarks."""

import os
import numpy as np
import pytest

from .experiments import (
    setup_experiment_a,
    setup_experiment_b,
    run_simulation_a,
    run_simulation_b,
    extract_results_b,
)
from .validation import validate_experiment_a, validate_experiment_b
from .plots import plot_exp_a, plot_exp_b


def _print_header(title: str) -> None:
    """Print formatted section header."""
    print(f"\n{'='*70}")
    print(title)
    print(f"{'='*70}")


def _try_plot(plot_fn, results) -> None:
    """Attempt to generate plot, warn on failure."""
    try:
        plot_fn(results)
    except Exception as e:
        print(f"Warning: plotting failed: {e}")


def run_experiment_test(
    monkeypatch: pytest.MonkeyPatch,
    experiment: str,
    dt: float = 200.0,
    Nz_E: int = 50,
) -> None:
    """
    Generic test runner for Kleiner enthalpy benchmarks.

    Args:
        monkeypatch: pytest fixture
        experiment: Experiment name (exp_a, exp_b)
        dt: Time step in years (for exp_a)
        Nz_E: Number of vertical levels
    """
    test_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", experiment
    )
    monkeypatch.chdir(test_dir)

    _print_header(f"Running Kleiner Experiment {experiment[-1].upper()}")

    if experiment == "exp_a":
        _run_exp_a(dt, Nz_E)
    elif experiment == "exp_b":
        _run_exp_b(Nz_E)
    else:
        raise ValueError(f"Unknown experiment: {experiment}")


def _save_results_a(results: dict, output_dir: str = ".") -> None:
    """Save numerical time series for Experiment A to a CSV file."""
    num_path = os.path.join(output_dir, "numerical_results_exp_a.csv")
    np.savetxt(
        num_path,
        np.column_stack(
            [
                results["time"],
                results["T_base"],
                results["melt_rate"],
                results["till_water"],
            ]
        ),
        delimiter=",",
        header="time_yr,T_base_C,melt_rate_m_per_yr,till_water_m",
        comments="",
    )
    print(f"  Numerical results saved to: {num_path}")


def _run_exp_a(dt: float, Nz_E: int) -> None:
    """Run and validate Experiment A."""
    cfg, state = setup_experiment_a(dt=dt, Nz_E=Nz_E)
    results = run_simulation_a(cfg, state, dt)

    _try_plot(plot_exp_a, results)

    is_valid, errors = validate_experiment_a(results)

    _save_results_a(results)

    _print_header("VALIDATION SUMMARY")
    for phase, data in errors.items():
        print(f"  {phase}: {'PASS' if data['valid'] else 'FAIL'}")
    print("=" * 70)

    assert is_valid, f"Validation failed: {errors}"


def _save_results_b(results: dict, validation: dict, output_dir: str = ".") -> None:
    """Save numerical and analytical profiles for Experiment B to CSV files."""
    z_num = results["z"]
    E_num = results["E"]
    T_num = results["T"]
    omega_num = results["omega"]

    E_ana = validation["enthalpy"]["analytical"]
    T_ana = validation["temperature"]["analytical"]
    omega_ana = validation["water_content"]["analytical"]

    num_path = os.path.join(output_dir, "numerical_results_exp_b.csv")
    np.savetxt(
        num_path,
        np.column_stack([z_num, E_num, T_num, omega_num]),
        delimiter=",",
        header="z_m,E_J_per_kg,T_C,omega_pct",
        comments="",
    )
    print(f"  Numerical results saved to: {num_path}")

    ana_path = os.path.join(output_dir, "analytical_results_exp_b.csv")
    np.savetxt(
        ana_path,
        np.column_stack([z_num, E_ana, T_ana, omega_ana]),
        delimiter=",",
        header="z_m,E_J_per_kg,T_C,omega_pct",
        comments="",
    )
    print(f"  Analytical results saved to: {ana_path}")


def _run_exp_b(Nz_E: int) -> None:
    """Run and validate Experiment B."""
    cfg, state = setup_experiment_b(Nz_E=Nz_E)
    run_simulation_b(cfg, state)
    results = extract_results_b(state)

    _try_plot(plot_exp_b, results)

    is_valid, errors = validate_experiment_b(results)

    _save_results_b(results, errors)

    _print_header("VALIDATION SUMMARY")
    for key, data in errors.items():
        if key in ("overall_valid", "full_analytical"):
            continue
        if isinstance(data, dict) and "valid" in data:
            metric = data.get("rmsd", data.get("error", ""))
            print(f"  {key}: {'PASS' if data['valid'] else 'FAIL'} ({metric:.3f})")
    print("=" * 70)

    assert is_valid, f"Validation failed: {errors}"
