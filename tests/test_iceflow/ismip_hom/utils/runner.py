#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import os
from typing import Optional
import numpy as np
import xarray as xr
import pytest

from .simulator import run_igm
from .validation import validate_results
from .plots import plot_comparison
from .config import load_test_config, get_tolerance


def run_experiment_test(
    monkeypatch: pytest.MonkeyPatch,
    experiment: str,
    length: Optional[int] = None,
    method: str = "unified",
    mapping: Optional[str] = None,
    optimizer: Optional[str] = None,
) -> None:
    """
    Generic test runner for any ISMIP-HOM experiment.

    Args:
        monkeypatch: pytest fixture
        experiment: Experiment name (exp_a, exp_b, exp_c, exp_e_1, etc.)
        length: Length scale in km (None for experiments without length parameter like exp_e)
        method: Method to use (unified, emulated, solved)
        mapping: Mapping type (for unified method)
        optimizer: Optimizer type (for unified method)
    """
    # Change to test directory
    test_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(test_dir, "..", experiment)
    monkeypatch.chdir(test_dir)

    # Run simulation
    output_dir = run_igm(monkeypatch, method, length, mapping, optimizer)

    # Load simulation results
    path_results = os.path.join(output_dir, "results.nc")
    x_igm, v_igm = _extract_results(path_results, experiment)

    # Load reference data
    x_ref, v_ref = _load_reference(experiment, length)

    # Validate results
    is_valid, error = validate_results(x_ref, v_ref, x_igm, v_igm, experiment)

    # Create plot and save raw data
    title, filename = _get_plot_info(experiment, length, method, mapping, optimizer)
    plot_comparison(x_ref, v_ref, x_igm, v_igm, title, filename)
    _save_data(x_ref, v_ref, x_igm, v_igm, filename)

    # Display error in terminal
    if error is not None:
        tolerance = get_tolerance(experiment)
        error_pct = error * 100
        tol_pct = tolerance * 100
        status = "✅" if is_valid else "❌"
        comparison = "<" if is_valid else ">"
        length_str = f"L={length}km" if length is not None else ""
        print(f"{status} {experiment} {length_str} {mapping}/{optimizer}: error = {error_pct:.1f}% {comparison} {tol_pct:.0f}%")

    # Assert validation if in compare mode
    if error is not None:
        assert is_valid, (
            f"Validation failed: error={error:.4f} exceeds tolerance. "
            f"See {filename} for details."
        )


def _extract_results(path_results: str, experiment: str) -> tuple:
    """Extract velocity results from simulation output."""
    with xr.open_dataset(path_results) as file:
        values = file["velsurf_mag"].values

        if experiment in ["exp_a", "exp_c"]:
            # Extract y=0.25 line
            ny = values.shape[1]
            v_igm = values[0, int(0.25 * ny)]
        elif experiment in ["exp_b", "exp_d", "exp_e_1", "exp_e_2"]:
            # Extract y=0 line (2D flowline experiments)
            v_igm = values[0, 0]
        else:
            raise ValueError(f"Unknown experiment: {experiment}")

        x_igm = np.linspace(0.0, 1.0, len(v_igm))

    return x_igm, v_igm


def _load_reference(experiment: str, length: Optional[int] = None) -> tuple:
    """Load reference data for experiment."""
    # Determine reference filename
    if experiment == "exp_e_1":
        filename = "oga1e000.txt"
    elif experiment == "exp_e_2":
        filename = "oga1e001.txt"
    else:
        exp_letter = experiment.split("_")[1]  # exp_a -> a
        filename = f"oga1{exp_letter}{length:03d}.txt"

    path_file_ref = os.path.join("..", "data", "oga", filename)

    try:
        file_ref = np.loadtxt(path_file_ref)
    except FileNotFoundError:
        pytest.fail(
            f"❌ The file <{path_file_ref}> is not available. "
            + "Please run <igm/tests/get_data.sh> to download it."
        )

    if experiment in ["exp_a", "exp_c"]:
        # Extract y=0.25 line
        idx = file_ref[:, 1] == 0.25
        x_ref = file_ref[idx, 0]
        v_ref = np.hypot(file_ref[idx, 2], file_ref[idx, 3])
    elif experiment in ["exp_b", "exp_d", "exp_e_1", "exp_e_2"]:
        # Single line data (2D flowline experiments)
        x_ref = file_ref[:, 0]
        v_ref = file_ref[:, 1]
    else:
        raise ValueError(f"Unknown experiment: {experiment}")

    return x_ref, v_ref


def _save_data(
    x_ref: np.ndarray,
    v_ref: np.ndarray,
    x_igm: np.ndarray,
    v_igm: np.ndarray,
    plot_filename: str,
) -> None:
    """Save reference and IGM data as CSV files (one file each)."""
    base = plot_filename.replace(".pdf", "")
    np.savetxt(f"{base}_ref.csv", np.column_stack([x_ref, v_ref]),
               delimiter=",", header="x,v", comments="")
    np.savetxt(f"{base}_igm.csv", np.column_stack([x_igm, v_igm]),
               delimiter=",", header="x,v", comments="")


def _get_plot_info(
    experiment: str,
    length: Optional[int],
    method: str,
    mapping: Optional[str],
    optimizer: Optional[str],
) -> tuple:
    """Generate plot title and filename."""
    exp_name = experiment.replace("_", "-").upper()

    if method == "unified":
        method_str = f"{method}/{mapping}/{optimizer}"
        if length is not None:
            filename_str = f"{method}_{length}km_{mapping}_{optimizer}"
        else:
            filename_str = f"{method}_{mapping}_{optimizer}"
    else:
        method_str = method
        if length is not None:
            filename_str = f"{method}_{length}km"
        else:
            filename_str = method

    if length is not None:
        title = f"ISMIP-HOM | {exp_name} | L={length}km | {method_str}"
    else:
        title = f"ISMIP-HOM | {exp_name} | {method_str}"

    filename = f"{experiment}_{filename_str}.pdf"

    return title, filename
