#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Central test runner for the Hewitt & Schoof (2017) ice cap benchmark."""

import os
import pytest

from .experiments import setup_experiment, run_simulation
from .plots import plot_results
from .validation import validate_results

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_PLOTS_DIR = os.path.join(_TEST_DIR, "..", "plots")


def run_ice_cap_test(
    monkeypatch: pytest.MonkeyPatch,
    Nx: int = 101,
    Nz_E: int = 100,
    T_s: float = -10.0,
    drain: bool = True,
    dt: float = 1.0,
    time_simu: float = 5_000.0,
    store_every: int = 2000,
) -> None:
    """Run and validate the Hewitt & Schoof (2017) ice cap benchmark."""
    drain_label = "with drainage" if drain else "no drainage"
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  Hewitt & Schoof ice cap — Ts = {T_s} °C, {drain_label}")
    print(f"  Grid: Nx={Nx}, Nz={Nz_E} | dt={dt} yr | {time_simu:.0f} yr total")
    print(sep)

    monkeypatch.chdir(os.path.join(_TEST_DIR, ".."))

    cfg, state = setup_experiment(dt=dt, Nz_E=Nz_E, Nx=Nx, T_s=T_s, drain=drain)
    results = run_simulation(
        cfg, state, dt=dt, time_simu=time_simu, store_every=store_every
    )

    os.makedirs(_PLOTS_DIR, exist_ok=True)
    try:
        plot_results(results, state, output_dir=_PLOTS_DIR, T_s=T_s, drain=drain)
    except Exception as e:
        print(f"Warning: plotting failed: {e}")

    is_valid, errors = validate_results(results, cfg, T_s=T_s)
    status = "PASSED" if is_valid else "FAILED"
    print(f"{sep}\n  {status} — Ts = {T_s} °C, {drain_label}\n{sep}\n")
    assert is_valid, "Hewitt ice cap validation failed:\n" + "\n".join(errors)
