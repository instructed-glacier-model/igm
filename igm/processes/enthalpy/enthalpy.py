#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from omegaconf import DictConfig

from igm.common import State

from .dissipation import compute_dissipation
from .solver import update_enthalpy
from .surface import compute_surface
from .temperature import compute_pa, compute_pmp, compute_temperature
from .utils import checks, initialize_enthalpy_fields


def initialize(cfg: DictConfig, state: State) -> None:
    """Initialize enthalpy module state variables."""

    # Do preliminary checks
    checks(cfg, state)

    # Initialize enthalpy fields
    initialize_enthalpy_fields(cfg, state)


def update(cfg: DictConfig, state: State) -> None:
    """Update enthalpy and derived fields."""
    if hasattr(state, "logger"):
        state.logger.info(f"Update ENTHALPY at time: {state.t.numpy()}")

    # (i) SOURCE & BOUNDARY TERMS

    # Surface enthalpy BC from air temperature
    E_s, _ = compute_surface(cfg, state)

    # Pressure melting point enthalpy
    E_pmp, _ = compute_pmp(cfg, state)

    # Volumetric strain heating and basal frictional heating
    strain_heat, friction_heat = compute_dissipation(cfg, state)

    # (ii) SOLVE FOR ENTHALPY (state.E, state.basal_melt_rate)
    update_enthalpy(cfg, state, strain_heat, friction_heat, E_pmp, E_s)


def finalize(cfg: DictConfig, state: State) -> None:
    pass


def compute_diagnostics(cfg: DictConfig, state: State, requested=None) -> None:
    E_s, T_s = compute_surface(cfg, state)
    E_pmp, T_pmp = compute_pmp(cfg, state)
    T, omega = compute_temperature(cfg, state, E_pmp)
    T_pa = compute_pa(cfg, state, T)
    T_pa_b = T_pa[0]

    computed = {
        "E_s": E_s,
        "T_s": T_s,
        "E_pmp": E_pmp,
        "T_pmp": T_pmp,
        "T": T,
        "omega": omega,
        "T_pa": T_pa,
        "T_pa_b": T_pa_b,
    }
    for var in (requested if requested is not None else computed):
        setattr(state, var, computed[var])
