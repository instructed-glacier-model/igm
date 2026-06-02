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
    # Function-local groups reflect shared computation (E_pmp shared across several vars)
    _temp_group = frozenset({"T", "omega", "E_pmp", "T_pmp", "T_pa", "T_pa_b"})
    _surf_group = frozenset({"E_s", "T_s"})
    _all = _temp_group | _surf_group

    want = set(requested) if requested is not None else _all

    T = None
    if want & _temp_group:
        E_pmp, T_pmp = compute_pmp(cfg, state)
        T, omega = compute_temperature(cfg, state, E_pmp)
        if "E_pmp" in want:
            state.E_pmp = E_pmp
        if "T_pmp" in want:
            state.T_pmp = T_pmp
        if "T" in want:
            state.T = T
        if "omega" in want:
            state.omega = omega

    if T is not None and want & {"T_pa", "T_pa_b"}:
        T_pa = compute_pa(cfg, state, T)
        if "T_pa" in want:
            state.T_pa = T_pa
        if "T_pa_b" in want:
            state.T_pa_b = T_pa[0]

    if want & _surf_group:
        E_s, T_s = compute_surface(cfg, state)
        if "E_s" in want:
            state.E_s = E_s
        if "T_s" in want:
            state.T_s = T_s
