#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from omegaconf import DictConfig

from igm.common import State

from .dissipation import compute_dissipation
from .solver import update_enthalpy
from .surface import compute_surface
from .temperature import compute_temperature, compute_pmp
from .utils import checks, initialize_enthalpy_fields


def initialize(cfg: DictConfig, state: State) -> None:
    """Initialize enthalpy module state variables."""

    # Do preliminary checks
    checks(cfg, state)

    # Initialize enthalpy fields
    initialize_enthalpy_fields(cfg, state)

    # Compute E_pmp
    E_pmp, _ = compute_pmp(cfg, state)

    # Compute (T, omega) from E and publish on state for downstream
    # consumers (e.g. the standalone `arrhenius` process module, the
    # `effective_pressure` module with `vanpelt_bueler` mode, ...).
    T, omega = compute_temperature(cfg, state, E_pmp)
    state.T = T
    state.omega = omega

    _publish_2d_temperatures(cfg, state, T)


def update(cfg: DictConfig, state: State) -> None:
    """Update enthalpy and derived (T, omega) fields."""
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

    # (iii) DERIVE QUANTITIES

    # Temperature and water content; publish on state so the standalone
    # `arrhenius` module (and any other consumer) can pick them up.
    T, omega = compute_temperature(cfg, state, E_pmp)
    state.T = T
    state.omega = omega

    _publish_2d_temperatures(cfg, state, T)


def _publish_2d_temperatures(cfg: DictConfig, state: State, T) -> None:
    """Publish 2D surface and basal temperatures on state, using the
    legacy `temppasurf` / `temppabase` names (Kelvin, pressure-adjusted)."""
    cfg_phys = cfg.processes.iceflow.physics
    cfg_thermal = cfg.processes.enthalpy.thermal
    # Surface temperature: p_ice = 0 at the surface, so T_pa_s = T_s
    state.temppasurf = T[-1]
    # Basal pressure-adjusted temperature:
    #     T_pa_b = T_b + beta * rho_ice * g * thk
    # so that at PMP it equals T_pmp_ref (= 273.15 K) regardless of depth.
    state.temppabase = T[0] + cfg_thermal.beta * cfg_phys.ice_density * cfg_phys.gravity_cst * state.thk


def finalize(cfg: DictConfig, state: State) -> None:
    pass
