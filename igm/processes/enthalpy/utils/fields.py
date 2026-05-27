#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import tensorflow as tf
from omegaconf import DictConfig
from typing import Dict

from igm.common import State
from ..surface import compute_surface
from ..temperature import compute_pmp, compute_temperature, compute_pa


def initialize_enthalpy_fields(cfg: DictConfig, state: State) -> None:
    """
    Initialize enthalpy-related fields with default values.

    Sets up 2D and 3D fields required for the enthalpy model if they do not
    already exist in state. Initializes enthalpy at the pressure melting point
    and default values for till hydrology and friction.

    Initializes state.basal_melt_rate (m ice yr^-1), state.E (J kg^-1),
    and state.basal_heat_flux (W m^-2).

    Till-related fields (state.h_water_till, state.effective_pressure) are
    now the responsibility of the `effective_pressure` process module
    (mode `vanpelt_bueler`); friction-related fields (state.phi, state.tauc)
    were folded into the `mohr_coulomb` sliding law in iceflow.
    """
    cfg_thermal = cfg.processes.enthalpy.thermal
    Nz = cfg.processes.enthalpy.numerics.Nz
    Ny = state.thk.shape[0]
    Nx = state.thk.shape[1]
    shape_2d = (Ny, Nx)
    shape_3d = (Nz, Ny, Nx)

    if not hasattr(state, "basal_melt_rate"):
        state.basal_melt_rate = tf.zeros(shape_2d)

    if not hasattr(state, "E"):
        c_ice = cfg_thermal.c_ice
        T_pmp_ref = cfg_thermal.T_pmp_ref
        T_ref = cfg_thermal.T_ref
        state.E = c_ice * (T_pmp_ref - T_ref) * tf.ones(shape_3d)

    if not hasattr(state, "basal_heat_flux"):
        basal_heat_flux_ref = cfg_thermal.basal_heat_flux_ref
        state.basal_heat_flux = basal_heat_flux_ref * tf.ones(shape_2d)

    # Defensive fallback: in a normal run, iceflow.initialize and/or the
    # standalone arrhenius module set state.arrhenius. We keep this guard
    # so enthalpy can be exercised in isolation (e.g. unit tests).
    if not hasattr(state, "arrhenius"):
        cfg_physics = cfg.processes.iceflow.physics
        arrhenius = cfg_physics.init_arrhenius
        state.arrhenius = arrhenius * tf.ones(shape_2d)

    # Fallback: initialize vertical discretization for testing purposes.
    # In principle, the iceflow module should provide this.
    if not hasattr(state, "iceflow"):
        from igm.processes.iceflow.iceflow import Iceflow
        from igm.processes.iceflow.vertical import VerticalDiscrs

        state.iceflow = Iceflow()
        cfg_numerics = cfg.processes.iceflow.numerics

        vertical_basis = cfg_numerics.basis_vertical.lower()
        vertical_discr = VerticalDiscrs[vertical_basis](cfg)
        state.iceflow.discr_v = vertical_discr


def compute_variables_enthalpy_state(cfg: DictConfig, state: State) -> None:

    # Compute auxiliary variables
    E_s, T_s = compute_surface(cfg, state)
    E_pmp, T_pmp = compute_pmp(cfg, state)
    T, omega = compute_temperature(cfg, state, E_pmp)
    T_pa = compute_pa(cfg, state, T)
    T_pa_b = T_pa[0]

    # Save in state
    state.E_s = E_s
    state.T_s = T_s
    state.E_pmp = E_pmp
    state.T_pmp = T_pmp
    state.T = T
    state.omega = omega
    state.T_pa = T_pa
    state.T_pa_b = T_pa_b


def compute_variables_enthalpy_np(
    cfg: DictConfig, state: State
) -> Dict[str, np.ndarray]:
    # Compute auxiliary variables
    E_s, T_s = compute_surface(cfg, state)
    E_pmp, T_pmp = compute_pmp(cfg, state)
    T, omega = compute_temperature(cfg, state, E_pmp)
    T_pa = compute_pa(cfg, state, T)
    T_pa_b = T_pa[0]

    out = {
        "E": state.E.numpy(),
        "E_pmp": E_pmp.numpy(),
        "E_s": E_s.numpy(),
        "T": T.numpy(),
        "T_pa": T_pa.numpy(),
        "T_pa_b": T_pa_b.numpy(),
        "T_pmp": T_pmp.numpy(),
        "T_s": T_s.numpy(),
        "arrhenius": state.arrhenius.numpy(),
        "basal_melt_rate": state.basal_melt_rate.numpy(),
        "omega": omega.numpy(),
    }
    # Till-related fields are now owned by `effective_pressure.vanpelt_bueler`
    # and the `mohr_coulomb` sliding law. Include them only if set elsewhere.
    for name in ("h_water_till", "effective_pressure", "N", "phi", "tauc"):
        val = getattr(state, name, None)
        if val is not None:
            out[name] = val.numpy()
    return out
