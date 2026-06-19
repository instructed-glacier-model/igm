#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Experiment setup and simulation runner for Hewitt & Schoof (2017) ice cap test.

Implements the ice cap benchmark (Section 4.3) of:
  Hewitt & Schoof (2017). Models for polythermal ice sheets and glaciers.
  The Cryosphere, 11, 541-551.

Three implementation choices to faithfully reproduce Fig. 7:

1. Paper's rho=916 kg/m³ and g=9.8 m/s² are used only for the prescribed SIA
   velocity, so strain rates match Table 1. IGM's solver keeps its own values.

2. state.arrhenius is pinned once at setup to A = 75.7 MPa^-3 yr^-1 (paper's
   Table 1). IGM's temperature-dependent formula gives ~15.5 at -10 °C
   (warm-regime branch), which would cause ~70% excess strain heating. Since
   enthalpy only reads state.arrhenius (never overwrites it), one assignment
   suffices for the whole run.

3. Cold start: all ice at T_s, omega=0 (paper: "cold, temperate-free conditions").
"""

import os
import numpy as np
import tensorflow as tf
from typing import Dict, Any, Tuple

import igm
from igm.common import State
from igm.common.runner.configuration.loader import load_yaml_recursive
from igm.processes.enthalpy import enthalpy, compute_variables_enthalpy_state
from igm.processes.subglacial_hydrology import subglacial_hydrology


# Hewitt & Schoof Table 1 constants
_A_HEWITT = 2.4e-4 / 1e20  # 2.4e-24 Pa^-3 s^-1
_N_GLEN = 3.0
_SPY = 31556926.0  # seconds per year
_RHO_HEWITT = 916.0  # kg/m^3 — used only for prescribed velocity
_G_HEWITT = 9.8  # m/s^2  — used only for prescribed velocity

# Paper's A in IGM units (MPa^-3 yr^-1): 2.4e-24 Pa^-3 s^-1 * 1e18 * SPY ≈ 75.7
_A_PAPER_MPAYR = float(_A_HEWITT * 1e18 * _SPY)

# Ice cap geometry (Sec. 4.3)
_H0 = 1500.0  # m — central ice thickness
_R0 = 100_000.0  # m — ice cap radius


def _load_config():
    """Load base IGM configuration."""
    return load_yaml_recursive(os.path.join(igm.__path__[0], "conf"))


def _compute_flowline_velocity(state, A, rho, g, n, H0, R0):
    """Compute SIA velocity (U, V, W) for a pseudo-1D flowline ice cap.

    Thickness and slope depend on x only; V=0; W follows from incompressibility.
    """
    x = tf.cast(state.x, tf.float32)  # (Nx,)
    y = tf.cast(state.y, tf.float32)  # (Ny,)
    Nx = tf.shape(x)[0]
    Ny = tf.shape(y)[0]

    Hx = tf.maximum(H0 * (1.0 - (x / R0) ** 2.0), 0.0)  # (Nx,) thickness
    Sx = tf.abs(-2.0 * H0 * x / R0**2)  # (Nx,) |ds/dx|

    zeta = state.iceflow.discr_v.enthalpy.zeta  # (Nz,)
    dzeta = state.iceflow.discr_v.enthalpy.dzeta  # (Nz-1, 1, 1)

    H = Hx[None, None, :]  # (1, 1, Nx)
    z = zeta[:, None, None] * H  # (Nz, 1, Nx)

    coef = 2.0 * A * (rho * g) ** n / (n + 1.0)
    Umag = (
        coef
        * (Sx[None, None, :] ** n)
        * (H ** (n + 1.0) - tf.maximum(H - z, 0.0) ** (n + 1.0))
    )  # (Nz, 1, Nx)

    # Flow is outward from the divide: sign follows x
    U = tf.broadcast_to(Umag * tf.sign(x)[None, None, :], (tf.shape(zeta)[0], Ny, Nx))
    V = tf.zeros_like(U)

    dx = tf.cast(state.dx, tf.float32)
    dUdx = (U[:, :, 2:] - U[:, :, :-2]) / (2.0 * dx)
    dUdx = tf.pad(dUdx, [[0, 0], [0, 0], [1, 1]], mode="SYMMETRIC")

    dz = tf.broadcast_to(dzeta * H, (tf.shape(zeta)[0] - 1, Ny, Nx))
    dz_ext = tf.concat([dz, dz[-1:]], axis=0)
    W = -tf.cumsum(dUdx * dz_ext, axis=0, exclusive=True)

    return U * _SPY, V, W * _SPY


def setup_experiment(
    dt: float = 5.0,
    Nz_E: int = 50,
    Nx: int = 201,
    Ny: int = 2,
    T_s: float = -10.0,
    drain: bool = True,
) -> Tuple[Any, State]:
    """
    Initialize configuration and state for the Hewitt & Schoof ice cap test.

    Parabolic ice cap over [-R0, R0] with a prescribed SIA velocity field,
    cold start (all ice at T_s, omega=0), beta=0 (Tm=0 °C), and zero geothermal flux.

    Args:
        dt: Time step in years.
        Nz_E: Number of vertical enthalpy levels.
        Nx: Number of horizontal grid points.
        Ny: Number of y grid points (default 2: pseudo-1D flowline).
        T_s: Surface temperature in °C.
        drain: Whether to drain excess water from the ice column.

    Returns:
        (cfg, state) ready for run_simulation().
    """
    cfg = _load_config()
    Nz_U = Nz_E

    # Vertical discretization (uniform spacing)
    cfg.processes.iceflow.numerics.Nz = Nz_U
    cfg.processes.iceflow.numerics.vert_spacing = 1
    cfg.processes.enthalpy.numerics.Nz = Nz_E
    cfg.processes.enthalpy.numerics.vert_spacing = 1

    # Thermal: beta=0 enforces Tm = 0 °C everywhere (paper condition, Sec. 3)
    cfg.processes.enthalpy.thermal.beta = 0.0
    cfg.processes.enthalpy.thermal.K_ratio = 1.0e-2

    # Till storage: drainage_rate=0 means basal melt accumulates in the till
    # (no groundwater drainage, matching Hewitt & Schoof Sec. 4.3)
    cfg.processes.subglacial_hydrology.mode = "till_storage"
    cfg.processes.subglacial_hydrology.till_storage.drainage_rate = 0.0

    # Drainage / solver
    cfg.processes.enthalpy.drainage.drain_ice_column = drain
    cfg.processes.enthalpy.solver.allow_basal_refreezing = False
    cfg.processes.enthalpy.solver.override_basal_at_pmp = True
    cfg.processes.enthalpy.solver.correct_w_for_melt = False

    c_ice = cfg.processes.enthalpy.thermal.c_ice
    T_ref = cfg.processes.enthalpy.thermal.T_ref

    # Grid: fixed domain [-R0, R0]
    x = np.linspace(-_R0, _R0, Nx, dtype=np.float32)
    dx = float(x[1] - x[0])
    y = np.arange(-Ny // 2, Ny // 2, dtype=np.float32) * dx

    R = np.abs(np.meshgrid(x, y)[0])  # (Ny, Nx), flowline radius
    usurf = np.maximum(_H0 * (1.0 - (R / _R0) ** 2.0), 0.0).astype(np.float32)

    # State
    state = State()
    state.x = tf.constant(x)
    state.y = tf.constant(y)
    state.dx = tf.Variable(dx, trainable=False)
    state.dX = tf.Variable(dx * tf.ones((Ny, Nx), dtype=tf.float32), trainable=False)
    state.topg = tf.Variable(np.zeros((Ny, Nx), dtype=np.float32))
    state.thk = tf.Variable(usurf)
    state.usurf = tf.Variable(usurf)
    state.t = tf.Variable(0.0, trainable=False)
    state.dt = tf.Variable(dt, trainable=False)
    state.air_temp = tf.Variable(T_s * tf.ones((1, Ny, Nx)), trainable=False)
    state.basal_heat_flux = tf.zeros((Ny, Nx))
    state.h_water_till = tf.zeros((Ny, Nx))
    state.tau_ref = tf.zeros((Ny, Nx), dtype=tf.float32)  # no basal sliding
    state.U = tf.Variable(tf.zeros((Nz_U, Ny, Nx)), trainable=False)
    state.V = tf.Variable(tf.zeros((Nz_U, Ny, Nx)), trainable=False)
    state.W = tf.Variable(tf.zeros((Nz_U, Ny, Nx)), trainable=False)

    # Cold start: all ice at T_s, omega=0 (paper Sec. 4.3: "cold, temperate-free")
    T_surface_K = T_s + 273.15
    T_init = T_surface_K * tf.ones((Nz_E, Ny, Nx), dtype=tf.float32)
    state.E = c_ice * (T_init - T_ref)
    state.T = T_init
    state.omega = tf.zeros_like(state.E)

    enthalpy.initialize(cfg, state)
    subglacial_hydrology.initialize(cfg, state)

    # Pin Arrhenius at paper's value (Table 1: 75.7 MPa^-3 yr^-1).
    # IGM's temperature-dependent formula gives ~15.5 at -10 °C (warm-regime
    # branch), which would cause ~70% excess strain heating. enthalpy only reads
    # state.arrhenius (never writes it after init), so this single assignment holds
    # for the entire run without any per-step reset.
    state.arrhenius = _A_PAPER_MPAYR * tf.ones_like(state.arrhenius)

    # Prescribed SIA velocity using paper's rho/g for correct strain-rate magnitude
    U, V, W = _compute_flowline_velocity(
        state, _A_HEWITT, _RHO_HEWITT, _G_HEWITT, _N_GLEN, _H0, _R0
    )
    state.U.assign(U)
    state.V.assign(V)
    state.W.assign(W)

    return cfg, state


def run_simulation(
    cfg,
    state: State,
    dt: float = 5.0,
    time_simu: float = 50_000.0,
    store_every: int = 1,
) -> Dict[str, np.ndarray]:
    """
    Run the Hewitt ice cap simulation for time_simu years.

    Returns a dict with keys times, T (K), omega, drainage (m/yr), E, E_pmp;
    shapes (Nt, ...) where Nt = number of stored snapshots.
    """
    time_list = np.arange(dt, time_simu + 0.5 * dt, dt)
    stored = {k: [] for k in ("times", "T", "omega", "drainage", "E", "E_pmp")}

    for step, t in enumerate(time_list):
        state.t.assign(float(t))
        enthalpy.update(cfg, state)
        subglacial_hydrology.update(cfg, state)

        if (step + 1) % store_every == 0 or step == len(time_list) - 1:
            compute_variables_enthalpy_state(cfg, state)
            max_T_C = float(tf.reduce_max(state.T)) - 273.15
            max_omega = float(tf.reduce_max(state.omega))
            print(
                f"  {t:7.0f} / {time_simu:.0f} yr ({100 * t / time_simu:3.0f}%)"
                f"  max T = {max_T_C:.2f} °C  max ω = {max_omega:.4f}"
            )
            stored["times"].append(float(t))
            stored["T"].append(state.T.numpy())
            stored["omega"].append(state.omega.numpy())
            stored["drainage"].append(state.basal_melt_rate.numpy())
            stored["E"].append(state.E.numpy())
            stored["E_pmp"].append(state.E_pmp.numpy())

    return {k: np.array(v) for k, v in stored.items()}
