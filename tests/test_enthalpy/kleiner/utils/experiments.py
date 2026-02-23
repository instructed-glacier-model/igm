#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Experiment setup functions for Kleiner et al. (2015) benchmarks."""

import os
import numpy as np
import tensorflow as tf

import igm
from igm.common import State
from igm.common.runner.configuration.loader import load_yaml_recursive
from igm.processes.enthalpy import enthalpy, compute_variables_enthalpy_state


def _load_config():
    """Load base IGM configuration."""
    return load_yaml_recursive(os.path.join(igm.__path__[0], "conf"))


def _init_state(Ny, Nx, H, dt, T_air):
    """Initialize common state fields."""
    state = State()
    state.thk = tf.Variable(H * tf.ones((Ny, Nx)), trainable=False)
    state.topg = tf.Variable(tf.zeros((Ny, Nx)), trainable=False)
    state.usurf = state.topg + state.thk
    state.t = tf.Variable(0.0, trainable=False)
    state.dt = tf.Variable(dt, trainable=False)
    state.air_temp = tf.Variable(T_air * tf.ones((1, Ny, Nx)), trainable=False)
    state.dx = tf.Variable(1000.0, trainable=False)
    state.dX = tf.Variable(1000.0 * tf.ones((Ny, Nx)), trainable=False)
    state.h_water_till = tf.zeros((Ny, Nx))
    return state


def _configure_enthalpy_solver(cfg, drain=False, refreezing=True, correct_w=False):
    """Configure common enthalpy solver settings."""
    cfg.processes.enthalpy.thermal.K_ratio = 1e-5
    cfg.processes.enthalpy.till.hydro.h_water_till_max = 200.0
    cfg.processes.enthalpy.till.hydro.drainage_rate = 0.0
    cfg.processes.enthalpy.drainage.drain_ice_column = drain
    cfg.processes.enthalpy.solver.allow_basal_refreezing = refreezing
    cfg.processes.enthalpy.solver.correct_w_for_melt = correct_w


def setup_experiment_a(dt: float = 200.0, Nz_E: int = 50):
    """
    Initialize configuration and state for Experiment A.

    Transient thermal evolution with three phases:
    - Phase I (0-100 ky): Equilibration with cold surface (-30C)
    - Phase II (100-150 ky): Warming pulse (-5C)
    - Phase III (150-300 ky): Return to cold surface (-30C)
    """
    cfg = _load_config()
    Nz_U, Ny, Nx = 10, 2, 2

    # Vertical discretization
    cfg.processes.iceflow.numerics.Nz = Nz_U
    cfg.processes.iceflow.numerics.vert_spacing = 1
    cfg.processes.enthalpy.numerics.Nz = Nz_E
    cfg.processes.enthalpy.numerics.vert_spacing = 1
    _configure_enthalpy_solver(cfg, refreezing=True)

    # Initialize state
    state = _init_state(Ny, Nx, H=1000.0, dt=dt, T_air=-30.0)
    state.basal_heat_flux = 0.042 * tf.ones((Ny, Nx))
    state.U = tf.Variable(tf.zeros((Nz_U, Ny, Nx)), trainable=False)
    state.V = tf.Variable(tf.zeros((Nz_U, Ny, Nx)), trainable=False)
    state.W = tf.Variable(tf.zeros((Nz_U, Ny, Nx)), trainable=False)

    # Initialize enthalpy with cold ice (-30C = 243.15K)
    enthalpy.initialize(cfg, state)
    c_ice = cfg.processes.enthalpy.thermal.c_ice
    T_ref = cfg.processes.enthalpy.thermal.T_ref
    T_init = 243.15 * tf.ones((Nz_E, Ny, Nx))
    state.E = c_ice * (T_init - T_ref)
    state.T = T_init
    state.omega = tf.zeros_like(state.E)

    return cfg, state


def setup_experiment_b(Nz_E: int = 500):
    """
    Initialize configuration and state for Experiment B.

    Polythermal parallel-sided slab with strain heating.
    """
    # Physical parameters
    H, gamma = 200.0, 4.0 * np.pi / 180.0
    A, rho, g = 5.3e-24, 910.0, 9.81
    spy, a_perp = 31556926.0, 0.2
    Nz_U, Ny, Nx = 10, 2, 2

    cfg = _load_config()

    # Vertical discretization
    cfg.processes.iceflow.numerics.Nz = Nz_U
    cfg.processes.iceflow.numerics.vert_spacing = 1
    cfg.processes.enthalpy.numerics.Nz = Nz_E
    cfg.processes.enthalpy.numerics.vert_spacing = 1
    _configure_enthalpy_solver(cfg, refreezing=False)

    # Initialize state
    state = _init_state(Ny, Nx, H=H, dt=1.0, T_air=-3.0)
    state.basal_heat_flux = tf.zeros((Ny, Nx))

    # Prescribed velocity field (Eqs. 13-15 from paper)
    z = tf.reshape(tf.linspace(0.0, H, Nz_U), (Nz_U, 1, 1))
    z = tf.tile(z, [1, Ny, Nx])
    tau = rho * g * tf.sin(gamma)
    vx = A * (tau**3) / 2 * (H**4 - (H - z) ** 4) * spy

    state.U = tf.Variable(vx, trainable=False)
    state.V = tf.Variable(tf.zeros((Nz_U, Ny, Nx)), trainable=False)
    state.W = tf.Variable(-a_perp * tf.ones((Nz_U, Ny, Nx)), trainable=False)

    # Initialize enthalpy near melting point
    c_ice = cfg.processes.enthalpy.thermal.c_ice
    T_ref = cfg.processes.enthalpy.thermal.T_ref
    T_init = (273.15 - 1.5) * tf.ones((Nz_E, Ny, Nx))
    state.E = c_ice * (T_init - T_ref)
    state.T = T_init
    state.omega = tf.zeros_like(state.E)
    state.arrhenius = tf.constant(A * 1e18 * spy, dtype=tf.float32) * tf.ones((Ny, Nx))
    enthalpy.initialize(cfg, state)

    return cfg, state


def run_simulation_a(cfg, state, dt: float):
    """Run Experiment A simulation (300 ky thermal evolution)."""
    times = np.arange(dt, 300001.0, dt)
    results = {"time": [], "T_base": [], "melt_rate": [], "till_water": []}

    for it, t in enumerate(times):
        state.t.assign(t)

        # Surface temperature forcing
        T_surface = -5.0 if 100000.0 <= t < 150000.0 else -30.0
        state.air_temp.assign(T_surface * tf.ones((1, 2, 2)))

        enthalpy.update(cfg, state)
        compute_variables_enthalpy_state(cfg, state)

        results["time"].append(t)
        results["T_base"].append(state.T[0, 0, 0].numpy() - 273.15)
        results["melt_rate"].append(state.basal_melt_rate[0, 0].numpy())
        results["till_water"].append(state.h_water_till[0, 0].numpy())

        if it % 100 == 0:
            phase = "I" if t < 100000.0 else ("II" if t < 150000.0 else "III")
            print(
                f"  t={t/1000:.0f} ky, Phase {phase}, T_b={results['T_base'][-1]:.2f}C"
            )

    return {k: np.array(v) for k, v in results.items()}


def run_simulation_b(cfg, state, max_iter: int = 1500, tol: float = 1e-3):
    """Run Experiment B to steady state."""
    dt = state.dt.numpy()
    prev_E = state.E.numpy().copy()
    A, spy = 5.3e-24, 31556926.0

    for it in range(max_iter):
        state.t.assign(it * dt)
        state.arrhenius = (A * 1e18 * spy) * tf.ones_like(state.arrhenius)
        enthalpy.update(cfg, state)

        if it % 10 == 0 and it > 0:
            max_change = np.max(np.abs(state.E.numpy() - prev_E))
            if max_change < tol:
                print(f"  Converged at iteration {it}")
                break
            prev_E = state.E.numpy().copy()
            if it % 100 == 0:
                compute_variables_enthalpy_state(cfg, state)
                print(
                    f"  Iteration {it}: T_base={state.T[0, 0, 0].numpy() - 273.15:.2f}C"
                )

    compute_variables_enthalpy_state(cfg, state)


def extract_results_b(state):
    """Extract results from Experiment B state."""
    H = state.thk[0, 0].numpy()
    Nz = state.T.shape[0]
    z = np.linspace(0, H, Nz)

    E = state.E[:, 0, 0].numpy()
    T = state.T[:, 0, 0].numpy() - 273.15
    omega = state.omega[:, 0, 0].numpy() * 100

    cts_idx = np.where(omega > 0.01)[0]
    cts_position = z[cts_idx[-1]] if len(cts_idx) > 0 else 0.0

    return {"z": z, "E": E, "T": T, "omega": omega, "cts_position": cts_position}
