#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Plotting utilities for the Hewitt & Schoof (2017) ice cap enthalpy test."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, Any
from scipy.interpolate import RectBivariateSpline

from igm.common import State


def plot_results(
    results: Dict[str, Any],
    state: State,
    output_dir: str = ".",
    T_s: float = -10.0,
    drain: bool = True,
) -> None:
    """
    Generate diagnostic plots for the Hewitt ice cap test.

    Produces two figures per case:
      - hewitt_T_evolution_<tag>.png — temperature cross-section + depth-averaged evolution
      - hewitt_polythermal_<tag>.png — polythermal structure (T/omega) + drainage
    """
    ts_tag = f"Ts{int(T_s)}".replace("-", "m")
    drain_tag = "drain" if drain else "nodrain"
    tag = f"{ts_tag}_{drain_tag}"

    x_km = state.x.numpy() / 1000.0
    times = results["times"]
    T = results["T"]       # (Nt, Nz, Ny, Nx)
    omega = results["omega"]
    E = results["E"]
    E_pmp = results["E_pmp"]
    drainage = results["drainage"]  # (Nt, Ny, Nx)

    zeta = state.iceflow.discr_v.enthalpy.zeta.numpy()
    thk = state.thk.numpy()
    z_all = zeta[:, np.newaxis, np.newaxis] * thk[np.newaxis, :, :]  # (Nz, Ny, Nx)

    iy = T.shape[2] // 2
    mask = (x_km >= 0.0) & (x_km <= 100.0)
    x_sub = x_km[mask]

    T_last = T[-1, :, iy, :] - 273.15   # (Nz, Nx) in Celsius
    z_last = z_all[:, iy, :]             # (Nz, Nx)
    omega_last = omega[-1, :, iy, :]

    cold = (E[-1, :, iy, :] - E_pmp[-1, :, iy, :]) < 0.0
    field = np.zeros_like(T_last)
    field[cold] = T_last[cold]
    field[~cold] = omega_last[~cold] * 100  # convert to %

    z_plot = z_last[:, mask]
    field_plot = field[:, mask]

    melt_ylim = 45 if T_s == -1.0 else 25
    _plot_temperature(output_dir, tag, x_sub, z_last, T_last, mask, times, T, iy)
    _plot_polythermal(output_dir, tag, x_sub, z_plot, field_plot, state, mask, times, drainage, iy, melt_ylim)


# ---------------------------------------------------------------------------
# Standard figures
# ---------------------------------------------------------------------------

def _plot_temperature(output_dir, tag, x_sub, z_last, T_last, mask, times, T, iy):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    pcm = ax1.pcolormesh(
        x_sub, z_last[:, mask], T_last[:, mask],
        shading="auto", cmap="turbo", vmin=-10, vmax=0,
    )
    plt.colorbar(pcm, ax=ax1, label="Temperature (°C)")
    ax1.set_xlabel("x (km)")
    ax1.set_ylabel("z (m)")
    ax1.set_title(f"Temperature at t = {times[-1]:.0f} yr  (Hewitt & Schoof 2017, Fig. 7)")

    dt_curve = times[-1] / 10.0
    requested = np.arange(0, times[-1] + 0.5 * dt_curve, dt_curve)
    for t_req in requested:
        i = int(np.argmin(np.abs(times - t_req)))
        T_mean = np.mean(T[i, :, iy, :] - 273.15, axis=0)
        ax2.plot(x_sub, T_mean[mask], label=f"{times[i]:.0f} yr")

    ax2.set_xlabel("x (km)")
    ax2.set_ylabel("Depth-averaged temperature (°C)")
    ax2.set_title("Evolution of depth-averaged temperature")
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/hewitt_T_evolution_{tag}.png", dpi=150)
    plt.close()


def _plot_polythermal(output_dir, tag, x_sub, z_plot, field_plot, state, mask, times, drainage, iy, melt_ylim=25):
    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    cmap_combined = np.vstack([
        plt.cm.jet(np.linspace(0, 1, 200)),
        plt.cm.Blues(np.linspace(0, 1, 200)),
    ])
    cmap_field = mcolors.ListedColormap(cmap_combined)
    norm = mcolors.TwoSlopeNorm(vmin=-10, vcenter=0, vmax=3)

    X = np.tile(x_sub, (field_plot.shape[0], 1))
    ax1.pcolormesh(X, z_plot, field_plot, cmap=cmap_field, norm=norm, shading="auto")
    sm = plt.cm.ScalarMappable(cmap=cmap_field, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=[ax1, ax2], fraction=0.045, pad=0.04)
    cb.set_label("Temperature (°C) / Water fraction (%)")
    cb.set_ticks([-10, -5, 0, 1, 2, 3])

    ax1.set_ylabel("z (m)")
    ax1.set_title(f"Polythermal structure at t = {times[-1]:.0f} yr")

    U = state.U[:, iy, :].numpy()[:, mask]
    W = state.W[:, iy, :].numpy()[:, mask]
    Nz = U.shape[0]
    x_stream = np.linspace(x_sub.min(), x_sub.max(), U.shape[1])
    z_stream = np.linspace(0, np.max(z_plot), Nz)
    x_i = np.linspace(x_stream.min(), x_stream.max(), 300)
    z_i = np.linspace(z_stream.min(), z_stream.max(), 300)
    U_i = RectBivariateSpline(z_stream, x_stream, np.nan_to_num(U))(z_i, x_i)
    W_i = RectBivariateSpline(z_stream, x_stream, np.nan_to_num(W))(z_i, x_i)
    ax1.streamplot(x_i, z_i, U_i, W_i * 1000, color="white", density=0.5,
                   linewidth=1.5, arrowstyle="-")

    drain_last = drainage[-1, iy, mask] * 1000.0  # m/yr → mm/yr
    ax2.plot(x_sub, drain_last, color="black", lw=2)
    ax2.set_ylabel("melt (mm yr⁻¹)")
    ax2.set_xlabel("x (km)")
    ax2.set_ylim(0, melt_ylim)
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3)

    plt.savefig(f"{output_dir}/hewitt_polythermal_{tag}.png", dpi=200)
    plt.close()

