#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""
Live dashboard for IGM simulations.

Two modes selectable via config (mode: "2d" or "3d"):

  2d — Matplotlib 4-panel planview dashboard:
       thickness, velocity, volume/area time series, live stats + IGM logo

  3d — PyVista GPU-accelerated 3D view:
       ice surface colored by speed, bedrock underneath,
       with on-screen text overlay for live stats
"""

import numpy as np
import tensorflow as tf
import time as clock


# ─── helpers ────────────────────────────────────────────────────────────────

def _ice_volume_km3(thk, dx):
    return float(np.sum(thk) * dx * dx / 1e9)

def _ice_area_km2(thk, dx):
    return float(np.sum(np.array(thk) > 1.0) * dx * dx / 1e6)

def _live_ela(thk, smb, usurf):
    """Estimate ELA as mean surface altitude where SMB ~ 0 and ice exists."""
    thk = np.array(thk)
    smb = np.array(smb)
    usurf = np.array(usurf)
    mask = (thk > 1.0) & (np.abs(smb) < 0.5)
    if np.any(mask):
        return float(np.mean(usurf[mask]))
    return np.nan


# ═══════════════════════════════════════════════════════════════════════════
#  2D MODE  (matplotlib)
# ═══════════════════════════════════════════════════════════════════════════

def _init_2d(cfg, state):
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, LightSource
    from matplotlib.gridspec import GridSpec

    p = cfg.outputs.live_dashboard
    plt.ion()
    plt.style.use("dark_background")

    extent = [float(np.min(state.x)), float(np.max(state.x)),
              float(np.min(state.y)), float(np.max(state.y))]
    domain_w = extent[1] - extent[0]
    domain_h = extent[3] - extent[2]
    aspect = domain_h / domain_w  # domain aspect ratio

    # figure width from config; compute height so maps fill tightly
    fig_w = p.figsize[0]
    # each map panel gets ~half the usable width (minus margins/wspace/colorbars)
    usable_w = fig_w * 0.82  # approx usable fraction after margins + colorbars
    panel_w = usable_w / 2
    panel_h = panel_w * aspect
    # time-series row = 25% of map row height
    ts_h = panel_h * 0.25
    # total figure height (with margins)
    fig_h = (panel_h + ts_h) / 0.82

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=p.dpi, facecolor="#0e1117")
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25,
                  left=0.06, right=0.94, top=0.94, bottom=0.08,
                  height_ratios=[3, 1])

    ax_thk = fig.add_subplot(gs[0, 0])
    ax_vel = fig.add_subplot(gs[0, 1])
    ax_ts  = fig.add_subplot(gs[1, :])  # full width

    for ax in [ax_thk, ax_vel]:
        ax.set_aspect("equal")

    # hillshade (static)
    ls = LightSource(azdeg=315, altdeg=35)
    hs = ls.hillshade(np.array(state.topg), vert_exag=2,
                      dx=float(state.dx), dy=float(state.dx))

    # ── thickness panel ──
    ax_thk.imshow(hs, origin="lower", cmap="gray", extent=extent, vmin=0, vmax=1)
    im_thk = ax_thk.imshow(
        np.where(state.thk > 0, state.thk, np.nan),
        origin="lower", cmap="cividis", extent=extent,
        vmin=0, vmax=p.thk_max, alpha=0.85,
    )
    cb = fig.colorbar(im_thk, ax=ax_thk, fraction=0.046, pad=0.04, label="Thickness (m)")
    cb.ax.yaxis.label.set_color("white"); cb.ax.yaxis.label.set_fontsize(14)
    cb.ax.tick_params(colors="white", labelsize=11)
    ax_thk.set_title("Ice Thickness", color="#58a6ff", fontsize=18, fontweight="bold")
    ax_thk.set_xticks([]); ax_thk.set_yticks([])

    # ── velocity panel ──
    ax_vel.imshow(hs, origin="lower", cmap="gray", extent=extent, vmin=0, vmax=1)
    im_vel = ax_vel.imshow(
        np.full_like(np.array(state.thk), np.nan),
        origin="lower", cmap="inferno", extent=extent,
        norm=LogNorm(vmin=1, vmax=p.vel_max), alpha=0.9,
    )
    cb2 = fig.colorbar(im_vel, ax=ax_vel, fraction=0.046, pad=0.04, label="Speed (m/a)")
    cb2.ax.yaxis.label.set_color("white"); cb2.ax.yaxis.label.set_fontsize(14)
    cb2.ax.tick_params(colors="white", labelsize=11)
    ax_vel.set_title("Velocity", color="#58a6ff", fontsize=18, fontweight="bold")
    ax_vel.set_xticks([]); ax_vel.set_yticks([])

    # ── time-series panel (full width, dual y-axis) ──
    t0 = float(cfg.processes.time.start)
    ax_ts.set_facecolor("#161b22")
    ax_ts.set_xlabel("Time (years)", color="gray", fontsize=14)
    ax_ts.set_ylabel("ELA (m)", color="#ff7b72", fontsize=14)
    ax_ts.tick_params(axis="y", colors="#ff7b72", labelsize=11)
    ax_ts.tick_params(axis="x", colors="gray", labelsize=11)
    ax_ts.set_xlim(t0, cfg.processes.time.end)
    ax_ts.grid(True, alpha=0.15, color="white")
    for sp in ax_ts.spines.values():
        sp.set_color("#30363d")

    # Volume on right y-axis
    ax_ts2 = ax_ts.twinx()
    ax_ts2.set_ylabel("Volume (km³)", color="#7ee787", fontsize=14)
    ax_ts2.tick_params(axis="y", colors="#7ee787", labelsize=11)
    ax_ts2.yaxis.set_label_position("right")
    ax_ts2.yaxis.tick_right()
    for sp in ax_ts2.spines.values():
        sp.set_color("#30363d")

    # ELA data will be populated on first update (smbpar not yet available at init)
    state._dash_has_smbpar = None  # deferred check
    state._dash_ela_times = []
    state._dash_ela_vals = []

    state._dash_fig = fig
    state._dash_ax_thk = ax_thk
    state._dash_ax_vel = ax_vel
    state._dash_ax_ts = ax_ts
    state._dash_ax_ts2 = ax_ts2
    state._dash_extent = extent
    state._dash_contour = None

    state._dash_times = []
    state._dash_vols = []

    state._dash_save_count = 0
    state._dash_wall_start = clock.time()

    fig.canvas.draw()
    fig.canvas.flush_events()


def _update_2d(cfg, state):
    if not state.saveresult:
        return

    p = cfg.outputs.live_dashboard
    state._dash_save_count += 1
    if state._dash_save_count % p.update_freq != 0:
        return

    # deferred smbpar check (not available at init time)
    if state._dash_has_smbpar is None:
        state._dash_has_smbpar = hasattr(state, "smbpar")
        if state._dash_has_smbpar:
            state._dash_ela_times = state.smbpar[:, 0].tolist()
            state._dash_ela_vals = state.smbpar[:, 3].tolist()

    t = float(state.t.numpy()) if hasattr(state.t, "numpy") else float(state.t)
    thk = np.array(state.thk)
    dx = float(state.dx)
    extent = state._dash_extent
    ubar = np.array(state.ubar)
    vbar = np.array(state.vbar)
    vel = np.sqrt(ubar**2 + vbar**2)

    # thickness
    state._dash_ax_thk.images[-1].set_data(np.where(thk > 0, thk, np.nan))

    # velocity
    state._dash_ax_vel.images[-1].set_data(
        np.where(thk > 0, np.clip(vel, 1, None), np.nan))

    # ice edge contour on velocity panel
    if state._dash_contour is not None:
        try:
            state._dash_contour.remove()
        except Exception:
            pass
    try:
        state._dash_contour = state._dash_ax_vel.contour(
            np.linspace(extent[0], extent[1], thk.shape[1]),
            np.linspace(extent[2], extent[3], thk.shape[0]),
            thk, levels=[1.0], colors=["#58a6ff"], linewidths=0.6, alpha=0.7,
        )
    except Exception:
        state._dash_contour = None

    # single time display as figure suptitle
    state._dash_fig.suptitle(f"t = {t:.0f} yr",
                              color="white", fontsize=20, fontweight="bold")

    # time series
    vol = _ice_volume_km3(thk, dx)
    vmax = float(np.max(vel[thk > 0])) if np.any(thk > 0) else 0.0
    elapsed = clock.time() - state._dash_wall_start
    dt = float(state.dt.numpy()) if hasattr(state.dt, "numpy") else float(state.dt)

    state._dash_times.append(t)
    state._dash_vols.append(vol)

    # compute live ELA if no smbpar
    if not state._dash_has_smbpar and hasattr(state, "smb"):
        ela = _live_ela(thk, state.smb, state.usurf)
        state._dash_ela_times.append(t)
        state._dash_ela_vals.append(ela)

    ax_ts = state._dash_ax_ts
    ax_ts2 = state._dash_ax_ts2
    ax_ts.cla()
    ax_ts2.cla()
    ax_ts.set_facecolor("#161b22")
    ax_ts.grid(True, alpha=0.15, color="white")
    ax_ts.set_xlabel("Time (years)", color="gray", fontsize=14)
    ax_ts.set_ylabel("ELA (m)", color="#ff7b72", fontsize=14)
    ax_ts.tick_params(axis="y", colors="#ff7b72", labelsize=11)
    ax_ts.tick_params(axis="x", colors="gray", labelsize=11)
    ax_ts.set_xlim(state._dash_times[0], cfg.processes.time.end)
    for sp in ax_ts.spines.values():
        sp.set_color("#30363d")
    # ELA on left axis
    if state._dash_has_smbpar:
        ax_ts.plot(state._dash_ela_times, state._dash_ela_vals,
                   color="#ff7b72", linewidth=2, linestyle="--", alpha=0.8)
    elif len(state._dash_ela_vals) > 0:
        ax_ts.plot(state._dash_ela_times, state._dash_ela_vals,
                   color="#ff7b72", linewidth=2, alpha=0.8)

    # Volume on right axis
    ax_ts2.set_ylabel("Volume (km³)", color="#7ee787", fontsize=14)
    ax_ts2.tick_params(axis="y", colors="#7ee787", labelsize=11)
    ax_ts2.yaxis.set_label_position("right")
    ax_ts2.yaxis.tick_right()
    for sp in ax_ts2.spines.values():
        sp.set_color("#30363d")
    ax_ts2.plot(state._dash_times, state._dash_vols, color="#7ee787", linewidth=2)

    # info text on the timeline
    area = _ice_area_km2(thk, dx)
    info_str = (f"dt = {dt:.2f}   "
                f"Vol = {vol:.3f} km³   Area = {area:.1f} km²   "
                f"Vmax = {vmax:.0f} m/a   Wall = {elapsed:.0f} s")
    ax_ts.set_title(info_str, color="gray", fontsize=12, fontfamily="monospace", pad=6)

    state._dash_fig.canvas.draw_idle()
    state._dash_fig.canvas.flush_events()

    if cfg.outputs.live_dashboard.save_frames:
        state._dash_fig.savefig(f"dashboard_{int(t):06d}.png", facecolor="#0e1117",
                                bbox_inches="tight", pad_inches=0.1)


def _finalize_2d(cfg, state):
    import matplotlib.pyplot as plt
    if hasattr(state, "_dash_fig"):
        plt.ioff()
        plt.close(state._dash_fig)


# ═══════════════════════════════════════════════════════════════════════════
#  3D MODE  (PyVista, GPU-accelerated)
# ═══════════════════════════════════════════════════════════════════════════

def _build_surface_mesh(x, y, z):
    import pyvista as pv
    X, Y = np.meshgrid(x, y)
    return pv.StructuredGrid(X, Y, np.array(z, dtype=np.float64))


def _init_3d(cfg, state):
    import pyvista as pv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = cfg.outputs.live_dashboard
    x = np.array(state.x)
    y = np.array(state.y)
    topg = np.array(state.topg, dtype=np.float64)
    thk = np.array(state.thk, dtype=np.float64)
    usurf = np.array(state.usurf, dtype=np.float64)

    plotter = pv.Plotter(title="IGM 3D Dashboard",
                         window_size=list(p.window_size))
    plotter.set_background("#0e1117")

    # bedrock
    bed_mesh = _build_surface_mesh(x, y, topg)
    plotter.add_mesh(bed_mesh, color="#5a5a5a", opacity=0.7,
                     name="bedrock", lighting=True, smooth_shading=True)

    # ice surface — use Fortran order for scalar assignment
    ice_z = np.where(thk > 1.0, usurf, np.nan)
    ice_mesh = _build_surface_mesh(x, y, ice_z)
    ice_mesh["speed"] = np.zeros(ice_mesh.n_points)
    plotter.add_mesh(ice_mesh, scalars="speed", cmap="inferno",
                     clim=[0, p.vel_max], name="ice",
                     lighting=True, smooth_shading=True,
                     show_scalar_bar=False)
    # add scalar bar once (won't be touched by updates)
    plotter.add_scalar_bar("Speed (m/a)", color="white",
                           title_font_size=14, label_font_size=12,
                           vertical=False, width=0.4, height=0.06,
                           position_x=0.55, position_y=0.02)

    # centered time title (big)
    plotter.add_text("t = 0 yr", position="upper_edge", font_size=20,
                     color="white", name="time_text")

    # stats overlay
    plotter.add_text("Initializing...", position="upper_left", font_size=10,
                     color="white", name="info_text")

    # inset chart (matplotlib figure embedded via ChartMPL)
    plt.style.use("dark_background")
    inset_fig, inset_ax = plt.subplots(figsize=(3.5, 1.5), dpi=100,
                                        facecolor="#161b22")
    inset_ax.set_facecolor("#161b22")
    t0 = float(cfg.processes.time.start)
    inset_ax.set_xlim(t0, cfg.processes.time.end)
    inset_ax.set_xlabel("Time (yr)", fontsize=8, color="gray")
    inset_ax.set_ylabel("ELA (m)", fontsize=8, color="#ff7b72")
    inset_ax.tick_params(labelsize=7, colors="gray")
    inset_ax.tick_params(axis="y", colors="#ff7b72", labelsize=7)
    # create twin axis once for volume
    inset_ax2 = inset_ax.twinx()
    inset_ax2.set_ylabel("Vol (km³)", fontsize=8, color="#7ee787")
    inset_ax2.tick_params(axis="y", colors="#7ee787", labelsize=7)
    inset_fig.tight_layout(pad=0.5)
    chart = pv.ChartMPL(inset_fig, size=(0.30, 0.25), loc=(0.01, 0.01))
    plotter.add_chart(chart)

    # smart initial camera: position above glacier looking down at an angle
    cx = 0.5 * (x[0] + x[-1])
    cy = 0.5 * (y[0] + y[-1])
    z_min, z_max = float(np.nanmin(topg)), float(np.nanmax(usurf))
    z_range = z_max - z_min
    cam_altitude = z_max + 3.0 * z_range  # 300% above elevation range
    domain_len = max(x[-1] - x[0], y[-1] - y[0])
    # place camera offset to the south, looking north and down
    cam_pos = (cx, cy - 0.6 * domain_len, cam_altitude)
    focal_pt = (cx, cy, 0.5 * (z_min + z_max))
    plotter.camera_position = [cam_pos, focal_pt, (0, 0, 1)]
    plotter.show(interactive_update=True)

    state._dash_plotter = plotter
    state._dash_x = x
    state._dash_y = y
    state._dash_inset_fig = inset_fig
    state._dash_inset_ax = inset_ax
    state._dash_inset_ax2 = inset_ax2
    state._dash_inset_chart = chart

    # ELA data will be populated on first update (smbpar not yet available at init)
    state._dash_has_smbpar = None  # deferred check
    state._dash_ela_times = []
    state._dash_ela_vals = []

    state._dash_times = []
    state._dash_vols = []
    state._dash_save_count = 0
    state._dash_wall_start = clock.time()


def _update_3d(cfg, state):
    if not state.saveresult:
        return

    p = cfg.outputs.live_dashboard
    state._dash_save_count += 1
    if state._dash_save_count % p.update_freq != 0:
        return

    # deferred smbpar check (not available at init time)
    if state._dash_has_smbpar is None:
        state._dash_has_smbpar = hasattr(state, "smbpar")
        if state._dash_has_smbpar:
            state._dash_ela_times = state.smbpar[:, 0].tolist()
            state._dash_ela_vals = state.smbpar[:, 3].tolist()

    t = float(state.t.numpy()) if hasattr(state.t, "numpy") else float(state.t)
    thk = np.array(state.thk)
    usurf = np.array(state.usurf, dtype=np.float64)
    dx = float(state.dx)
    ubar = np.array(state.ubar)
    vbar = np.array(state.vbar)
    vel = np.sqrt(ubar**2 + vbar**2)

    plotter = state._dash_plotter
    x = state._dash_x
    y = state._dash_y

    # rebuild ice mesh each frame (scalar bar added separately at init)
    ice_z = np.where(thk > 1.0, usurf, np.nan)
    ice_mesh = _build_surface_mesh(x, y, ice_z)
    ice_mesh["speed"] = np.where(thk.ravel(order='F') > 1.0,
                                  vel.ravel(order='F'), 0.0)
    plotter.add_mesh(ice_mesh, scalars="speed", cmap="inferno",
                     clim=[0, p.vel_max], name="ice",
                     lighting=True, smooth_shading=True,
                     show_scalar_bar=False)

    # stats
    vol = _ice_volume_km3(thk, dx)
    area = _ice_area_km2(thk, dx)
    vmax = float(np.max(vel[thk > 0])) if np.any(thk > 0) else 0.0
    elapsed = clock.time() - state._dash_wall_start
    dt = float(state.dt.numpy()) if hasattr(state.dt, "numpy") else float(state.dt)

    state._dash_times.append(t)
    state._dash_vols.append(vol)

    # live ELA if no smbpar
    if not state._dash_has_smbpar and hasattr(state, "smb"):
        ela = _live_ela(thk, state.smb, state.usurf)
        state._dash_ela_times.append(t)
        state._dash_ela_vals.append(ela)

    # centered time title
    plotter.add_text(f"t = {t:.0f} yr", position="upper_edge", font_size=20,
                     color="white", name="time_text")

    # stats text
    info = (
        f"Vol = {vol:.3f} km³   Area = {area:.1f} km²\n"
        f"Vmax = {vmax:.0f} m/a   dt = {dt:.2f}\n"
        f"Wall = {elapsed:.0f} s"
    )
    plotter.add_text(info, position="upper_left", font_size=10,
                     color="white", name="info_text")

    # update inset chart — clear both axes, reuse twin
    ax = state._dash_inset_ax
    ax2 = state._dash_inset_ax2
    ax.cla()
    ax2.cla()
    ax.set_facecolor("#161b22")
    ax.set_xlim(state._dash_times[0], cfg.processes.time.end)
    ax.set_xlabel("Time (yr)", fontsize=8, color="gray")
    ax.tick_params(labelsize=7, colors="gray")

    # ELA on left axis
    ax.set_ylabel("ELA (m)", fontsize=8, color="#ff7b72")
    ax.tick_params(axis="y", colors="#ff7b72", labelsize=7)
    if state._dash_has_smbpar:
        ax.plot(state._dash_ela_times, state._dash_ela_vals,
                color="#ff7b72", linewidth=1.5, linestyle="--", alpha=0.8)
    elif len(state._dash_ela_vals) > 0:
        ax.plot(state._dash_ela_times, state._dash_ela_vals,
                color="#ff7b72", linewidth=1.5, alpha=0.8)

    # Volume on right axis (reuse existing twin)
    ax2.set_ylabel("Vol (km³)", fontsize=8, color="#7ee787")
    ax2.tick_params(axis="y", colors="#7ee787", labelsize=7)
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax2.plot(state._dash_times, state._dash_vols, color="#7ee787", linewidth=1.5)

    state._dash_inset_fig.tight_layout(pad=0.5)
    state._dash_inset_chart.Modified()

    plotter.update()


def _finalize_3d(cfg, state):
    if hasattr(state, "_dash_plotter"):
        state._dash_plotter.close()


# ═══════════════════════════════════════════════════════════════════════════
#  Module interface (dispatches to 2d or 3d)
# ═══════════════════════════════════════════════════════════════════════════

def initialize(cfg, state):
    state._dash_mode = cfg.outputs.live_dashboard.mode
    if state._dash_mode == "3d":
        _init_3d(cfg, state)
    else:
        _init_2d(cfg, state)


def run(cfg, state):
    if state._dash_mode == "3d":
        _update_3d(cfg, state)
    else:
        _update_2d(cfg, state)
