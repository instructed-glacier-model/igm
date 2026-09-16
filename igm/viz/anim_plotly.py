#!/usr/bin/env python3
"""
IGM glacier visualizer — interactive Plotly/Dash 3-D animation.

Usage:
    igm_viz
    igm_viz --output_folder path/to/outputs

With no arguments, igm_viz looks for netCDF output anywhere under "outputs/"
in the current directory (the default igm_run layout is "outputs/<date>/
<time>/"; a custom hydra.run.dir that skips the date/time nesting, e.g.
"outputs/my_run/", is also picked up). --output_folder points it at a
different top-level "outputs" folder instead. Either way, every run found
under it is listed in a dropdown in the app, and the exact netCDF file to
view is always chosen there.
"""

import argparse, copy, glob, os, threading, webbrowser
import numpy as np, xarray as xr
import plotly.graph_objects as go
from dash import Dash, dcc, html, no_update, Input, Output, State

GOOGLE_FONT = "https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap"
FONT_FAMILY = "'Inter', sans-serif"

# ── colour palettes ───────────────────────────────────────────────────────────

BEDROCK_CS = [
    [0.00, "#d6c4a0"],
    [0.20, "#b8a882"],
    [0.45, "#9a9a96"],
    [0.70, "#6b5030"],
    [1.00, "#3d2b18"],
]

OCEAN_CS = [[0, "rgb(18,90,160)"], [1, "rgb(40,130,210)"]]  # solid ocean blue

CALVING_COLOR = "rgb(160,215,245)"  # ice-blue walls

BASE_COLOR = BEDROCK_CS[0][1]  # flat plate under the terrain, matching the low
# end of the bedrock colorbar so it reads as a continuation of the bedrock
# rather than a different material.

SLIDER_ACCENT = "#1a6faf"  # highlighted track/handle color for every dcc.Slider

# Colorscales cycled across any output.nc variable that isn't one of the
# curated physical quantities below (thickness, velocity, smb, sliding).
EXTRA_COLORSCALES = [
    "Viridis",
    "Cividis",
    "Turbo",
    "YlOrBr",
    "PuBuGn",
    "YlGnBu",
    "OrRd",
    "Greens",
]

# Variables used to build the 3-D geometry itself (bedrock, ice surface,
# grid coordinates) or otherwise not offered as something to color the
# glacier by (uvelsurf/vvelsurf: signed x/y velocity components, redundant
# with the velocity magnitude already offered under "velocity"; icemask:
# a 0/1 mask, not a physical quantity worth its own colorbar).
_GEOMETRY_VARS = {
    "topg",
    "usurf",
    "thk",
    "smb",
    "velsurf_mag",
    "velbar_mag",
    "slidingco",
    "tau_ref",
    "uvelsurf",
    "vvelsurf",
    "icemask",
    "x",
    "y",
    "z",
    "time",
    "dx",
    "dy",
    "X",
    "Y",
    "dX",
    "dY",
}

# Colorscale overrides for extras whose name alone should decide their
# color: any velocity field matches the curated velocity/log_velocity
# colorscale, and thkobs should read like thk since both share its scale.
_COLORSCALE_OVERRIDES = {"thkobs": "Blues", "divflux": "RdBu"}


# Fallback (long_name, units) for variables that carry no long_name/units
# netCDF attributes of their own — data-assimilation outputs (optimize.nc,
# geology-optimized.nc) write plain arrays with no attrs at all. Sourced
# from igm/outputs/local.py's var_info_ncdf_ex and the sliding-law physics
# code (igm/processes/iceflow/energy/components/sliding/laws/_power_law.py),
# which document tau_ref/slidingco as a basal shear stress in MPa.
_KNOWN_UNITS = {
    "tau_ref": ("reference basal shear stress", "MPa"),
    "slidingco": ("reference basal shear stress", "MPa"),
    "velbase_mag": ("basal velocity magnitude", "m a⁻¹"),
    "velsurf_mag": ("surface velocity magnitude", "m a⁻¹"),
    "velsurfobs_mag": ("observed surface velocity magnitude", "m a⁻¹"),
    "velbar_mag": ("depth-average velocity magnitude", "m a⁻¹"),
    "divflux": ("flux divergence", "m a⁻¹"),
    "thk": ("ice thickness", "m"),
    "thkobs": ("observed ice thickness", "m"),
    "usurf": ("surface elevation", "m"),
    "topg": ("bedrock elevation", "m"),
}


def build_property_catalog(ds: xr.Dataset) -> dict:
    """Build the "property" dropdown for this dataset: the curated physical
    quantities (only when their source variable is actually present) plus
    one entry for every other 2-D+time variable found in the output.nc, so
    extra saved fields (divflux, arrhenius, ...) are browsable without
    editing this file. Maps dropdown label -> (internal_key, axis_label,
    colorscale).
    """
    catalog = {
        "thickness (m)": ("thickness", "thickness (m)", "Blues"),
    }

    if "velsurf_mag" in ds or "velbar_mag" in ds:
        catalog["velocity (m a\u207b\u00b9)"] = (
            "velocity",
            "velocity (m a\u207b\u00b9)",
            "magma",
        )
        catalog["log velocity (m a\u207b\u00b9)"] = (
            "log_velocity",
            "log\u2081\u2080 velocity (m a\u207b\u00b9)",
            "magma",
        )
    if "smb" in ds:
        catalog["SMB (m a\u207b\u00b9)"] = ("smb", "SMB (m a\u207b\u00b9)", "RdBu")
    if "slidingco" in ds or "tau_ref" in ds:
        catalog["sliding coeff. (MPa)"] = ("sliding", "sliding coeff. (MPa)", "plasma")

    extras = sorted(
        var
        for var in ds.data_vars
        if var not in _GEOMETRY_VARS and {"y", "x"} <= set(ds[var].dims)
    )
    next_cycled = 0
    for var in extras:
        long_name = ds[var].attrs.get("long_name")
        units = ds[var].attrs.get("units")
        if long_name is None and units is None and var in _KNOWN_UNITS:
            long_name, units = _KNOWN_UNITS[var]
        long_name = long_name or var
        units = units or ""
        label = f"{long_name} ({units})" if units else long_name

        if "vel" in var.lower():
            colorscale = "magma"
        elif var in _COLORSCALE_OVERRIDES:
            colorscale = _COLORSCALE_OVERRIDES[var]
        else:
            colorscale = EXTRA_COLORSCALES[next_cycled % len(EXTRA_COLORSCALES)]
            next_cycled += 1
        catalog[label] = (var, label, colorscale)

    # Prefix every dropdown entry with a map emoji; the axis/colorbar label
    # stored alongside it (the tuple's second element) is left unprefixed.
    return {f"\U0001f5fa️ {label}": value for label, value in catalog.items()}


# ── data helpers ──────────────────────────────────────────────────────────────


def discover_output_tree(outputs_dir: str) -> dict:
    """Find every netCDF file anywhere under <outputs_dir>, grouped
    date -> time -> [{"label": filename, "value": path}, ...].

    Matches the run-directory layout Hydra creates for igm_run by default
    (see igm/conf/config.yaml, which leaves hydra.run.dir at
    "outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}"): the file's two innermost
    parent directories become "date" and "time". A custom hydra.run.dir
    with fewer path segments below <outputs_dir> (e.g. "outputs/my_run",
    landing files directly in outputs/my_run/) still works: whichever of
    "date"/"time" has no real directory to take its name from falls back to
    "" (rendered as "(root)" in the UI), so there's always a two-level tree
    to feed the "model run" picker even when the run itself isn't nested
    two folders deep.

    Every kind of output a run can drop there (output.nc from a forward
    run, optimize.nc from data assimilation, ...) is picked up so the UI
    can browse it like a small file explorer. Skips "*_ts.nc" companions:
    scalar time series only, not the gridded fields this viewer plots.
    """
    pattern = os.path.join(outputs_dir, "**", "*.nc")
    paths = sorted(
        p for p in glob.glob(pattern, recursive=True) if not p.endswith("_ts.nc")
    )
    tree = {}
    for path in paths:
        rel_dir = os.path.relpath(os.path.dirname(path), outputs_dir)
        parts = [] if rel_dir == "." else rel_dir.split(os.sep)
        if len(parts) >= 2:
            date_dir, time_dir = parts[-2], parts[-1]
        elif len(parts) == 1:
            date_dir, time_dir = "", parts[0]
        else:
            date_dir, time_dir = "", ""
        tree.setdefault(date_dir, {}).setdefault(time_dir, []).append(
            {"label": os.path.basename(path), "value": path}
        )
    return tree


def _dir_label(name: str, folder_emoji: str, root_name: str) -> str:
    """Dropdown label for a date/time directory: the "(root)" fallback
    (name == "", used when a run isn't nested two folders deep) is labeled
    with `root_name` (the outputs folder's own name) and gets its own emoji
    distinct from a real folder's."""
    return f"{folder_emoji} {name}" if name else f"🏠 {root_name} (root)"


def _pick_default_file(
    tree: dict, preferred_filename: str | None = None
) -> tuple[str, str, str]:
    """The latest date, latest time, preferring `preferred_filename` among
    that time folder's files if present, else the first file there.
    Returns (date, time, path)."""
    date = sorted(tree)[-1]
    time = sorted(tree[date])[-1]
    files = tree[date][time]
    if preferred_filename:
        for f in files:
            if f["label"] == preferred_filename:
                return date, time, f["value"]
    return date, time, files[0]["value"]


_NETCDF_LOCK = threading.Lock()  # netCDF4's HDF5 backend is not thread-safe;
# Dash's dev server can run several callbacks (property, run selector,
# clamp range) on separate threads for one interaction, and concurrent
# calls into the same file can segfault the process, not just raise.


def load_ds(path: str) -> xr.Dataset:
    # Every callback below re-opens this on each interaction; close the file
    # handle immediately (loading data into memory) instead of leaving it
    # open, or repeated calls exhaust HDF5's file locking and start
    # throwing "Can't open HDF5 attribute" on unrelated later opens.
    with _NETCDF_LOCK, xr.open_dataset(path, engine="netcdf4") as f:
        ds = f.load()

    # Data-assimilation output (optimize.nc) has no bedrock variable since
    # it isn't part of the inversion; derive it the same way igm's own
    # complete_data() does for forward-model inputs lacking topg.
    if "topg" not in ds and {"usurf", "thk"} <= set(ds.data_vars):
        ds["topg"] = ds.usurf - ds.thk

    # A single static snapshot (e.g. geology-optimized.nc) has no time or
    # iterations axis at all; treat it as one frame so the rest of the
    # pipeline, which always indexes a leading frame dimension, needs no
    # separate code path for it.
    if "time" not in ds.dims and "iterations" not in ds.dims:
        ds = ds.expand_dims(iterations=[0])

    return ds


def frame_dim_of(ds: xr.Dataset) -> str:
    """The dimension IGM outputs step/animate over: "time" for forward-model
    runs (output.nc), "iterations" for data-assimilation runs (optimize.nc)."""
    return "time" if "time" in ds.dims else "iterations"


def extract_property(ds, key, label, colorscale, vmin_clamp=None, vmax_clamp=None):
    if key == "velocity":
        arr = np.array(ds.velsurf_mag if "velsurf_mag" in ds else ds.velbar_mag)
    elif key == "log_velocity":
        raw = np.array(ds.velsurf_mag if "velsurf_mag" in ds else ds.velbar_mag)
        arr = np.log10(np.clip(raw, 1e-3, None))
    elif key == "sliding":
        arr = np.array(ds.slidingco if "slidingco" in ds else ds.tau_ref)
    elif key == "thickness":
        arr = np.array(ds.thk)
    else:
        arr = np.array(ds[key])

    vmin = float(vmin_clamp) if vmin_clamp is not None else float(np.nanmin(arr))
    vmax = float(vmax_clamp) if vmax_clamp is not None else float(np.nanmax(arr))
    if key in ("smb", "divflux") and vmin_clamp is None:
        # Both are signed (accumulation/loss) fields shown on a diverging
        # colorscale; center it on zero so white actually means "no change"
        # instead of landing wherever the data's raw min/max happens to be.
        mx = max(abs(vmin), abs(vmax))
        vmin, vmax = -mx, mx

    return arr, colorscale, label, vmin, vmax


def ice_stats(ds):
    thk = np.array(ds.thk)
    cell = float(ds.x[1] - ds.x[0]) * float(ds.y[1] - ds.y[0])
    return (
        np.array(ds[frame_dim_of(ds)]),
        thk.sum(axis=(1, 2)) * cell / 1e9,
        (thk > 1).sum(axis=(1, 2)) * cell / 1e6,
    )


def property_range(ds, key, label, colorscale):
    _, _, _, vmin, vmax = extract_property(ds, key, label, colorscale)
    return round(vmin, 3), round(vmax, 3)


# ── plotly trace builders ─────────────────────────────────────────────────────


def fmt2d(arr, fmt=".3g"):
    """2-D numpy array → 2-D list of formatted strings (for Surface hover text)."""
    return [[f"{v:{fmt}}" for v in row] for row in arr]


def base_trace(bedrock, x, y) -> go.Surface:
    """Flat opaque plate beneath the whole terrain block, closing it off
    from below so it reads as a solid slab rather than a floating sheet.
    Sits exactly at the bedrock's lowest point so there's no visible gap
    between the bedrock surface and this plate."""
    z_base = float(bedrock.min())
    z = np.full_like(bedrock, z_base)
    return go.Surface(
        z=z,
        x=x,
        y=y,
        colorscale=[[0, BASE_COLOR], [1, BASE_COLOR]],
        showscale=False,
        opacity=1.0,
        name="base",
        showlegend=False,
        hoverinfo="skip",
    )


def bedrock_trace(bedrock, x, y) -> go.Surface:
    border = copy.copy(bedrock)
    min_z = float(bedrock.min())
    border[[0, -1], :] = min_z
    border[:, [0, -1]] = min_z
    return go.Surface(
        z=border,
        x=x,
        y=y,
        colorscale=BEDROCK_CS,
        opacity=1.0,
        cmin=min_z,
        cmax=float(bedrock.max()),
        colorbar=dict(
            x=-0.09,
            len=0.55,
            thickness=12,
            title=dict(text="elevation (m)", side="right", font=dict(size=11)),
            tickfont_size=10,
        ),
        name="bedrock",
        showlegend=True,
        hovertemplate="elevation: %{z:.0f} m<extra>bedrock</extra>",
    )


def ocean_trace(bedrock, thk, x, y) -> go.Surface:
    """Solid ocean plane at z = 0, only where bedrock < 0 and no ice."""
    z = np.where((bedrock < 0) & (thk < 1), 0.0, np.nan)
    return go.Surface(
        z=z,
        x=x,
        y=y,
        colorscale=OCEAN_CS,
        showscale=False,
        opacity=1.0,
        name="ocean (0 m)",
        showlegend=True,
        hovertemplate="sea level: 0 m<extra>ocean</extra>",
    )


def calving_front_trace(bedrock, thk, surf, x, y):
    """
    Vertical ice walls at the calving front: edges where glacier (thk > 1)
    is directly adjacent to open ocean (bedrock < 0, thk < 1).
    Each wall panel spans from z = 0 (sea level) up to the ice surface.
    Returns a go.Mesh3d, or None if no calving front exists.
    """
    ice = thk > 1
    ocean = (bedrock < 0) & ~ice
    if not ocean.any() or not ice.any():
        return None

    ny, nx = bedrock.shape
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])

    vx, vy, vz = [], [], []
    ti, tj, tk = [], [], []

    def add_quad(p0, p1, p2, p3):
        """Append a quad (p0–p3 are (x,y,z)) as two triangles."""
        b = len(vx)
        for p in (p0, p1, p2, p3):
            vx.append(p[0])
            vy.append(p[1])
            vz.append(p[2])
        ti.extend([b, b])
        tj.extend([b + 1, b + 2])
        tk.extend([b + 2, b + 3])

    for i, j in zip(*np.where(ice)):
        zt = float(surf[i, j])
        zb = 0.0  # sea level
        xi, yi = float(x[j]), float(y[i])

        if j + 1 < nx and ocean[i, j + 1]:  # +x face
            xw = xi + dx / 2
            add_quad(
                (xw, yi - dy / 2, zb),
                (xw, yi + dy / 2, zb),
                (xw, yi + dy / 2, zt),
                (xw, yi - dy / 2, zt),
            )
        if j - 1 >= 0 and ocean[i, j - 1]:  # -x face
            xw = xi - dx / 2
            add_quad(
                (xw, yi - dy / 2, zb),
                (xw, yi + dy / 2, zb),
                (xw, yi + dy / 2, zt),
                (xw, yi - dy / 2, zt),
            )
        if i + 1 < ny and ocean[i + 1, j]:  # +y face
            yw = yi + dy / 2
            add_quad(
                (xi - dx / 2, yw, zb),
                (xi + dx / 2, yw, zb),
                (xi + dx / 2, yw, zt),
                (xi - dx / 2, yw, zt),
            )
        if i - 1 >= 0 and ocean[i - 1, j]:  # -y face
            yw = yi - dy / 2
            add_quad(
                (xi - dx / 2, yw, zb),
                (xi + dx / 2, yw, zb),
                (xi + dx / 2, yw, zt),
                (xi - dx / 2, yw, zt),
            )

    if not vx:
        return None

    return go.Mesh3d(
        x=vx,
        y=vy,
        z=vz,
        i=ti,
        j=tj,
        k=tk,
        color=CALVING_COLOR,
        opacity=1.0,
        name="calving front",
        showlegend=True,
        hoverinfo="skip",
    )


def glacier_traces(surf, bottom, x, y, prop, cs, lbl, vmin, vmax, opacity) -> list:
    prop_txt = fmt2d(prop)
    shared = dict(
        x=x,
        y=y,
        colorscale=cs,
        cmin=vmin,
        cmax=vmax,
        opacity=opacity,
        surfacecolor=prop,
        text=prop_txt,
        hovertemplate=lbl + ": %{text}<extra>%{fullData.name}</extra>",
    )
    top = go.Surface(
        z=surf,
        **shared,
        colorbar=dict(
            x=1.04,
            len=0.55,
            thickness=12,
            title=dict(text=lbl, side="right", font=dict(size=11)),
            tickfont_size=10,
        ),
        name="glacier surface",
        showlegend=True,
    )
    bot = go.Surface(
        z=bottom, **shared, name="ice bottom", showlegend=True, showscale=False
    )
    return [top, bot]


def make_frame(
    i,
    frame_value,
    bedrock,
    thk_arr,
    surf_arr,
    prop_arr,
    cs,
    lbl,
    vmin,
    vmax,
    x,
    y,
    opacity,
    show_ocean,
    show_calving,
):
    rho_i_over_rhow = 918.0 / 1028.0
    thk = thk_arr[i]
    surf = np.where(thk < 1, np.nan, surf_arr[i])
    bottom = np.where(thk < 1, np.nan, np.maximum(bedrock, -rho_i_over_rhow * thk))
    # bottom = np.where(thk < 1, np.nan, bedrock)

    traces = [base_trace(bedrock, x, y), bedrock_trace(bedrock, x, y)]
    if show_ocean:
        traces.append(ocean_trace(bedrock, thk, x, y))
    if show_calving:
        cf = calving_front_trace(bedrock, thk, surf_arr[i], x, y)
        if cf is not None:
            traces.append(cf)
    traces += glacier_traces(
        surf, bottom, x, y, prop_arr[i], cs, lbl, vmin, vmax, opacity
    )
    return {"data": traces, "name": int(frame_value)}


# ── 3-D figure ────────────────────────────────────────────────────────────────


def build_3d_figure(
    ds,
    prop_key,
    prop_label,
    prop_colorscale,
    z_exag,
    opacity,
    show_ocean,
    show_calving,
    vmin_clamp,
    vmax_clamp,
    camera=None,
):
    # bedrock_arr varies per frame: for output.nc, topg is genuinely
    # constant across time so every slice is identical; for optimize.nc
    # (no topg saved), load_ds() derives topg = usurf - thk, which does
    # change from one optimization iteration to the next as thk is refined.
    bedrock_arr = np.array(ds.topg)
    x, y = np.array(ds.x), np.array(ds.y)
    frame_dim = frame_dim_of(ds)
    frame_vals = np.array(ds[frame_dim])
    frame_prefix = "year: " if frame_dim == "time" else "iteration: "
    thk_arr = np.array(ds.thk)
    surf_arr = np.array(ds.usurf)
    prop_arr, cs, lbl, vmin, vmax = extract_property(
        ds, prop_key, prop_label, prop_colorscale, vmin_clamp, vmax_clamp
    )

    frames = [
        make_frame(
            i,
            val,
            bedrock_arr[i],
            thk_arr,
            surf_arr,
            prop_arr,
            cs,
            lbl,
            vmin,
            vmax,
            x,
            y,
            opacity,
            show_ocean,
            show_calving,
        )
        for i, val in enumerate(frame_vals)
    ]

    steps = [
        {
            "args": [[int(val)], {"frame": {"duration": 0, "redraw": True}}],
            "label": str(int(val)),
            "method": "animate",
        }
        for val in frame_vals
    ]

    res = float(x[1] - x[0])
    ratio_y = bedrock_arr.shape[1] / bedrock_arr.shape[2]
    # Use the elevation range across every frame (not just one) so the
    # vertical scale doesn't jump as you step between frames.
    ratio_z = (
        (float(bedrock_arr.max()) - float(bedrock_arr.min()))
        / (bedrock_arr.shape[1] * res)
        * z_exag
    )

    layout = go.Layout(
        height=760,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_FAMILY, size=12),
        uirevision="static",
        showlegend=False,
        scene=dict(
            xaxis=dict(
                showbackground=False,
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                showspikes=False,
                title="",
            ),
            yaxis=dict(
                showbackground=False,
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                showspikes=False,
                title="",
            ),
            zaxis=dict(
                showbackground=False,
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                showspikes=False,
                title="",
            ),
            bgcolor="rgba(0,0,0,0)",
            **({"camera": camera} if camera else {}),
        ),
        scene_aspectratio=dict(x=1, y=ratio_y, z=ratio_z),
        sliders=[
            dict(
                active=0,
                currentvalue=dict(
                    font=dict(size=14, family=FONT_FAMILY),
                    prefix=frame_prefix,
                    xanchor="right",
                ),
                transition=dict(duration=0),
                pad=dict(b=10, t=50),
                len=0.88,
                x=0.1,
                y=0,
                steps=steps,
            )
        ],
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                pad=dict(r=10, t=87),
                showactive=False,
                x=0.09,
                xanchor="right",
                y=0,
                yanchor="top",
                buttons=[
                    dict(
                        label="\u25b6  play",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 250, "redraw": True},
                                "fromcurrent": True,
                            },
                        ],
                    ),
                    dict(
                        label="\u23f8 pause",
                        method="animate",
                        args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}],
                    ),
                ],
            )
        ],
    )
    return go.Figure(data=frames[0]["data"], frames=frames, layout=layout)


# ── statistics panel ──────────────────────────────────────────────────────────


def build_stats_figure(ds) -> go.Figure:
    time, vol, area = ice_stats(ds)
    frame_label = "year" if frame_dim_of(ds) == "time" else "iteration"
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=time,
            y=vol,
            name="volume (km\u00b3)",
            mode="lines+markers",
            line=dict(color="#1a6faf", width=2.5),
            marker=dict(size=5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=time,
            y=area,
            name="area (km\u00b2)",
            mode="lines+markers",
            line=dict(color="#b03a2e", width=2.5, dash="dot"),
            marker=dict(size=5),
            yaxis="y2",
        )
    )
    fig.update_layout(
        height=210,
        margin=dict(l=60, r=70, t=15, b=90),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(220,232,248,0.4)",
        font=dict(family=FONT_FAMILY, size=11),
        xaxis=dict(
            title=frame_label, showgrid=True, gridcolor="#c8d8e8", zeroline=False
        ),
        yaxis=dict(
            title="volume (km\u00b3)",
            title_font_color="#1a6faf",
            showgrid=True,
            gridcolor="#c8d8e8",
            zeroline=False,
        ),
        yaxis2=dict(
            title="area (km\u00b2)",
            title_font_color="#b03a2e",
            overlaying="y",
            side="right",
            showgrid=False,
            zeroline=False,
        ),
        legend=dict(
            orientation="h",
            y=-0.55,
            x=0.5,
            xanchor="center",
            bgcolor="rgba(255,255,255,0.75)",
            bordercolor="#bbb",
            borderwidth=1,
        ),
    )
    return fig


# ── Dash app ──────────────────────────────────────────────────────────────────


def finalize(
    tree: dict,
    default_date: str,
    default_time: str,
    default_run: str,
    title_base: str,
    root_name: str,
):
    ds = load_ds(default_run)

    # tree is date -> time -> [{"label": filename, "value": path}, ...],
    # mirroring outputs/<date>/<time>/<file>.nc so the "model run" picker
    # behaves like a small file browser: folder, then folder, then file.
    # date/time fall back to "" (shown as "<root_name> (root)") when the run
    # folder isn't nested two levels deep under outputs_dir.
    date_options = [
        {"label": _dir_label(d, "📁", root_name), "value": d} for d in sorted(tree)
    ]
    time_options = [
        {"label": _dir_label(t, "📂", root_name), "value": t}
        for t in sorted(tree[default_date])
    ]
    file_options = [
        {"label": f"📄 {f['label']}", "value": f["value"]}
        for f in tree[default_date][default_time]
    ]

    catalog = build_property_catalog(ds)  # label -> (key, axis_label, colorscale)
    RANGES = {
        key: property_range(ds, key, axis_label, cs)
        for key, axis_label, cs in catalog.values()
    }

    app = Dash(__name__, external_stylesheets=[GOOGLE_FONT])
    app.title = title_base

    # dash-core-components sliders are the rc-slider library under the hood;
    # style its highlighted track/handle directly since dcc.Slider has no
    # color prop of its own.
    app.index_string = f"""<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{{%title%}}</title>
        {{%favicon%}}
        {{%css%}}
        <style>
            .rc-slider-track {{ background-color: {SLIDER_ACCENT} !important; }}
            .rc-slider-handle {{ border-color: {SLIDER_ACCENT} !important; }}
            .rc-slider-handle:hover,
            .rc-slider-handle:focus,
            .rc-slider-handle-dragging {{
                border-color: {SLIDER_ACCENT} !important;
                box-shadow: 0 0 0 5px rgba(26, 111, 175, 0.2) !important;
            }}
            .rc-slider-dot-active {{ border-color: {SLIDER_ACCENT} !important; }}
        </style>
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>
"""

    CTRL = {"flex": "1 1 200px"}
    BAR = {
        "display": "flex",
        "gap": "28px",
        "alignItems": "flex-end",
        "padding": "14px 24px",
        "background": "#eef2f7",
        "borderBottom": "1px solid #c4cdd8",
        "fontFamily": FONT_FAMILY,
    }
    LABEL = {
        "fontWeight": "600",
        "fontSize": "12px",
        "color": "#444",
        "marginBottom": "4px",
        "display": "block",
    }

    lo0, hi0 = RANGES["thickness"]

    app.layout = html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("model run", style=LABEL),
                            html.Div(
                                [
                                    dcc.Dropdown(
                                        date_options,
                                        default_date,
                                        id="date_selector",
                                        clearable=False,
                                        searchable=False,
                                        style={"minWidth": "160px"},
                                    ),
                                    html.Span(
                                        "/",
                                        style={
                                            "margin": "0 8px",
                                            "color": "#888",
                                            "fontWeight": "600",
                                        },
                                    ),
                                    dcc.Dropdown(
                                        time_options,
                                        default_time,
                                        id="time_selector",
                                        clearable=False,
                                        searchable=False,
                                        style={"minWidth": "140px"},
                                    ),
                                    html.Span(
                                        "/",
                                        style={
                                            "margin": "0 8px",
                                            "color": "#888",
                                            "fontWeight": "600",
                                        },
                                    ),
                                    dcc.Dropdown(
                                        file_options,
                                        default_run,
                                        id="file_selector",
                                        clearable=False,
                                        searchable=False,
                                        style={"minWidth": "160px", "flex": "1"},
                                    ),
                                ],
                                style={"display": "flex", "alignItems": "center"},
                            ),
                        ],
                        style={"flex": "1 1 auto"},
                    ),
                    html.Div(
                        [
                            html.Label("property", style=LABEL),
                            dcc.Dropdown(
                                list(catalog),
                                list(catalog)[0],
                                id="property",
                                clearable=False,
                                searchable=False,
                                style={"minWidth": "220px"},
                            ),
                        ],
                        style={"flex": "0 0 240px", "marginLeft": "24px"},
                    ),
                ],
                style={
                    "display": "flex",
                    "alignItems": "flex-end",
                    "padding": "10px 24px 4px",
                    "background": "#eef2f7",
                    "borderBottom": "1px solid #c4cdd8",
                    "fontFamily": FONT_FAMILY,
                },
            ),
            html.Details(
                [
                    html.Summary(
                        "display settings",
                        style={
                            "cursor": "pointer",
                            "fontWeight": "600",
                            "fontSize": "12px",
                            "color": "#444",
                            "padding": "10px 24px",
                            "background": "#eef2f7",
                            "userSelect": "none",
                        },
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("vertical exaggeration", style=LABEL),
                                    dcc.Slider(
                                        1,
                                        20,
                                        0.5,
                                        value=2,
                                        id="z_exag",
                                        marks={i: str(i) for i in range(1, 21, 4)},
                                        tooltip={
                                            "placement": "bottom",
                                            "always_visible": False,
                                        },
                                    ),
                                ],
                                style={**CTRL, "flex": "2 1 260px"},
                            ),
                            html.Div(
                                [
                                    html.Label("glacier opacity", style=LABEL),
                                    dcc.Slider(
                                        0.1,
                                        1.0,
                                        0.05,
                                        value=1.0,
                                        id="opacity",
                                        marks={
                                            v: f"{int(v*100)}%"
                                            for v in (0.25, 0.5, 0.75, 1.0)
                                        },
                                        tooltip={
                                            "placement": "bottom",
                                            "always_visible": False,
                                        },
                                    ),
                                ],
                                style={**CTRL, "flex": "2 1 260px"},
                            ),
                            html.Div(
                                [
                                    dcc.Checklist(
                                        id="show_ocean",
                                        options=[
                                            {
                                                "label": "  ocean (z = 0)",
                                                "value": "ocean",
                                            }
                                        ],
                                        value=[],
                                        style={"marginBottom": "8px"},
                                    ),
                                    dcc.Checklist(
                                        id="show_calving",
                                        options=[
                                            {
                                                "label": "  calving front",
                                                "value": "calving",
                                            }
                                        ],
                                        value=[],
                                    ),
                                ],
                                style={"flex": "0 0 170px", "paddingTop": "22px"},
                            ),
                        ],
                        style=BAR,
                    ),
                    html.Div(
                        [
                            html.Label(
                                "colorbar range",
                                style={
                                    **LABEL,
                                    "whiteSpace": "nowrap",
                                    "marginRight": "16px",
                                    "marginBottom": 0,
                                },
                            ),
                            html.Div(
                                dcc.RangeSlider(
                                    id="clamp_range",
                                    min=lo0,
                                    max=hi0,
                                    step=(hi0 - lo0) / 200,
                                    value=[lo0, hi0],
                                    allowCross=False,
                                    tooltip={
                                        "placement": "bottom",
                                        "always_visible": False,
                                    },
                                ),
                                style={"flex": "1"},
                            ),
                        ],
                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "padding": "10px 24px 4px",
                            "background": "#f5f7fb",
                            "borderBottom": "1px solid #c4cdd8",
                            "fontFamily": FONT_FAMILY,
                        },
                    ),
                ],
                open=False,
                style={"borderBottom": "1px solid #c4cdd8"},
            ),
            dcc.Store(id="camera_store"),
            dcc.Graph(id="surface_3d", config={"scrollZoom": True}),
            html.Div(
                [dcc.Graph(id="stats_chart", figure=build_stats_figure(ds))],
                style={"borderTop": "1px solid #c4cdd8"},
            ),
        ],
        style={"fontFamily": FONT_FAMILY, "fontSize": "13px", "background": "#f7f9fc"},
    )

    @app.callback(
        Output("time_selector", "options"),
        Output("time_selector", "value"),
        Input("date_selector", "value"),
    )
    def update_time_options(date):
        times = sorted(tree[date])
        options = [{"label": _dir_label(t, "📂", root_name), "value": t} for t in times]
        return options, times[-1]  # most recent time in that date

    @app.callback(
        Output("file_selector", "options"),
        Output("file_selector", "value"),
        Input("time_selector", "value"),
        State("date_selector", "value"),
    )
    def update_file_options(time, date):
        files = tree[date][time]
        options = [{"label": f"📄 {f['label']}", "value": f["value"]} for f in files]
        return options, options[0]["value"]

    @app.callback(
        Output("property", "options"),
        Output("property", "value"),
        Input("file_selector", "value"),
        State("property", "value"),
    )
    def update_property_options(run_path, current_prop):
        # Different files (output.nc vs. optimize.nc, say) can save
        # different variables, so the property list has to be rebuilt for
        # whichever file is now selected rather than staying fixed at the
        # one the app started on.
        file_catalog = build_property_catalog(load_ds(run_path))
        options = list(file_catalog)
        value = current_prop if current_prop in file_catalog else options[0]
        return options, value

    @app.callback(
        Output("clamp_range", "min"),
        Output("clamp_range", "max"),
        Output("clamp_range", "step"),
        Output("clamp_range", "value"),
        Output("clamp_range", "marks"),
        Input("property", "value"),
        Input("file_selector", "value"),
    )
    def update_clamp_bounds(prop_label, run_path):
        ds = load_ds(run_path)
        file_catalog = build_property_catalog(ds)
        # property.value hasn't been corrected yet on the same round-trip
        # that switches file_selector to a file missing the current
        # property; fall back to something that does exist so this doesn't
        # crash before update_property_options catches up.
        if prop_label not in file_catalog:
            prop_label = next(iter(file_catalog))
        key, axis_label, cs = file_catalog[prop_label]
        lo, hi = property_range(ds, key, axis_label, cs)
        step = (hi - lo) / 200 if hi != lo else 0.01
        marks = {float(v): f"{v:.3g}" for v in np.linspace(lo, hi, 5)}
        return lo, hi, step, [lo, hi], marks

    @app.callback(
        Output("camera_store", "data"),
        Input("surface_3d", "relayoutData"),
    )
    def store_camera(relayout_data):
        # relayoutData only carries "scene.camera" when the user actually
        # dragged/zoomed the 3-D view; other relayouts (e.g. from a figure
        # rebuild) fire with unrelated or empty payloads, so leave the
        # stored camera untouched unless we see a real camera change.
        if relayout_data and "scene.camera" in relayout_data:
            return relayout_data["scene.camera"]
        return no_update

    @app.callback(
        Output("surface_3d", "figure"),
        Input("file_selector", "value"),
        Input("property", "value"),
        Input("z_exag", "value"),
        Input("opacity", "value"),
        Input("show_ocean", "value"),
        Input("show_calving", "value"),
        Input("clamp_range", "value"),
        State("camera_store", "data"),
    )
    def update_figure(
        run_path,
        prop_label,
        z_exag,
        opacity,
        show_ocean,
        show_calving,
        clamp_range,
        camera,
    ):
        ds = load_ds(run_path)
        file_catalog = build_property_catalog(ds)
        if prop_label not in file_catalog:
            prop_label = next(iter(file_catalog))
        prop_key, prop_axis_label, prop_cs = file_catalog[prop_label]
        ocean = "ocean" in (show_ocean or [])
        calving = "calving" in (show_calving or [])
        vmin, vmax = clamp_range if clamp_range else [None, None]
        return build_3d_figure(
            ds,
            prop_key,
            prop_axis_label,
            prop_cs,
            z_exag,
            opacity,
            ocean,
            calving,
            vmin,
            vmax,
            camera,
        )

    @app.callback(
        Output("stats_chart", "figure"),
        Input("file_selector", "value"),
    )
    def update_stats(run_path):
        return build_stats_figure(load_ds(run_path))

    port = 8050

    app.run(debug=True, host="127.0.0.1", port=port)


# ── CLI ───────────────────────────────────────────────────────────────────────


def _resolve(output_folder: str | None):
    """Find the run tree under the top-level "outputs" folder and pick a
    default run to open. output_folder defaults to "outputs" in the current
    directory (the layout igm_run produces); the exact netCDF file to view
    is always chosen afterward in the dashboard's "model run" picker."""
    outputs_dir = (
        os.path.abspath(output_folder)
        if output_folder
        else os.path.join(os.getcwd(), "outputs")
    )
    tree = discover_output_tree(outputs_dir)
    if not tree:
        raise FileNotFoundError(
            f"No netCDF output found under {outputs_dir}. "
            "Run igm_run first, or pass --output_folder."
        )
    default_date, default_time, default_run = _pick_default_file(tree, "output.nc")
    title_base = os.path.basename(os.path.normpath(os.path.dirname(outputs_dir)))
    root_name = os.path.basename(os.path.normpath(outputs_dir))
    return tree, default_date, default_time, default_run, title_base, root_name


def main():
    parser = argparse.ArgumentParser(
        description="Visualize IGM glacier output — Plotly/Dash 3-D animation."
    )
    parser.add_argument(
        "--output_folder",
        help='Path to the top-level "outputs" folder (default: "outputs" in the current directory)',
    )
    args = parser.parse_args()

    tree, default_date, default_time, default_run, title_base, root_name = _resolve(
        args.output_folder
    )
    finalize(tree, default_date, default_time, default_run, title_base, root_name)


if __name__ == "__main__":
    main()
