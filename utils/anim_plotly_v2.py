#!/usr/bin/env python3
"""
IGM glacier visualizer — interactive Plotly/Dash 3-D animation.

Usage:
    python anim_plotly_v2.py --param_file params.yaml
    python anim_plotly_v2.py --output_file path/to/output.nc
"""

import argparse, copy, os
import numpy as np, xarray as xr, yaml
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

GOOGLE_FONT = "https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap"
FONT_FAMILY = "'Inter', sans-serif"

# ── colour palettes ───────────────────────────────────────────────────────────

COLORSCALES = dict(
    thickness="Blues",
    velocity="Blues",
    log_velocity="Blues",
    smb="RdBu",
    sliding="plasma",
)

BEDROCK_CS = [
    [0.00, "#d6c4a0"],
    [0.20, "#b8a882"],
    [0.45, "#9a9a96"],
    [0.70, "#6b5030"],
    [1.00, "#3d2b18"],
]

OCEAN_CS = [[0, "rgb(18,90,160)"], [1, "rgb(40,130,210)"]]  # solid ocean blue

CALVING_COLOR = "rgb(160,215,245)"  # ice-blue walls

PROPERTY_MAP = {  # dropdown label → (internal key, colorbar label)
    "thickness (m)": ("thickness", "thickness (m)"),
    "velocity (m a\u207b\u00b9)": ("velocity", "velocity (m a\u207b\u00b9)"),
    "log velocity (m a\u207b\u00b9)": (
        "log_velocity",
        "log\u2081\u2080 velocity (m a\u207b\u00b9)",
    ),
    "SMB (m a\u207b\u00b9)": ("smb", "SMB (m a\u207b\u00b9)"),
    "sliding coeff. (units)": ("sliding", "sliding coeff. (units)"),
}

# ── data helpers ──────────────────────────────────────────────────────────────


def get_output_path(params) -> str:
    return getattr(params, "output_file", None) or getattr(
        params, "wncd_output_file", None
    )


def load_ds(path: str) -> xr.Dataset:
    return xr.open_dataset(path, engine="netcdf4")


def extract_property(ds, key, vmin_clamp=None, vmax_clamp=None):
    if key in ("velocity", "log_velocity"):
        raw = np.array(ds.velsurf_mag if "velsurf_mag" in ds else ds.velbar_mag)
        if key == "log_velocity":
            arr = np.log10(np.clip(raw, 1e-3, None))
            lbl = "log\u2081\u2080 velocity (m a\u207b\u00b9)"
        else:
            arr, lbl = raw, "velocity (m a\u207b\u00b9)"
    elif key == "smb" and "smb" in ds:
        arr, lbl = np.array(ds.smb), "SMB (m a\u207b\u00b9)"
    elif key == "sliding" and "slidingco" in ds:
        arr, lbl = np.array(ds.slidingco), "sliding coeff. (units)"
    else:
        arr, lbl, key = np.array(ds.thk), "thickness (m)", "thickness"

    vmin = float(vmin_clamp) if vmin_clamp is not None else float(arr.min())
    vmax = float(vmax_clamp) if vmax_clamp is not None else float(arr.max())
    if key == "smb" and vmin_clamp is None:
        mx = max(abs(vmin), abs(vmax))
        vmin, vmax = -mx, mx

    return arr, COLORSCALES[key], lbl, vmin, vmax


def ice_stats(ds):
    thk = np.array(ds.thk)
    cell = float(ds.x[1] - ds.x[0]) * float(ds.y[1] - ds.y[0])
    return (
        np.array(ds.time),
        thk.sum(axis=(1, 2)) * cell / 1e9,
        (thk > 1).sum(axis=(1, 2)) * cell / 1e6,
    )


def property_range(ds, key):
    arr, _, _, vmin, vmax = extract_property(ds, key)
    return round(vmin, 3), round(vmax, 3)


# ── plotly trace builders ─────────────────────────────────────────────────────


def fmt2d(arr, fmt=".3g"):
    """2-D numpy array → 2-D list of formatted strings (for Surface hover text)."""
    return [[f"{v:{fmt}}" for v in row] for row in arr]


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
            title="elevation (m)",
            title_font_size=11,
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
            title=lbl,
            title_font_size=11,
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
    year,
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
    thk = thk_arr[i]
    surf = np.where(thk < 1, np.nan, surf_arr[i])
    bottom = np.where(thk < 1, np.nan, bedrock)

    traces = [bedrock_trace(bedrock, x, y)]
    if show_ocean:
        traces.append(ocean_trace(bedrock, thk, x, y))
    if show_calving:
        cf = calving_front_trace(bedrock, thk, surf_arr[i], x, y)
        if cf is not None:
            traces.append(cf)
    traces += glacier_traces(
        surf, bottom, x, y, prop_arr[i], cs, lbl, vmin, vmax, opacity
    )
    return {"data": traces, "name": int(year)}


# ── 3-D figure ────────────────────────────────────────────────────────────────


def build_3d_figure(
    ds,
    prop_key,
    z_exag,
    opacity,
    show_ocean,
    show_calving,
    title,
    vmin_clamp,
    vmax_clamp,
):
    bedrock = np.array(ds.topg[0])
    x, y = np.array(ds.x), np.array(ds.y)
    time = np.array(ds.time)
    thk_arr = np.array(ds.thk)
    surf_arr = np.array(ds.usurf)
    prop_arr, cs, lbl, vmin, vmax = extract_property(
        ds, prop_key, vmin_clamp, vmax_clamp
    )

    frames = [
        make_frame(
            i,
            yr,
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
        )
        for i, yr in enumerate(time)
    ]

    steps = [
        {
            "args": [[int(yr)], {"frame": {"duration": 0, "redraw": True}}],
            "label": str(int(yr)),
            "method": "animate",
        }
        for yr in time
    ]

    res = float(x[1] - x[0])
    ratio_y = bedrock.shape[0] / bedrock.shape[1]
    ratio_z = (
        (float(bedrock.max()) - float(bedrock.min()))
        / (bedrock.shape[0] * res)
        * z_exag
    )

    layout = go.Layout(
        height=760,
        margin=dict(l=0, r=0, t=38, b=0),
        title=dict(text=title, font=dict(size=14, family=FONT_FAMILY)),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_FAMILY, size=12),
        uirevision="static",
        legend=dict(
            orientation="h",
            y=1.03,
            x=0,
            bgcolor="rgba(255,255,255,0.75)",
            bordercolor="#bbb",
            borderwidth=1,
        ),
        scene=dict(
            xaxis=dict(showbackground=False, showticklabels=False, title=""),
            yaxis=dict(showbackground=False, showticklabels=False, title=""),
            zaxis=dict(showbackground=False, showticklabels=False, title=""),
            bgcolor="rgba(220,232,248,1)",
        ),
        scene_aspectratio=dict(x=1, y=ratio_y, z=ratio_z),
        sliders=[
            dict(
                active=0,
                currentvalue=dict(
                    font=dict(size=14, family=FONT_FAMILY),
                    prefix="year: ",
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
        margin=dict(l=60, r=70, t=30, b=40),
        title=dict(
            text="glacier statistics over time", font=dict(size=13, family=FONT_FAMILY)
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(220,232,248,0.4)",
        font=dict(family=FONT_FAMILY, size=11),
        xaxis=dict(title="year", showgrid=True, gridcolor="#c8d8e8", zeroline=False),
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
            y=1.22,
            x=0,
            bgcolor="rgba(255,255,255,0.75)",
            bordercolor="#bbb",
            borderwidth=1,
        ),
    )
    return fig


# ── Dash app ──────────────────────────────────────────────────────────────────


def finalize(params, state):
    output_file = get_output_path(params)
    ds = load_ds(output_file)

    try:
        title = params.oggm_RGI_ID
    except AttributeError:
        title = os.path.splitext(os.path.basename(output_file))[0]

    RANGES = {k: property_range(ds, k) for k in COLORSCALES}

    app = Dash(__name__, external_stylesheets=[GOOGLE_FONT])

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
                            html.Label("property", style=LABEL),
                            dcc.Dropdown(
                                list(PROPERTY_MAP),
                                list(PROPERTY_MAP)[0],
                                id="property",
                                clearable=False,
                            ),
                        ],
                        style={**CTRL, "flex": "0 1 230px"},
                    ),
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
                                tooltip={"placement": "bottom", "always_visible": True},
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
                                value=0.9,
                                id="opacity",
                                marks={
                                    v: f"{int(v*100)}%" for v in (0.25, 0.5, 0.75, 1.0)
                                },
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                        ],
                        style={**CTRL, "flex": "2 1 260px"},
                    ),
                    html.Div(
                        [
                            dcc.Checklist(
                                id="show_ocean",
                                options=[
                                    {"label": "  ocean (z = 0)", "value": "ocean"}
                                ],
                                value=[],
                                style={"marginBottom": "8px"},
                            ),
                            dcc.Checklist(
                                id="show_calving",
                                options=[
                                    {"label": "  calving front", "value": "calving"}
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
                            tooltip={"placement": "bottom", "always_visible": True},
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
            dcc.Graph(id="surface_3d", config={"scrollZoom": True}),
            html.Div(
                [dcc.Graph(id="stats_chart", figure=build_stats_figure(ds))],
                style={"borderTop": "1px solid #c4cdd8"},
            ),
        ],
        style={"fontFamily": FONT_FAMILY, "fontSize": "13px", "background": "#f7f9fc"},
    )

    @app.callback(
        Output("clamp_range", "min"),
        Output("clamp_range", "max"),
        Output("clamp_range", "step"),
        Output("clamp_range", "value"),
        Output("clamp_range", "marks"),
        Input("property", "value"),
    )
    def update_clamp_bounds(prop_label):
        key = PROPERTY_MAP[prop_label][0]
        lo, hi = RANGES.get(key, RANGES["thickness"])
        step = (hi - lo) / 200 if hi != lo else 0.01
        marks = {float(v): f"{v:.3g}" for v in np.linspace(lo, hi, 5)}
        return lo, hi, step, [lo, hi], marks

    @app.callback(
        Output("surface_3d", "figure"),
        Input("property", "value"),
        Input("z_exag", "value"),
        Input("opacity", "value"),
        Input("show_ocean", "value"),
        Input("show_calving", "value"),
        Input("clamp_range", "value"),
    )
    def update_figure(
        prop_label, z_exag, opacity, show_ocean, show_calving, clamp_range
    ):
        prop_key = PROPERTY_MAP[prop_label][0]
        ocean = "ocean" in (show_ocean or [])
        calving = "calving" in (show_calving or [])
        vmin, vmax = clamp_range if clamp_range else [None, None]
        return build_3d_figure(
            load_ds(output_file),
            prop_key,
            z_exag,
            opacity,
            ocean,
            calving,
            title,
            vmin,
            vmax,
        )

    app.run(debug=True, host="0.0.0.0", port=8050)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize IGM glacier output — Plotly/Dash 3-D animation."
    )
    parser.add_argument("--param_file", default="params.yaml")
    parser.add_argument("--output_file", default=None)
    args = parser.parse_args()

    if args.output_file:
        nc_path = args.output_file
    else:
        with open(args.param_file) as f:
            cfg = yaml.safe_load(f)
        nc_path = (
            cfg.get("outputs", {}).get("write_ncdf", {}).get("output_file", "output.nc")
        )
        base = os.path.dirname(os.path.abspath(args.param_file))
        for candidate in [
            os.path.join(base, nc_path),
            os.path.join(base, "outputs", nc_path),
            os.path.join(base, "..", "outputs", nc_path),
        ]:
            if os.path.exists(candidate):
                nc_path = os.path.normpath(candidate)
                break

    finalize(argparse.Namespace(output_file=nc_path), None)
