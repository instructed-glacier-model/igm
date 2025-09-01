#!/usr/bin/env python3
import copy
import numpy as np
import json
import xarray as xr
 
def finalize(params, state):
    import plotly.graph_objects as go
    import math
    from dash import Dash, dcc, html, Input, Output

    import matplotlib.cm as cm  # Import the Matplotlib colormap library

    # create HTML layout for app
    fig = go.Figure()
    app = Dash(__name__)
    app.layout = html.Div(
        [
            html.Div(
                [
                    # dropdown menu for property
                    html.Div(
                        children=[
                            dcc.Dropdown(
                                ["thickness [m]", "velocity [m/a]", "SMB [m]"],
                                "thickness [m]",
                                id="property",
                            )
                        ],
                        style={"margin-bottom": "10px"},
                    ),

                    # 3D surface plot
                    dcc.Graph(id="mnt_surface", figure=fig),
                ],
                style={"width": "100%", "height": "800px", "display": "inline-block"},
            ),
        ],
        style={
            "font-family": "monospace",
            "font-size": "x-large",
        },
    )

    ### update graph everytime an input is changed
    @app.callback(
        Output("mnt_surface", "figure"),
        Input("property", "value"),
    )
    def updata_graph(property):
        # read output.nc
        output_file = params.wncd_output_file
        ds = xr.open_dataset(output_file, engine="netcdf4")

        # get attributes from ds
        bedrock = np.array(ds.topg[0])
        time = np.array(ds.time)
        lat_range = np.array(ds.x)
        lon_range = np.array(ds.y)
        glacier_surfaces = np.array(ds.usurf)
        thicknesses = np.array(ds.thk)
        velocities = np.array(ds.velsurf_mag)
        smbs = np.array(ds.smb)

        # choose property that is displayed on the glacier surface
        if property == "thickness [m]":
            property_maps = thicknesses
            color_scale = "Blues"
            max_property_map = np.max(property_maps)
            min_property_map = np.min(property_maps)
        elif property == "velocity [m/a]":
            property_maps = velocities
            color_scale = "magma"
            max_property_map = np.max(property_maps)
            min_property_map = np.min(property_maps)
        else:
            property_maps = smbs
            color_scale = "rdbu"
            max_property_map = np.max(property_maps)
            min_property_map = np.min(property_maps)
            max_dis = np.max([abs(max_property_map), abs(min_property_map)])
            max_property_map = max_dis
            min_property_map = -max_dis

        # make edges equal so that it looks like a volume
        max_bedrock = np.max(bedrock)
        min_bedrock = np.min(bedrock)
        bedrock_border = copy.copy(bedrock)
        bedrock_border[0, :] = min_bedrock
        bedrock_border[-1, :] = min_bedrock
        bedrock_border[:, 0] = min_bedrock
        bedrock_border[:, -1] = min_bedrock

        # aim to mimic the matplotlib terrain
        custom_colorscale = [
            [0.0, "rgb(224,205,169)"],
            [0.2, "rgb(180,170,150)"],
            [0.4, "rgb(135,135,135)"],
            [0.6, "rgb(130,90,50)"],
            [0.8, "rgb(120,80,40)"],
            [1.0, "rgb(100,70,30)"],
        ]

        # create time frames for slider
        frames = []
        slider_steps = []
        for i, year in enumerate(time):
            # update elevation data and property map
            property_map = property_maps[i]

            glacier_surface = glacier_surfaces[i]
            glacier_surface[thicknesses[i] < 1] = None

            glacier_bottom = copy.copy(bedrock)
            glacier_bottom[thicknesses[i] < 1] = None

            # create 3D surface plots with property as surface color
            surface_fig = go.Surface(
                z=glacier_surface,
                x=lat_range,
                y=lon_range,
                colorscale=color_scale,
                cmax=max_property_map,
                cmin=min_property_map,
                surfacecolor=property_map,
                showlegend=True,
                name="glacier surface",
                colorbar=dict(title=property, titleside="right"),
            )

            # create 3D plot of the bottom of the glacier
            bottom_fig = go.Surface(
                z=glacier_bottom,
                x=lat_range,
                y=lon_range,
                colorscale=color_scale,
                cmax=max_property_map,
                cmin=min_property_map,
                surfacecolor=property_map,
                showlegend=True,
                name="glacier bottom",
                colorbar=dict(title=property, titleside="right"),
            )

            # create 3D bedrock plots
            bedrock_fig = go.Surface(
                z=bedrock_border,
                x=lat_range,
                y=lon_range,
                colorscale='gray',
                opacity=1,
                showlegend=True,
                name="bedrock",
                cmax=max_bedrock,
                cmin=0,
                colorbar=dict(x=-0.1, title="elevation [m]", titleside="right"),
            )

            # create frame
            frame = {"data": [bedrock_fig, surface_fig, bottom_fig], "name": int(year)}
            frames.append(frame)

            # add slider step
            slider_step = {
                "args": [
                    [int(year)],
                    {
                        "frame": {"duration": 0, "redraw": True},
                    },
                ],
                "label": str(int(year)),
                "method": "animate",
            }
            slider_steps.append(slider_step)

        # define slider with layout
        sliders_dict = {
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 20},
                "prefix": "Year:",
                "visible": True,
                "xanchor": "right",
            },
            "transition": {"duration": 0, "easing": "cubic-in-out"},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": slider_steps,
        }

        # define figure layout
        try:
            title = params.oggm_RGI_ID
        except:
            title = "glacier"

        # compute aspect ratio of the base
        resolution = int(lat_range[1]-lat_range[0])
        ratio_y = bedrock.shape[0] / bedrock.shape[1]
        ratio_z = (max_bedrock - min_bedrock) / (bedrock.shape[0] * resolution)
        ratio_z *= 2  # emphasize z-axis to make mountians look twice as steep

        fig_dict = dict(
            data=frames[0]["data"],
            frames=frames,
            layout=dict(  # width=1800,
                height=900,
                margin=dict(l=0, r=0, t=30, b=0),
                sliders=[sliders_dict],
                title=title,
                font=dict(family="monospace"),
                legend={"orientation": "h", "yanchor": "bottom", "xanchor": "left"},
                scene=dict(
                    zaxis=dict(showbackground=False, showticklabels=False, title=""),
                    xaxis=dict(
                        showbackground=False,
                        showticklabels=False,
                        visible=True,
                        range=[lat_range[0], lat_range[-1]],
                        title=""#"Longitude",
                    ),
                    yaxis=dict(
                        showbackground=False,
                        showticklabels=False,
                        visible=True,
                        range=[lon_range[0], lon_range[-1]],
                        title=""#"Latitude",

                    ),
                ),
                scene_aspectratio=dict(x=1, y=ratio_y, z=ratio_z),
                updatemenus=[
                    dict(
                        buttons=[
                            dict(
                                args=[
                                    None,
                                    {
                                        "frame": {"duration": 200, "redraw": True},
                                        "fromcurrent": True,
                                        "transition": {
                                            "duration": 0,
                                            "easing": "quadratic-in-out",
                                        },
                                    },
                                ],
                                label="Play",
                                method="animate",
                            ),
                            dict(
                                args=[
                                    [None],
                                    {
                                        "frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate",
                                        "transition": {"duration": 0},
                                    },
                                ],
                                label="Pause",
                                method="animate",
                            ),
                        ],
                        direction="left",
                        pad={"r": 10, "t": 87},
                        showactive=False,
                        type="buttons",
                        x=0.1,
                        xanchor="right",
                        y=0,
                        yanchor="top",
                    )
                ],
            ),
        )
        # create figure
        fig = go.Figure(fig_dict)
        return fig

    # start dash app
    app.run_server(debug=False, port=8050)


if __name__ == "__main__":
    import argparse
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Load a JSON file and pass its content to finalize.")
    parser.add_argument('--param_file',
                        type=str,
                        default='params_saved.json',
                        help='Path to the JSON parameter file')
    
    parser.add_argument('--wncd_output_file',
                        type=str,
                        default='output.nc',
                        help='output')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Load the JSON file
    with open(args.param_file, 'r') as f:
        param_dict = json.load(f)

    # Convert the loaded JSON dictionary into a Namespace
    param_data = argparse.Namespace(**param_dict)

    # Call finalize with the loaded JSON data as the first argument
    finalize(param_data, None)
