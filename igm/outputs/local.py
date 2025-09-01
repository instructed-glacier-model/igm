import xarray as xr
import numpy as np
import tensorflow as tf
import os

from igm.utils.math.getmag import getmag

def initialize(cfg, state):
    state.var_info_ncdf_ex = {
        "topg": ["Basal Topography", "m"],
        "usurf": ["Surface Topography", "m"],
        "thk": ["Ice Thickness", "m"],
        "icemask": ["Ice mask", "NO UNIT"],
        "smb": ["Surface Mass Balance", "m/y ice eq"],
        "ubar": ["x depth-average velocity of ice", "m/y"],
        "vbar": ["y depth-average velocity of ice", "m/y"],
        "velbar_mag": ["Depth-average velocity magnitude of ice", "m/y"],
        "uvelsurf": ["x surface velocity of ice", "m/y"],
        "vvelsurf": ["y surface velocity of ice", "m/y"],
        "wvelsurf": ["z surface velocity of ice", "m/y"],
        "velsurf_mag": ["Surface velocity magnitude of ice", "m/y"],
        "uvelbase": ["x basal velocity of ice", "m/y"],
        "vvelbase": ["y basal velocity of ice", "m/y"],
        "wvelbase": ["z basal velocity of ice", "m/y"],
        "velbase_mag": ["Basal velocity magnitude of ice", "m/y"],
        "divflux": ["Divergence of the ice flux", "m/y"],
        "strflowctrl": ["arrhenius+1.0*slidingco", "MPa$^{-3}$ a$^{-1}$"],
        "dtopgdt": ["Erosion rate", "m/y"],
        "arrhenius": ["Arrhenius factor", "MPa$^{-3}$ a$^{-1}$"],
        "slidingco": ["Sliding Coefficient", "km MPa$^{-3}$ a$^{-1}$"],
        "meantemp": ["Mean annual surface temperatures", "°C"],
        "meanprec": ["Mean annual precipitation", "Kg m^(-2) y^(-1)"],
        "velsurfobs_mag": ["Obs. surf. speed of ice", "m/y"],
        "weight_particles": ["weight_particles", "no"]
    }

    state.var_info_ncdf_ts = {}
    state.var_info_ncdf_ts["vol"] = ["Ice volume", "km^3"]
    state.var_info_ncdf_ts["area"] = ["Glaciated area", "km^2"]


def run(cfg, state):

    if not state.saveresult:
        return

    # Prepare any derived quantities
    if "velbar_mag" in cfg.outputs.local.vars_to_save:
        state.velbar_mag = getmag(state.ubar, state.vbar)

    if "velsurf_mag" in cfg.outputs.local.vars_to_save:
        state.velsurf_mag = getmag(state.uvelsurf, state.vvelsurf)

    if "velbase_mag" in cfg.outputs.local.vars_to_save:
        state.velbase_mag = getmag(state.uvelbase, state.vvelbase)

    if 'netcdf' in cfg.outputs.local.file_format_list:
        update_netcdf_ex(cfg,state)
    
    if 'tif' in cfg.outputs.local.file_format_list:
        write_tif(cfg,state)

    if cfg.outputs.local.write_ts:
        update_netcdf_ts(cfg,state)

#############################################

def write_tif(cfg,state):

    var_list = cfg.outputs.local.vars_to_save

    for var in var_list:
        if not hasattr(state, var):
                continue
        
        var_data = vars(state)[var].numpy()
        file_name = f"{var}-{str(getattr(state, 't', tf.constant(0)).numpy()).zfill(6)}.tif"

        data_array = xr.DataArray(
            var_data,
            dims=("y", "x"),
            coords={"y": state.y.numpy(), "x": state.x.numpy()}
        )

        if "crs" in cfg.outputs.local:
            data_array.rio.write_crs(cfg.outputs.local.crs, inplace=True)

        data_array.rio.to_raster(file_name)

#####################################

def update_netcdf_ex(cfg,state):

    file_path = cfg.outputs.local.output_file
    var_list = cfg.outputs.local.vars_to_save

    def create_data_vars():
        data_vars = {}
        for var in var_list:
            if not hasattr(state, var):
                continue
            arr = vars(state)[var].numpy()
            dims = ("y", "x") if arr.ndim == 2 else ("z", "y", "x")
            data = xr.DataArray(arr, dims=dims)
            data = data.expand_dims(time=[getattr(state, 't', tf.constant(0)).numpy()])
            attrs = {}
            if var in state.var_info_ncdf_ex:
                attrs["long_name"], attrs["units"] = state.var_info_ncdf_ex[var]
            data.attrs = attrs
            data_vars[var] = data
        return data_vars

    if not hasattr(state, "already_called_update_local"):
        if hasattr(state, "logger"):
            state.logger.info("Creating new NetCDF file with xarray")

        coords = {
            "x": ("x", state.x.numpy()),
            "y": ("y", state.y.numpy()),
            "time": ("time", [getattr(state, 't', tf.constant(0)).numpy()])
        }

        if (hasattr(cfg, 'processes') and hasattr(cfg.processes, 'iceflow')):
            coords["z"] = ("z", np.arange(cfg.processes.iceflow.numerics.Nz))

        ds = xr.Dataset(
            data_vars=create_data_vars(),
            coords=coords,
            attrs={"pyproj_srs": getattr(state, "pyproj_srs", "")},
        )

        ds.to_netcdf(file_path, mode="w")

        state.already_called_update_local = True
    else:
        if hasattr(state, "logger"):
            state.logger.info(f"Appending to NetCDF file at iteration {state.it}")

        ds_existing = xr.open_dataset(file_path)
        new_data = xr.Dataset(
            data_vars=create_data_vars(),
            coords={"time": [getattr(state, 't', tf.constant(0)).numpy()]},
        )

        # concat and write again
        ds_concat = xr.concat([ds_existing, new_data], dim="time")
        os.remove(file_path)
        ds_concat.to_netcdf(file_path, mode="w")  # overwrite safely

#########################################################

def update_netcdf_ts(cfg,state):

    file_path = cfg.outputs.local.output_ts_file
    
    vol = np.sum(state.thk) * (state.dx**2) / 10**9
    area = np.sum(state.thk > 1) * (state.dx**2) / 10**6

    if not hasattr(state, "already_called_update_write_ts"):
        state.already_called_update_write_ts = True

        if hasattr(state, "logger"):
            state.logger.info("Initialize NCDF ts output Files")

        # Initialize the xarray Dataset
        ds = xr.Dataset(
            {
                "time": ("time", [getattr(state, 't', tf.constant(0)).numpy()]),
                "vol": ("time", [vol]),
                "area": ("time", [area]),
            },
            attrs={
                "vol_long_name": state.var_info_ncdf_ts["vol"][0],
                "vol_units": state.var_info_ncdf_ts["vol"][1],
                "area_long_name": state.var_info_ncdf_ts["area"][0],
                "area_units": state.var_info_ncdf_ts["area"][1],
            }
        )
        ds.time.attrs["units"] = "yr"
        ds.time.attrs["long_name"] = "time"
        ds.to_netcdf(file_path, mode="w", format="NETCDF4")

    else:
        if hasattr(state, "logger"):
            state.logger.info(
                "Write NCDF ts file at itaration : " + str(state.it)
            )

        # Append new data to existing NetCDF file
        with xr.open_dataset(file_path) as ds:
            ds_new = xr.Dataset(
                {
                    "time": ("time", [getattr(state, 't', tf.constant(0)).numpy()]),
                    "vol": ("time", [vol]),
                    "area": ("time", [area]),
                }
            )
            ds_combined = xr.concat([ds, ds_new], dim="time")
            os.remove(file_path)
            ds_combined.to_netcdf(file_path, mode="w", format="NETCDF4") 

