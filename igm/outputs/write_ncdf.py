#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from netCDF4 import Dataset

from igm.utils.math.getmag import getmag


def initialize(cfg, state):

    # give information on variables for output ncdf, TODO: IMPROVE
    state.var_info_ncdf_ex = {}
    state.var_info_ncdf_ex["topg"] = ["Basal Topography", "m"]
    state.var_info_ncdf_ex["usurf"] = ["Surface Topography", "m"]
    state.var_info_ncdf_ex["thk"] = ["Ice Thickness", "m"]
    state.var_info_ncdf_ex["icemask"] = ["Ice mask", "NO UNIT"]
    state.var_info_ncdf_ex["smb"] = ["Surface Mass Balance", "m/y ice eq"]
    state.var_info_ncdf_ex["ubar"] = ["x depth-average velocity of ice", "m/y"]
    state.var_info_ncdf_ex["vbar"] = ["y depth-average velocity of ice", "m/y"]
    state.var_info_ncdf_ex["velbar_mag"] = [
        "Depth-average velocity magnitude of ice",
        "m/y",
    ]
    state.var_info_ncdf_ex["uvelsurf"] = ["x surface velocity of ice", "m/y"]
    state.var_info_ncdf_ex["vvelsurf"] = ["y surface velocity of ice", "m/y"]
    state.var_info_ncdf_ex["wvelsurf"] = ["z surface velocity of ice", "m/y"]
    state.var_info_ncdf_ex["velsurf_mag"] = ["Surface velocity magnitude of ice", "m/y"]
    state.var_info_ncdf_ex["uvelbase"] = ["x basal velocity of ice", "m/y"]
    state.var_info_ncdf_ex["vvelbase"] = ["y basal velocity of ice", "m/y"]
    state.var_info_ncdf_ex["wvelbase"] = ["z basal velocity of ice", "m/y"]
    state.var_info_ncdf_ex["velbase_mag"] = ["Basal velocity magnitude of ice", "m/y"]
    state.var_info_ncdf_ex["divflux"] = ["Divergence of the ice flux", "m/y"]
    state.var_info_ncdf_ex["strflowctrl"] = [
        "arrhenius+1.0*slidingco",
        "MPa$^{-3}$ a$^{-1}$",
    ]
    state.var_info_ncdf_ex["dtopgdt"] = ["Erosion rate", "m/y"]
    state.var_info_ncdf_ex["arrhenius"] = ["Arrhenius factor", "MPa$^{-3}$ a$^{-1}$"]
    state.var_info_ncdf_ex["slidingco"] = [
        "Sliding Coefficient",
        "km MPa$^{-3}$ a$^{-1}$",
    ]
    state.var_info_ncdf_ex["meantemp"] = ["Mean anual surface temperatures", "°C"]
    state.var_info_ncdf_ex["meanprec"] = ["Mean anual precipitation", "Kg m^(-2) y^(-1)"]
    state.var_info_ncdf_ex["velsurfobs_mag"] = ["Obs. surf. speed of ice", "m/y"]
    state.var_info_ncdf_ex["weight_particles"] = ["weight_particles", "no"]


def run(cfg, state):
    if state.saveresult:

        if "velbar_mag" in cfg.outputs.write_ncdf.vars_to_save:
            state.velbar_mag = getmag(state.ubar, state.vbar)

        if "velsurf_mag" in cfg.outputs.write_ncdf.vars_to_save:
            state.velsurf_mag = getmag(state.uvelsurf, state.vvelsurf)

        if "velbase_mag" in cfg.outputs.write_ncdf.vars_to_save:
            state.velbase_mag = getmag(state.uvelbase, state.vvelbase)

        if not hasattr(state, "already_called_update_write_ncdf"):
            state.already_called_update_write_ncdf = True

            if hasattr(state, "logger"):
                state.logger.info("Initialize NCDF ex output Files")

            nc = Dataset(cfg.outputs.write_ncdf.output_file, "w", format="NETCDF4")

            nc.createDimension("time", None)
            E = nc.createVariable("time", np.dtype("float32").char, ("time",))
            E.units = "yr"
            E.long_name = "time"
            E.axis = "T"
            E[0] = str(getattr(state, 't', tf.constant(0)).numpy())

            nc.createDimension("y", len(state.y))
            E = nc.createVariable("y", np.dtype("float32").char, ("y",))
            E.units = "m"
            E.long_name = "y"
            E.axis = "Y"
            E[:] = state.y.numpy()

            nc.createDimension("x", len(state.x))
            E = nc.createVariable("x", np.dtype("float32").char, ("x",))
            E.units = "m"
            E.long_name = "x"
            E.axis = "X"
            E[:] = state.x.numpy()

            if hasattr(state, 'pyproj_srs'):
                nc.pyproj_srs = state.pyproj_srs

            if hasattr(cfg.processes,'iceflow'):
               if "Nz" in cfg.processes.iceflow:
                   nc.createDimension("z", cfg.processes.iceflow.numerics.Nz)
                   E = nc.createVariable("z", np.dtype("float32").char, ("z",))
                   E.units = "m"
                   E.long_name = "z"
                   E.axis = "Z"
                   E[:] = np.arange(
                       cfg.processes.iceflow.numerics.Nz
                   )  # TODO: fix this, that's not what we want

            for var in cfg.outputs.write_ncdf.vars_to_save:
                if hasattr(state, var):
                    if vars(state)[var].numpy().ndim == 2:
                        E = nc.createVariable(
                            var, np.dtype("float32").char, ("time", "y", "x")
                        )
                        E[0, :, :] = vars(state)[var].numpy()
                    elif vars(state)[var].numpy().ndim == 3:
                        E = nc.createVariable(
                            var, np.dtype("float32").char, ("time", "z", "y", "x")
                        )
                        E[0, :, :, :] = vars(state)[var].numpy()
                    if var in state.var_info_ncdf_ex.keys():
                        E.long_name = state.var_info_ncdf_ex[var][0]
                        E.units = state.var_info_ncdf_ex[var][1]
            nc.close()

        else:
            if hasattr(state, "logger"):
                state.logger.info(
                    "Write NCDF ex file at time : " + str(state.t.numpy())
                )

            nc = Dataset(cfg.outputs.write_ncdf.output_file, "a", format="NETCDF4" )

            d = nc.variables["time"][:].shape[0]
            nc.variables["time"][d] = state.t.numpy()

            for var in cfg.outputs.write_ncdf.vars_to_save:
                if hasattr(state, var):
                    if vars(state)[var].numpy().ndim == 2:
                        nc.variables[var][d, :, :] = vars(state)[var].numpy()
                    elif vars(state)[var].numpy().ndim == 3:
                        nc.variables[var][d, :, :, :] = vars(state)[var].numpy()

            nc.close()

