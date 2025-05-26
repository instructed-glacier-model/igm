#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import tensorflow as tf
from scipy import stats
from netCDF4 import Dataset
from igm.utils.math.getmag import getmag

def update_ncdf_optimize(cfg, state, it):
    """
    Initialize and write the ncdf optimze file
    """

    if hasattr(state, "logger"):
        state.logger.info("Initialize  and write NCDF output Files")
        
    if "velbase_mag" in cfg.processes.data_assimilation.output.vars_to_save:
        state.velbase_mag = getmag(state.uvelbase, state.vvelbase)

    if "velsurf_mag" in cfg.processes.data_assimilation.output.vars_to_save:
        state.velsurf_mag = getmag(state.uvelsurf, state.vvelsurf)

    if "velsurfobs_mag" in cfg.processes.data_assimilation.output.vars_to_save:
        state.velsurfobs_mag = getmag(state.uvelsurfobs, state.vvelsurfobs)
    
    if "sliding_ratio" in cfg.processes.data_assimilation.output.vars_to_save:
        state.sliding_ratio = tf.where(state.velsurf_mag > 10, state.velbase_mag / state.velsurf_mag, np.nan)

    if it == 0:
        nc = Dataset(
            "optimize.nc",
            "w",
            format="NETCDF4",
        )

        nc.createDimension("iterations", None)
        E = nc.createVariable("iterations", np.dtype("float32").char, ("iterations",))
        E.units = "None"
        E.long_name = "iterations"
        E.axis = "ITERATIONS"
        E[0] = it

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

        for var in cfg.processes.data_assimilation.output.vars_to_save:
            E = nc.createVariable(
                var, np.dtype("float32").char, ("iterations", "y", "x")
            )
            E[0, :, :] = vars(state)[var].numpy()

        nc.close()

    else:
        nc = Dataset("optimize.nc", "a", format="NETCDF4", )

        d = nc.variables["iterations"][:].shape[0]

        nc.variables["iterations"][d] = it

        for var in cfg.processes.data_assimilation.output.vars_to_save:
            nc.variables[var][d, :, :] = vars(state)[var].numpy()

        nc.close()


def output_ncdf_optimize_final(cfg, state):
    """
    Write final geology after optimizing
    """
    if cfg.processes.data_assimilation.output.save_iterat_in_ncdf==False:
        if "velbase_mag" in cfg.processes.data_assimilation.output.vars_to_save:
            state.velbase_mag = getmag(state.uvelbase, state.vvelbase)

        if "velsurf_mag" in cfg.processes.data_assimilation.output.vars_to_save:
            state.velsurf_mag = getmag(state.uvelsurf, state.vvelsurf)

        if "velsurfobs_mag" in cfg.processes.data_assimilation.output.vars_to_save:
            state.velsurfobs_mag = getmag(state.uvelsurfobs, state.vvelsurfobs)
        
        if "sliding_ratio" in cfg.processes.data_assimilation.output.vars_to_save:
            state.sliding_ratio = tf.where(state.velsurf_mag > 10, state.velbase_mag / state.velsurf_mag, np.nan)

    nc = Dataset(
        cfg.processes.data_assimilation.output.save_result_in_ncdf,
        "w",
        format="NETCDF4",
    )

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

    for v in cfg.processes.data_assimilation.output.vars_to_save:
        if hasattr(state, v):
            E = nc.createVariable(v, np.dtype("float32").char, ("y", "x"))
            E.standard_name = v
            E[:] = vars(state)[v]

    nc.close()
