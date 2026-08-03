#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import tensorflow as tf
from scipy import stats
from netCDF4 import Dataset
from igm.utils.math.getmag import getmag


def _ensure_z_dim(nc, nz):
    """Create (or check) the vertical dimension for 3D variables (e.g. U, V).
    Same convention as outputs/local.py: dims (z, y, x); NetCDF mixes 2D and
    3D variables in one file without restriction."""
    if "z" not in nc.dimensions:
        nc.createDimension("z", nz)
    elif len(nc.dimensions["z"]) != nz:
        raise ValueError(
            f"3D output: z dimension mismatch ({len(nc.dimensions['z'])} vs {nz})"
        )

def update_ncdf_optimize(cfg, state, it):
    """
    Initialize and write the ncdf optimze file
    """

    if hasattr(state, "logger"):
        state.logger.info("Initialize  and write NCDF output Files")
        
    if "velbase_mag" in cfg.assimilations.data_assimilation.output.vars_to_save:
        state.velbase_mag = getmag(state.uvelbase, state.vvelbase)

    if "velsurf_mag" in cfg.assimilations.data_assimilation.output.vars_to_save:
        state.velsurf_mag = getmag(state.uvelsurf, state.vvelsurf)

    if "velsurfobs_mag" in cfg.assimilations.data_assimilation.output.vars_to_save:
        state.velsurfobs_mag = getmag(state.uvelsurfobs, state.vvelsurfobs)
    
    if "sliding_ratio" in cfg.assimilations.data_assimilation.output.vars_to_save:
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

        for var in cfg.assimilations.data_assimilation.output.vars_to_save:
            arr = getattr(state, var).numpy()
            if arr.ndim == 3:
                _ensure_z_dim(nc, arr.shape[0])
                E = nc.createVariable(
                    var, np.dtype("float32").char, ("iterations", "z", "y", "x")
                )
                E[0, :, :, :] = arr
            else:
                E = nc.createVariable(
                    var, np.dtype("float32").char, ("iterations", "y", "x")
                )
                E[0, :, :] = arr

        nc.close()

    else:
        nc = Dataset("optimize.nc", "a", format="NETCDF4", )

        d = nc.variables["iterations"][:].shape[0]

        nc.variables["iterations"][d] = it

        for var in cfg.assimilations.data_assimilation.output.vars_to_save:
            arr = getattr(state, var).numpy()
            if arr.ndim == 3:
                nc.variables[var][d, :, :, :] = arr
            else:
                nc.variables[var][d, :, :] = arr

        nc.close()


def output_ncdf_optimize_final(cfg, state):
    """
    Write final geology after optimizing
    """
    if cfg.assimilations.data_assimilation.output.save_iterat_in_ncdf==False:
        if "velbase_mag" in cfg.assimilations.data_assimilation.output.vars_to_save:
            state.velbase_mag = getmag(state.uvelbase, state.vvelbase)

        if "velsurf_mag" in cfg.assimilations.data_assimilation.output.vars_to_save:
            state.velsurf_mag = getmag(state.uvelsurf, state.vvelsurf)

        if "velsurfobs_mag" in cfg.assimilations.data_assimilation.output.vars_to_save:
            state.velsurfobs_mag = getmag(state.uvelsurfobs, state.vvelsurfobs)
        
        if "sliding_ratio" in cfg.assimilations.data_assimilation.output.vars_to_save:
            state.sliding_ratio = tf.where(state.velsurf_mag > 10, state.velbase_mag / state.velsurf_mag, np.nan)

    nc = Dataset(
        cfg.assimilations.data_assimilation.output.save_result_in_ncdf,
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

    for v in cfg.assimilations.data_assimilation.output.vars_to_save:
        if hasattr(state, v):
            arr = np.asarray(getattr(state, v))
            if arr.ndim == 3:
                _ensure_z_dim(nc, arr.shape[0])
                E = nc.createVariable(v, np.dtype("float32").char, ("z", "y", "x"))
            else:
                E = nc.createVariable(v, np.dtype("float32").char, ("y", "x"))
            E.standard_name = v
            E[:] = arr

    nc.close()
