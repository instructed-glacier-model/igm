#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import tensorflow as tf
from netCDF4 import Dataset
from igm.utils.math.getmag import getmag


def update_ncdf_optimize(cfg, state, it):
    """
    Initialize and write the ncdf optimize file.
    """

    if hasattr(state, "logger"):
        state.logger.info("Initialize and write NCDF output Files")

    has_costs = hasattr(state, "da_cost_total")
    has_cost_history = hasattr(state, "da_cost_total_hist")
    has_retrain_iter_num = hasattr(state, "retrain_iter_num")

    if "velbase_mag" in cfg.assimilations.field_inversion.output.vars_to_save:
        state.velbase_mag = getmag(state.uvelbase, state.vvelbase)

    if "velsurf_mag" in cfg.assimilations.field_inversion.output.vars_to_save:
        state.velsurf_mag = getmag(state.uvelsurf, state.vvelsurf)

    if "velsurfobs_mag" in cfg.assimilations.field_inversion.output.vars_to_save:
        state.velsurfobs_mag = getmag(state.uvelsurfobs, state.vvelsurfobs)

    if "sliding_ratio" in cfg.assimilations.field_inversion.output.vars_to_save:
        state.sliding_ratio = tf.where(
            state.velsurf_mag > 10,
            state.velbase_mag / state.velsurf_mag,
            np.nan,
        )

    if it == 0:
        nc = Dataset("optimize.nc", "w", format="NETCDF4")

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

        if has_costs:
            C = nc.createVariable("da_cost_total", np.dtype("float32").char, ("iterations",))
            C.long_name = "DA total cost"
            C[0] = state.da_cost_total

            C = nc.createVariable("da_cost_data", np.dtype("float32").char, ("iterations",))
            C.long_name = "DA data misfit cost"
            C[0] = state.da_cost_data

            C = nc.createVariable("da_cost_reg", np.dtype("float32").char, ("iterations",))
            C.long_name = "DA regularization cost"
            C[0] = state.da_cost_reg

        if has_retrain_iter_num:
            C = nc.createVariable("retrain_iter_num", np.dtype("int32").char, ("iterations",))
            C.long_name = "Retraining iteration number"
            C[0] = int(state.retrain_iter_num)

        if has_cost_history:
            hist_iter = np.asarray(state.da_cost_hist_iter, dtype=np.int32)
            hist_total = np.asarray(state.da_cost_total_hist, dtype=np.float32)
            hist_data = np.asarray(state.da_cost_data_hist, dtype=np.float32)
            hist_reg = np.asarray(state.da_cost_reg_hist, dtype=np.float32)

            nc.createDimension("da_hist", None)

            H = nc.createVariable("da_cost_hist_iter", np.dtype("int32").char, ("da_hist",))
            H.long_name = "Accepted DA iteration index"
            H[:] = hist_iter

            H = nc.createVariable("da_cost_total_hist", np.dtype("float32").char, ("da_hist",))
            H.long_name = "DA total cost history"
            H[:] = hist_total

            H = nc.createVariable("da_cost_data_hist", np.dtype("float32").char, ("da_hist",))
            H.long_name = "DA data misfit cost history"
            H[:] = hist_data

            H = nc.createVariable("da_cost_reg_hist", np.dtype("float32").char, ("da_hist",))
            H.long_name = "DA regularization cost history"
            H[:] = hist_reg

        for var in cfg.assimilations.field_inversion.output.vars_to_save:
            E = nc.createVariable(var, np.dtype("float32").char, ("iterations", "y", "x"))
            E[0, :, :] = getattr(state, var).numpy()

        nc.close()

    else:
        nc = Dataset("optimize.nc", "a", format="NETCDF4")

        d = nc.variables["iterations"][:].shape[0]

        if has_costs:
            nc.variables["da_cost_total"][d] = state.da_cost_total
            nc.variables["da_cost_data"][d] = state.da_cost_data
            nc.variables["da_cost_reg"][d] = state.da_cost_reg

        if has_retrain_iter_num:
            nc.variables["retrain_iter_num"][d] = int(state.retrain_iter_num)

        if has_cost_history:
            hist_iter = np.asarray(state.da_cost_hist_iter, dtype=np.int32)
            hist_total = np.asarray(state.da_cost_total_hist, dtype=np.float32)
            hist_data = np.asarray(state.da_cost_data_hist, dtype=np.float32)
            hist_reg = np.asarray(state.da_cost_reg_hist, dtype=np.float32)

            n_hist = hist_iter.shape[0]

            nc.variables["da_cost_hist_iter"][:n_hist] = hist_iter
            nc.variables["da_cost_total_hist"][:n_hist] = hist_total
            nc.variables["da_cost_data_hist"][:n_hist] = hist_data
            nc.variables["da_cost_reg_hist"][:n_hist] = hist_reg

        nc.variables["iterations"][d] = it

        for var in cfg.assimilations.field_inversion.output.vars_to_save:
            nc.variables[var][d, :, :] = getattr(state, var).numpy()

        nc.close()