#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import tensorflow as tf
import xarray as xr 
import os

def initialize(cfg, state):

    # Lazy loading with chunking in time
    ds = xr.open_dataset(os.path.join(state.original_cwd,cfg.processes.read_output.input_file), 
                         chunks={"time": 1}) # comment this line if you don't want to install dask

    # Read coordinates and time
    state.x = ds["x"].values.astype("float32")
    state.y = ds["y"].values.astype("float32")
    state.time = ds["time"].values.astype("float32")

    state.list_var = [v for v in ds.data_vars if v not in ["x", "y", "z", "time"]]

    # Crop if needed
    if cfg.processes.read_output.crop:
        x = state.x
        y = state.y
        i0 = max(0, int((cfg.processes.read_output.xmin - x[0]) / (x[1] - x[0])))
        i1 = min(int((cfg.processes.read_output.xmax - x[0]) / (x[1] - x[0])), x.shape[0] - 1)
        i1 = max(i0 + 1, i1)
        j0 = max(0, int((cfg.processes.read_output.ymin - y[0]) / (y[1] - y[0])))
        j1 = min(int((cfg.processes.read_output.ymax - y[0]) / (y[1] - y[0])), y.shape[0] - 1)
        j1 = max(j0 + 1, j1)
        state.crop = dict(x=slice(i0, i1), y=slice(j0, j1))
        state.x = state.x[i0:i1]
        state.y = state.y[j0:j1]
    else:
        state.crop = None

    # Store lazy-loaded dataset for later
    state.ds = ds

    # Save x and y as constants
    state.x = tf.constant(state.x.astype("float32"))
    state.y = tf.constant(state.y.astype("float32"))

    start = state.time[0]
    end = state.time[-1]
     
    state.t = tf.Variable(float(start))
    state.it = 0
    state.saveresult = True
    state.dt = state.time[1] - state.time[0]
    state.dx = state.x[1] - state.x[0]
 
def update(cfg, state):
    state.t.assign(state.time[state.it])

    for var in state.list_var:
        if state.crop:
            data = state.ds[var].isel(time=state.it, x=state.crop["x"], y=state.crop["y"])
        else:
            data = state.ds[var].isel(time=state.it)
        vars(state)[var] = tf.Variable(data.values.astype("float32"))

    if (state.it >= state.time.shape[0] - 1):
        state.continue_run = False

def finalize(cfg, state):
    # Clean up if needed (not strictly necessary with xarray)
    state.ds.close()