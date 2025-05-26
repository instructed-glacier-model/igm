#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import os, sys, shutil
import matplotlib.pyplot as plt
import tensorflow as tf
import xarray as xr
import json
from igm.utils.math.interp1d_tf import interp1d_tf

def initialize(cfg, state):
    # load the given parameters from the json file

    path_data = os.path.join(state.original_cwd,cfg.core.folder_data)
    path_RGI = os.path.join(path_data, cfg.inputs.oggm_shop.RGI_ID)
    
    with open(os.path.join(path_RGI, "mb_calib.json"), "r") as json_file:
        jsonString = json_file.read()

    oggm_mb_calib = json.loads(jsonString)

    state.temp_default_gradient = oggm_mb_calib["mb_global_params"][
        "temp_default_gradient"
    ]
    
    # ! I am passing these through the 'state' object instead of the cfg object as the cfg should be static ideally... we can change this later...
    
    state.temp_bias = oggm_mb_calib["temp_bias"]
    state.prcp_fac = oggm_mb_calib["prcp_fac"]
 
    ds = xr.open_dataset(os.path.join(path_RGI, "climate_historical.nc"))
    
    time = ds["time"].values.astype("float32").squeeze()       # unit: year
    prcp = ds["prcp"].values.astype("float32").squeeze()       # unit: kg * m^(-2)
    temp = ds["temp"].values.astype("float32").squeeze()       # unit: degree Celsius
    temp_std = ds["temp_std"].values.astype("float32").squeeze()  # unit: degree Celsius

    state.ref_hgt = ds.attrs["ref_hgt"]
    state.yr_0 = ds.attrs["yr_0"]
  
    # reshape climate data per year and month
    nb_y = int(time.shape[0] / 12)
    nb_m = 12

    state.prec = prcp.reshape((nb_y, nb_m))
    state.temp = temp.reshape((nb_y, nb_m))
    state.temp_std = temp_std.reshape((nb_y, nb_m))

    # correct the temperature and precipitation with factor and bias
    state.temp = state.temp + state.temp_bias
    state.prec = state.prec * state.prcp_fac

    # fix the units of precipitation
    state.prec = nb_m * state.prec  # kg * m^(-2) * month^(-1) ->  kg * m^(-2) * y^(-1)

    # intitalize air_temp and precipitation fields
    state.air_temp = tf.Variable(
        tf.zeros((nb_m, state.y.shape[0], state.x.shape[0])),
        dtype="float32", trainable=False
    )
    state.air_temp_std = tf.Variable(
        tf.zeros((nb_m, state.y.shape[0], state.x.shape[0])),
        dtype="float32", trainable=False
    )
    state.precipitation = tf.Variable(
        tf.zeros((nb_m, state.y.shape[0], state.x.shape[0])),
        dtype="float32", trainable=False
    )

    state.meanprec = tf.math.reduce_mean(state.precipitation, axis=0)
    state.meantemp = tf.math.reduce_mean(state.air_temp, axis=0)

    state.tlast_clim_oggm = tf.Variable(-(10**10), dtype="float32", trainable=False)

    if cfg.processes.clim_oggm.clim_trend_array == []:
        state.climpar = np.loadtxt(
            cfg.processes.clim_oggm.file, # ! does this exist? if not, I will set it...
            skiprows=1,
            dtype=np.float32,
        )
    else:
        state.climpar = np.array(cfg.processes.clim_oggm.clim_trend_array[1:]).astype(
            np.float32
        )

    np.random.seed(cfg.processes.clim_oggm.seed_par)  # fix the seed


def update(cfg, state):
    if (state.t - state.tlast_clim_oggm) >= cfg.processes.clim_oggm.update_freq:
        if hasattr(state, "logger"):
            state.logger.info("update climate at time : " + str(state.t.numpy()))

        # find out the index that corresponds to the current year
        index_year = int(state.t - state.yr_0)

        if (index_year >= 0) & (index_year < state.prec.shape[0]):
            II = index_year
            delta_temp = 0.0
            prec_scal = 1.0
        else:
            i0, i1 = np.round(cfg.processes.clim_oggm.ref_period - state.yr_0)
            II = np.random.randint(i0, i1)
            delta_temp = interp1d_tf(state.climpar[:, 0], state.climpar[:, 1], state.t)
            prec_scal = interp1d_tf(state.climpar[:, 0], state.climpar[:, 2], state.t)

        PREC = tf.expand_dims(
            tf.expand_dims(np.squeeze(state.prec[II, :]), axis=-1), axis=-1
        )
        TEMP = tf.expand_dims(
            tf.expand_dims(np.squeeze(state.temp[II, :]), axis=-1), axis=-1
        )
        TEMP_STD = tf.expand_dims(
            tf.expand_dims(np.squeeze(state.temp_std[II, :]), axis=-1), axis=-1
        )

        # apply delta temp and precp scaling
        TEMP += delta_temp
        PREC *= prec_scal

        # extend air_temp and precipitation over the entire glacier and all day of the year
        state.precipitation = tf.tile(PREC, (1, state.y.shape[0], state.x.shape[0]))
        state.air_temp = tf.tile(TEMP, (1, state.y.shape[0], state.x.shape[0]))
        state.air_temp_std = tf.tile(TEMP_STD, (1, state.y.shape[0], state.x.shape[0]))

        # vertical correction (lapse rates)
        temp_corr_addi = state.temp_default_gradient * (state.usurf - state.ref_hgt)
        temp_corr_addi = tf.expand_dims(temp_corr_addi, axis=0)
        temp_corr_addi = tf.tile(temp_corr_addi, (state.temp.shape[1], 1, 1))

        # the final precipitation and temperature must have shape (12,ny,nx)
        state.air_temp = state.air_temp + temp_corr_addi
        
        state.meanprec = tf.math.reduce_mean(state.precipitation, axis=0)
        state.meantemp = tf.math.reduce_mean(state.air_temp, axis=0)

        state.tlast_clim_oggm.assign(state.t)



def finalize(cfg, state):
    pass
