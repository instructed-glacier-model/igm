#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import os
import tensorflow as tf
import xarray as xr
from scipy.interpolate import RectBivariateSpline, interp1d
from igm.utils.math.interp1d_tf import interp1d_tf
 
def initialize(cfg, state):
    load_climate_data_glacialindex(cfg, state)

    state.air_temp = tf.Variable(state.air_temp_snap[:, :, :, 0], dtype="float32")
    state.air_temp_sd = tf.Variable(state.air_temp_sd_snap[:, :, :, 0], dtype="float32")
    state.precipitation = tf.Variable(
        state.precipitation_snap[:, :, :, 0], dtype="float32"
    )
    state.tempsurfref = tf.Variable(state.tempsurfref_snap[:, :, 0], dtype="float32")
    state.LR = tf.Variable(state.LR_snap[0], dtype="float32")

    state.meanprec = tf.math.reduce_mean(state.precipitation, axis=0)
    state.meantemp = tf.math.reduce_mean(state.air_temp, axis=0)

    state.tlast_clim = tf.Variable(-1.0e50, dtype="float32")


def update(cfg, state):
    if (state.t - state.tlast_clim) >= cfg.processes.clim_glacialindex.update_freq:
        if hasattr(state, "logger"):
            state.logger.info("update climate at time : " + str(state.t.numpy()))


        s = interp1d_tf(state.signal[:, 0], state.signal[:, 1], state.t)

        state.LR.assign((1 - s) * state.LR_snap[0] + s * state.LR_snap[1])

        lapse_rate_cor = (
            state.tempsurfref_snap[:, :, 1] - state.tempsurfref_snap[:, :, 0]
        ) * (state.LR / 1000.0)
        air_temp_0_on_surf_1 = state.air_temp_snap[:, :, :, 0] - lapse_rate_cor

        state.air_temp.assign(
            (1 - s) * air_temp_0_on_surf_1 + s * state.air_temp_snap[:, :, :, 1]
        )
        state.precipitation.assign(
            (1 - s) * state.precipitation_snap[:, :, :, 0]
            + s * state.precipitation_snap[:, :, :, 1]
        )
        state.air_temp_sd.assign(
            (1 - s) * state.air_temp_sd_snap[:, :, :, 0]
            + s * state.air_temp_sd_snap[:, :, :, 1]
        )
        state.tempsurfref.assign(state.tempsurfref_snap[:, :, 1])

        lapse_rate_cor = (state.usurf - state.tempsurfref) * (state.LR / 1000.0)

        state.air_temp.assign(state.air_temp - lapse_rate_cor)
        
        state.meanprec = tf.math.reduce_mean(state.precipitation, axis=0)
        state.meantemp = tf.math.reduce_mean(state.air_temp, axis=0)

        state.tlast_clim.assign(state.t)


def finalize(cfg, state):
    pass


############################################################################################################


def load_climate_data_one_snapshot(cfg, state, filename):
    """
    load the ncdf climate file containing precipitation and temperatures
    """





    ds = xr.open_dataset(filename)

    x = ds["x"].values.astype("float32").squeeze()
    y = ds["y"].values.astype("float32").squeeze()

    air_temp = ds["air_temp"].values.astype("float32").squeeze()
    precipitation = ds["precipitation"].values.astype("float32").squeeze()
    surf_clim_ref = ds["usurf"].values.astype("float32").squeeze()

    if "air_temp_sd" in ds.variables:
        air_temp_sd = ds["air_temp_sd"].values.astype("float32").squeeze()
    else:
        air_temp_sd = np.ones_like(air_temp) * cfg.processes.clim_glacialindex.temp_std
  

    air_temp_sd = np.where(air_temp_sd > 0, air_temp_sd, 5.0)

    if not (y.shape == state.y.shape) & (x.shape == state.x.shape):
        print("Resample climate data")

        air_tempN = np.zeros((air_temp.shape[0], state.y.shape[0], state.x.shape[0]))
        precipitationN = np.zeros(
            (air_temp.shape[0], state.y.shape[0], state.x.shape[0])
        )
        air_temp_sdN = np.zeros((air_temp.shape[0], state.y.shape[0], state.x.shape[0]))
        surf_clim_refN = np.zeros((state.y.shape[0], state.x.shape[0]))

        for i in range(len(air_temp)):
            air_tempN[i] = RectBivariateSpline(y, x, air_temp[i])(state.y, state.x)
            air_temp_sdN[i] = RectBivariateSpline(y, x, air_temp_sd[i])(
                state.y, state.x
            )
            precipitationN[i] = RectBivariateSpline(y, x, precipitation[i])(
                state.y, state.x
            )

        surf_clim_refN = RectBivariateSpline(y, x, surf_clim_ref)(state.y, state.x)

        air_temp = air_tempN
        air_temp_sd = air_temp_sdN
        precipitation = precipitationN
        surf_clim_ref = surf_clim_refN

    air_temp -= 273.15           # unit to [ °C ]
    precipitation *= 31556952.0  # unit to [ kg * m^(-2) * s^(-1) ] -> [ kg * m^(-2) * y^(-1) ]
 
    return [air_temp, air_temp_sd, precipitation, surf_clim_ref]


def load_signal(cfg, state):
    """
    load signal file to force a transient climate
    """ 
    filepath =  os.path.join(state.original_cwd,cfg.processes.clim_glacialindex.signal_file)

    state.signal = np.loadtxt(filepath, dtype=np.float32)


def interpolate_climate_over_a_year(climate_snapshot,resampling):
    """
    this applies interpolation, and shift to all climatic variables
    """
    climate_snapshot_out = []
    for cs in climate_snapshot:
        air_temp, air_temp_sd, precipitation, usurf, LR = cs

        # currently we do not resample in time, but keep monthly resolution to save memory
        # use 52 instead of 12 for weekly resolution, but this will increase memory usage
        air_temp = climate_upsampling_and_shift(air_temp, resampling=resampling)
        air_temp_sd = climate_upsampling_and_shift(air_temp_sd, resampling=resampling)
        precipitation = climate_upsampling_and_shift(precipitation, resampling=resampling)

        climate_snapshot_out.append([air_temp, air_temp_sd, precipitation, usurf, LR])

    return climate_snapshot_out


def load_climate_data_glacialindex(cfg, state):
    """
    load climate data and transient signal, interpolate weekly, and shift
    """
 
    climate_snapshot_0 = load_climate_data_one_snapshot(
        cfg, state, os.path.join(state.original_cwd,cfg.processes.clim_glacialindex.climate_0_file)
    ) + [cfg.processes.clim_glacialindex.vertical_lapse_rate_0]
    climate_snapshot_1 = load_climate_data_one_snapshot(
        cfg, state, os.path.join(state.original_cwd,cfg.processes.clim_glacialindex.climate_1_file)
    ) + [cfg.processes.clim_glacialindex.vertical_lapse_rate_1]
    climate_snapshot = [climate_snapshot_0, climate_snapshot_1]

    ##############

    climate_snapshot = interpolate_climate_over_a_year(climate_snapshot,cfg.processes.clim_glacialindex.temporal_resampling)

    load_signal(cfg, state)

    ##############

    state.air_temp_snap = \
        np.concatenate(
            (
                np.expand_dims(climate_snapshot[0][0], axis=-1),
                np.expand_dims(climate_snapshot[1][0], axis=-1),
            ),
            axis=-1,
        )
    state.air_temp_sd_snap = \
        np.concatenate(
            (
                np.expand_dims(climate_snapshot[0][1], axis=-1),
                np.expand_dims(climate_snapshot[1][1], axis=-1),
            ),
            axis=-1,
        ) 
    state.precipitation_snap = \
        np.concatenate(
            (
                np.expand_dims(climate_snapshot[0][2], axis=-1),
                np.expand_dims(climate_snapshot[1][2], axis=-1),
            ),
            axis=-1,
        ) 
    state.tempsurfref_snap = \
        np.concatenate(
            (
                np.expand_dims(climate_snapshot[0][3], axis=-1),
                np.expand_dims(climate_snapshot[1][3], axis=-1),
            ),
            axis=-1,
        )
    state.LR_snap = \
        np.concatenate(
            (
                np.expand_dims(climate_snapshot[0][4], axis=-1),
                np.expand_dims(climate_snapshot[1][4], axis=-1),
            ),
            axis=-1,
        ) 


def climate_upsampling_and_shift(field, resampling, shift=0.0):
    """
    temporally resample (up) and shift to hydrological year
    """

    if resampling == len(field):
        Y = field

    else:
        assert resampling > len(field)
        mid = (field[-1] + field[0]) / 2.0
        x = np.concatenate(([0], (np.arange(len(field)) + 0.5) / len(field), [1]))
        y = np.concatenate(([mid], field, [mid]))
        X = (np.arange(resampling) + 0.5) / resampling
        #            Y = CubicSpline(x, y, bc_type='periodic', axis=0)(X)
        Y = interp1d(x, y, kind="linear", axis=0)(X)

    # the shift serves to adjust to hyroloogical year, i.e. start Oct 1st
    if shift > 0:
        Y = np.roll(Y, int(len(Y) * (1 - shift)), axis=0)

    return Y


def climate_donwsampling_and_shift(field, resampling, shift=0.75):
    """
    temporally resample (down) and shift to hydrological year
    """

    assert resampling < len(field)

    k = resampling
    m = field.shape[0]
    o = field.shape[1:]
    Y = np.nanmean(field[: (m // k) * k].reshape((m // k, k) + o), axis=0)
    Z = np.nanstd(field[: (m // k) * k].reshape((m // k, k) + o), axis=0)
    Z[Z < 0.1] = 0.1  # ensure the std is not zero

    # the shift serves to adjust to hyroloogical year, i.e. start Oct 1st
    if shift > 0:
        Y = np.roll(Y, int(len(Y) * (1 - shift)), axis=0)
        Z = np.roll(Z, int(len(Z) * (1 - shift)), axis=0)

    return Y, Z
