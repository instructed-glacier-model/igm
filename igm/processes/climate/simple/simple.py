#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file
#
# Minimal climate module: constant temperature and precipitation at a
# reference elevation, adjusted to the current surface (`state.usurf`)
# through vertical lapse rates. The fields are tiled over 12 months to
# match the time-dimension contract used by the SMB modules.

import numpy as np
import tensorflow as tf


_N_MONTHS = 12


def initialize(cfg, state):
    p = cfg.processes.climate.simple

    state.tlast_clim = tf.Variable(-1.0e50, dtype="float32")

    ny, nx = state.thk.shape

    state.air_temp = tf.Variable(
        tf.zeros((_N_MONTHS, ny, nx), dtype="float32"), trainable=False
    )
    state.precipitation = tf.Variable(
        tf.zeros((_N_MONTHS, ny, nx), dtype="float32"), trainable=False
    )
    state.air_temp_sd = tf.Variable(
        tf.fill((_N_MONTHS, ny, nx), float(p.temp_std)), dtype="float32",
        trainable=False,
    )

    _fill_constant_fields(cfg, state)
    state.meanprec = tf.math.reduce_mean(state.precipitation, axis=0)
    state.meantemp = tf.math.reduce_mean(state.air_temp, axis=0)


def update(cfg, state):
    p = cfg.processes.climate.simple
    if (state.t - state.tlast_clim) >= p.update_freq:
        if hasattr(state, "logger"):
            state.logger.info("update climate at time : " + str(state.t.numpy()))

        _fill_constant_fields(cfg, state)
        state.meanprec = tf.math.reduce_mean(state.precipitation, axis=0)
        state.meantemp = tf.math.reduce_mean(state.air_temp, axis=0)
        state.tlast_clim.assign(state.t)


def finalize(cfg, state):
    pass


def _fill_constant_fields(cfg, state):
    p = cfg.processes.climate.simple

    elev_diff = state.usurf - p.ref_elevation  # (ny, nx)

    # Temperature: T(x,y) = temp + temp_lapse_rate * (usurf - ref_elevation)
    air_temp_2d = float(p.temp) + p.temp_lapse_rate * elev_diff

    # Precipitation: P(x,y) = prec * max(0, 1 + prec_lapse_rate * (usurf - ref_elevation))
    prec_factor = 1.0 + p.prec_lapse_rate * elev_diff
    prec_factor = tf.maximum(prec_factor, 0.0)
    precip_2d = float(p.prec) * prec_factor

    # Tile over months
    air_temp = tf.tile(tf.expand_dims(air_temp_2d, 0), (_N_MONTHS, 1, 1))
    precip = tf.tile(tf.expand_dims(precip_2d, 0), (_N_MONTHS, 1, 1))

    # Optional cosine seasonal cycle on temperature (off by default)
    if bool(p.cosine_yearly_cycle_temp):
        months = np.arange(_N_MONTHS, dtype=np.float32)
        if bool(p.southern_hemisphere_climate):
            seasonal = p.cosine_yearly_cycle_amplitude * np.cos(
                2.0 * np.pi * months / _N_MONTHS
            )
        else:
            seasonal = p.cosine_yearly_cycle_amplitude * np.cos(
                2.0 * np.pi * (months - 6.0) / _N_MONTHS
            )
        seasonal = tf.reshape(tf.constant(seasonal, dtype=tf.float32), (_N_MONTHS, 1, 1))
        air_temp = air_temp + seasonal

    state.air_temp.assign(air_temp)
    state.precipitation.assign(precip)
