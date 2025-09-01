#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import os
import tensorflow as tf
import json

def initialize(cfg, state):
    state.tlast_mb = tf.Variable(-1.0e5000)

    path_data = os.path.join(state.original_cwd,cfg.core.folder_data)
    path_RGI = os.path.join(path_data, cfg.inputs.oggm_shop.RGI_ID)

    # load the given parameters from the json file
    with open(os.path.join(path_RGI, "mb_calib.json"), "r") as json_file:
        jsonString = json_file.read()

    oggm_mb_calib = json.loads(jsonString)

    # ! Changed these variables into state attributes instead of params!
    state.thr_temp_snow = oggm_mb_calib["mb_global_params"]["temp_all_solid"]
    state.thr_temp_rain = oggm_mb_calib["mb_global_params"]["temp_all_liq"]
    state.temp_melt = oggm_mb_calib["mb_global_params"]["temp_melt"]
    state.melt_f = oggm_mb_calib["melt_f"]  # unit: mm water / (celcius day)


def update(cfg, state):
    #    mass balance forced by climate with accumulation and temperature-index melt model
    #    Input:  state.precipitation [Unit: kg * m^(-2) * y^(-1)]
    #            state.air_temp      [Unit: °C           ]
    #    Output  state.smb           [Unit: m ice eq. / y]

    #   This mass balance routine implements the surface mass balance model of OGGM

    # update smb each X years
    if (state.t - state.tlast_mb) >= cfg.processes.smb_oggm.update_freq:
        if hasattr(state, "logger"):
            state.logger.info(
                "Construct mass balance at time : " + str(state.t.numpy())
            )

        # keep solid precipitation when temperature < thr_temp_snow
        # with linear transition to 0 between thr_temp_snow and thr_temp_rain
        accumulation = tf.where(
            state.air_temp <= state.thr_temp_snow,
            state.precipitation,
            tf.where(
                state.air_temp >= state.thr_temp_rain,
                0.0,
                state.precipitation
                * (state.thr_temp_rain - state.air_temp)
                / (state.thr_temp_rain - state.thr_temp_snow),
            ),
        )

        accumulation /= accumulation.shape[
            0
        ]  # unit to [ kg * m^(-2) * y^(-1) ] -> [ kg * m^(-2) water ]

        accumulation /= cfg.processes.smb_oggm.wat_density  # unit [ m water ]

        ablation = state.melt_f * cfg.processes.smb_oggm.melt_enhancer * tf.clip_by_value(
            state.air_temp - state.temp_melt, 0, 10**10
        )  # unit: [ mm * day^(-1) water ]

        ablation *= 365.242198781 / 1000.0  # unit to [ m * y^(-1) water ]

        ablation /= ablation.shape[0]  # unit to [ m  water ]

        # sum accumulation and ablation over the year, and conversion to ice equivalent
        state.smb = tf.math.reduce_sum(accumulation - ablation, axis=0) * (
            cfg.processes.smb_oggm.wat_density / cfg.processes.smb_oggm.ice_density
        )

        if hasattr(state, "icemask"):
            state.smb = tf.where(
                (state.smb < 0) | (state.icemask > 0.5), state.smb, -10
            )

        state.tlast_mb.assign(state.t)


def finalize(cfg, state):
    pass
