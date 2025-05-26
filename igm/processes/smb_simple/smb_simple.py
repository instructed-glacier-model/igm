#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import tensorflow as tf 
from igm.utils.math.interp1d_tf import interp1d_tf 

def initialize(cfg, state):

    if cfg.processes.smb_simple.array == []:
        state.smbpar = np.loadtxt(
            cfg.processes.smb_simple.file,
            skiprows=1,
            dtype=np.float32,
        )
    else:
        state.smbpar = np.array(cfg.processes.smb_simple.array[1:]).astype(np.float32)

    state.tlast_mb = tf.Variable(-1.0e5000)


def update(cfg, state):

    # update smb each X years
    if (state.t - state.tlast_mb) >= cfg.processes.smb_simple.update_freq:
        if hasattr(state, "logger"):
            state.logger.info(
                "Construct mass balance at time : " + str(state.t.numpy())
            )

        # get the smb parameters at given time t
        gradabl = interp1d_tf(state.smbpar[:, 0], state.smbpar[:, 1], state.t)
        gradacc = interp1d_tf(state.smbpar[:, 0], state.smbpar[:, 2], state.t)
        ela = interp1d_tf(state.smbpar[:, 0], state.smbpar[:, 3], state.t)
        maxacc = interp1d_tf(state.smbpar[:, 0], state.smbpar[:, 4], state.t)

        # compute smb from glacier surface elevation and parameters
        state.smb = state.usurf - ela
        state.smb *= tf.where(tf.less(state.smb, 0), gradabl, gradacc)
        state.smb = tf.clip_by_value(state.smb, -100, maxacc)

        # if an icemask exists, then force negative smb aside to prevent leaks
        if hasattr(state, "icemask"):
            state.smb = tf.where(
                (state.smb < 0) | (state.icemask > 0.5), state.smb, -10
            )

        state.tlast_mb.assign(state.t)



def finalize(cfg, state):
    pass
