#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf

def misfit_usurf(cfg,state):

    ACT = state.icemaskobs > 0.5

    return 0.5 * tf.reduce_mean(
        (
            (state.usurf[ACT] - state.usurfobs[ACT])
            / cfg.processes.data_assimilation.fitting.usurfobs_std
        )
        ** 2
    )
