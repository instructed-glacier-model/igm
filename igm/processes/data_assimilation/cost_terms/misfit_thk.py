#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import tensorflow as tf

def misfit_thk(cfg,state):

    ACT = ~tf.math.is_nan(state.thkobs)

    return 0.5 * tf.reduce_mean( state.dens_thkobs[ACT] * 
        ((state.thkobs[ACT] - state.thk[ACT]) / cfg.processes.data_assimilation.fitting.thkobs_std) ** 2
    )

