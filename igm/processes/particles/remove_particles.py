#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
import numpy as np

def remove_particles_ablation(cfg, state):
 
    COND1 = (state.particle["r"] == 1)  

    I = tf.squeeze(tf.where(~(COND1)))

    if tf.size(I) > 0:
        for key in state.particle:
            state.particle[key] = tf.gather(state.particle[key], I)
    else:
        for key in state.particle:
            state.particle[key] = tf.Variable([], dtype=state.particle[key].dtype)




    

        