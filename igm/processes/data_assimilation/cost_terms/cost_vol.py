#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf

def cost_vol(cfg,state):

    ACT = state.icemaskobs > 0.5
    
    num_basins = int(tf.reduce_max(state.icemaskobs).numpy())
    ModVols = tf.experimental.numpy.copy(state.icemaskobs)
    
    for j in range(1,num_basins+1):
        ModVols = tf.where(ModVols==j,(tf.reduce_sum(tf.where(state.icemask==j,state.thk,0.0))*state.dx**2)/1e9,ModVols)

    cost = 0.5 * tf.reduce_mean(
           ( (state.volumes[ACT] - ModVols[ACT]) / state.volume_weights[ACT]  )** 2
    )
    return cost