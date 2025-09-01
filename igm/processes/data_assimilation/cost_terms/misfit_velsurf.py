#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf 

def misfit_velsurf(cfg,state):    

    velsurf    = tf.stack([state.uvelsurf,    state.vvelsurf],    axis=-1) 
    velsurfobs = tf.stack([state.uvelsurfobs, state.vvelsurfobs], axis=-1)

    REL = tf.expand_dims( (tf.norm(velsurfobs,axis=-1) >= cfg.processes.data_assimilation.fitting.velsurfobs_thr ) , axis=-1)

    ACT = ~tf.math.is_nan(velsurfobs) 

    cost = 0.5 * tf.reduce_mean(
           ( (velsurfobs[ACT & REL] - velsurf[ACT & REL]) / cfg.processes.data_assimilation.fitting.velsurfobs_std  )** 2
    )

    if cfg.processes.data_assimilation.fitting.include_low_speed_term:

        # This terms penalize the cost function when the velocity is low
        # Reference : Inversion of basal friction in Antarctica using exact and incompleteadjoints of a higher-order model
        # M. Morlighem, H. Seroussi, E. Larour, and E. Rignot, JGR, 2013
        cost += 0.5 * 100 * tf.reduce_mean(
            tf.math.log( (tf.norm(velsurf[ACT],axis=-1)+1) / (tf.norm(velsurfobs[ACT],axis=-1)+1) )** 2
        )

    return cost
