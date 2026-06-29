#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from ..utils import create_density_matrix
from ..cook.infer_params_cook import infer_params_cook
 
def optimize_initialize(cfg, state):

    ###### PERFORM CHECKS PRIOR OPTIMIZATIONS

    # from scipy.ndimage import gaussian_filter
    # state.usurfobs = tf.Variable(gaussian_filter(state.usurfobs.numpy(), 3, mode="reflect"))
    # state.usurf    = tf.Variable(gaussian_filter(state.usurf.numpy(), 3, mode="reflect"))

    assert ("usurf" in cfg.assimilations.data_assimilation.cost_list) == ("usurf" in cfg.assimilations.data_assimilation.control_list)

    # make sure that there are least some profiles in thkobs
    if tf.reduce_all(tf.math.is_nan(state.thkobs)):
        if "thk" in cfg.assimilations.data_assimilation.cost_list:
            cfg.assimilations.data_assimilation.cost_list.remove("thk")

    ###### PREPARE DATA PRIOR OPTIMIZATIONS
 
    if "divfluxobs" in cfg.assimilations.data_assimilation.cost_list:
        if not hasattr(state, "divfluxobs"):
            state.divfluxobs = state.smb - state.dhdt

    if hasattr(state, "thkinit"):
        state.thk = state.thkinit
    else:
        state.thk = tf.zeros_like(state.thk)

    if cfg.assimilations.data_assimilation.optimization.init_zero_thk:
        state.thk = state.thk*0.0
        
    # this is a density matrix that will be used to weight the cost function
    if cfg.assimilations.data_assimilation.fitting.uniformize_thkobs:
        state.dens_thkobs = create_density_matrix(state.thkobs, kernel_size=5)
        state.dens_thkobs = tf.where(state.dens_thkobs>0, 1.0/state.dens_thkobs, 0.0)
        state.dens_thkobs = tf.where(tf.math.is_nan(state.thkobs),0.0,state.dens_thkobs)
        state.dens_thkobs = state.dens_thkobs / tf.reduce_mean(state.dens_thkobs[state.dens_thkobs>0])
    else:
        state.dens_thkobs = tf.ones_like(state.thkobs)
        
    # force zero friction control (slidingco or tau_ref) in the floating areas
    fric = state.da_friction
    setattr(state, fric, tf.where(state.icemaskobs == 2, 0.0, getattr(state, fric)))
    
    # this will infer values for slidingco and convexity weight based on the ice velocity and an empirical relationship from test glaciers with thickness profiles
    if cfg.assimilations.data_assimilation.cook.infer_params:
        #Because OGGM will index icemask from 0
        dummy = infer_params_cook(state, cfg)
        if tf.reduce_max(state.icemask).numpy() < 1:
            return
    
    if (int(tf.__version__.split(".")[1]) <= 10) | (int(tf.__version__.split(".")[1]) >= 16) :
        state.optimizer = tf.keras.optimizers.Adam(
            learning_rate=cfg.assimilations.data_assimilation.optimization.step_size,
            epsilon=cfg.assimilations.data_assimilation.optimization.optimizer_epsilon,
            clipnorm=cfg.assimilations.data_assimilation.optimization.optimizer_clipnorm
            )
    else:
        state.optimizer = tf.keras.optimizers.legacy.Adam(
            learning_rate=cfg.assimilations.data_assimilation.optimization.step_size,
            epsilon=cfg.assimilations.data_assimilation.optimization.optimizer_epsilon,
            clipnorm=cfg.assimilations.data_assimilation.optimization.optimizer_clipnorm
        )