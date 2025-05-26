#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file
 
import tensorflow as tf
 
def complete_data(state):
    """
    This function adds a postriori import fields such as X, Y, x, dx, ....
    """

    # define grids, i.e. state.X and state.Y has same shape as state.thk
    if not hasattr(state, "X"):
        state.X, state.Y = tf.meshgrid(state.x, state.y)

    # define cell spacing
    if not hasattr(state, "dx"):
        state.dx = state.x[1] - state.x[0]

    # define dX
    if not hasattr(state, "dX"):
        state.dX = tf.ones_like(state.X) * state.dx       
    
    # if thickness is not defined in the netcdf, then it is set to zero
    if not hasattr(state, "thk"):
        state.thk = tf.Variable(tf.zeros((state.y.shape[0], state.x.shape[0])), trainable=False)
        
    assert hasattr(state, "topg") | hasattr(state, "usurf")
    
    # case usurf defined, topg is not defined
    if not hasattr(state, "topg"):
        state.topg = tf.Variable(state.usurf - state.thk, trainable=False) 

    # case usurf not defined, topg is defined
    if not hasattr(state, "usurf"): 
        state.usurf = tf.Variable(state.topg + state.thk, trainable=False) 

