#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np 
import tensorflow as tf 

@tf.function()
def compute_levels(Nz, vert_spacing):
    zeta = tf.cast(tf.range(Nz) / (Nz - 1), "float32")
    return (zeta / vert_spacing) * (1.0 + (vert_spacing - 1.0) * zeta)

@tf.function()
def compute_dz(thk, levels):
    ddz = levels[1:] - levels[:-1]
    return tf.expand_dims(thk, 0) * tf.expand_dims(tf.expand_dims(ddz, -1), -1)

@tf.function()
def compute_depth(dz):
    D = tf.concat([dz, tf.zeros((1, dz.shape[1], dz.shape[2]))], axis=0)
    return tf.math.cumsum(D, axis=0, reverse=True)

# @tf.function()
# def vertically_discretize_tf(thk, Nz, vert_spacing):
#     levels = compute_levels(Nz, vert_spacing)
#     dz = compute_dz(thk, levels)
#     depth = compute_depth(dz)
#     return levels, dz, depth
 
def define_vertical_weight(Nz, vert_spacing):
    zeta = tf.cast(tf.range(Nz+1) / Nz, "float32")
    weight = (zeta / vert_spacing) * (1.0 + (vert_spacing - 1.0) * zeta)
    weight = tf.Variable(weight[1:] - weight[:-1], dtype=tf.float32, trainable=False)
    return tf.expand_dims(tf.expand_dims(weight, axis=-1), axis=-1)

