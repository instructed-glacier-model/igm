#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np 
import tensorflow as tf 

# Shape of levels is (Nz,)
@tf.function()
def compute_levels(Nz, vert_spacing):
    zeta = tf.cast(tf.range(Nz) / (Nz - 1), "float32")
    return (zeta / vert_spacing) * (1.0 + (vert_spacing - 1.0) * zeta)

# Shape of dz is (Nz-1, Ny, Nx)
@tf.function()
def compute_dz(thk, levels):
    Nz = levels.shape[0]
    if Nz > 1:
        ddz = levels[1:] - levels[:-1]
        return thk[None, ...] * ddz[..., None, None]
    else:
        return thk[None, ...]
    
# Shape of dz is (Nz-1, Ny, Nx)
@tf.function()
def compute_zeta_dzeta(levels):
    Nz = levels.shape[0]
    if Nz > 1:
        zeta = (levels[1:] + levels[:-1]) / 2
        dzeta = levels[1:] - levels[:-1]
        return zeta, dzeta
    else: 
        return 0.5 * tf.ones((1), dtype=tf.float32), \
                     tf.ones((1), dtype=tf.float32)

# Shape of depth is (Nz, Ny, Nx)
@tf.function()
def compute_depth(dz):
    D = tf.concat([dz, tf.zeros((1, dz.shape[1], dz.shape[2]))], axis=0)
    return tf.math.cumsum(D, axis=0, reverse=True)
 
def define_vertical_weight(Nz, vert_spacing):
    zeta = tf.cast(tf.range(Nz+1) / Nz, "float32")
    weight = (zeta / vert_spacing) * (1.0 + (vert_spacing - 1.0) * zeta)
    weight = tf.Variable(weight[1:] - weight[:-1], dtype=tf.float32, trainable=False)
    return weight[..., None, None]
