#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf

# def zeta_to_rhs(cfg, zeta):
#     return (zeta / cfg.processes.iceflow.numerics.vert_spacing) * (
#         1.0 + (cfg.processes.iceflow.numerics.vert_spacing - 1.0) * zeta
#     )

# get the position in the column
def rhs_to_zeta(vert_spacing, rhs):
    if vert_spacing == 1:
        zeta = rhs
    else:
        DET = tf.sqrt(1 + 4 * (vert_spacing - 1) * vert_spacing * rhs)
        zeta = (DET - 1) / (2 * (vert_spacing - 1))
 
    return zeta
 
def get_weights_lagrange(vert_spacing, Nz, particle_r):
    "This function gets the weight to extract the value of any field at a given position along the ice column"

    # rng_outer = srange("indices in weights", color="blue")
    zeta = rhs_to_zeta(vert_spacing, particle_r)  # get the position in the column
    I0 = tf.math.floor(zeta * (Nz - 1))

    I0 = tf.minimum(I0, Nz - 2)  # make sure to not reach the upper-most pt
    I1 = I0 + 1

    zeta0 = I0 / (Nz - 1)
    zeta1 = I1 / (Nz - 1)
    lamb = (zeta - zeta0) / (zeta1 - zeta0)

    ind0 = tf.stack([tf.cast(I0, tf.int64), tf.range(I0.shape[0], dtype=tf.int64)], axis=1)
    ind1 = tf.stack([tf.cast(I1, tf.int64), tf.range(I1.shape[0], dtype=tf.int64)], axis=1)
    
    weights = tf.zeros((Nz, particle_r.shape[0]))
    weights = tf.tensor_scatter_nd_add(
        weights, indices=ind0, updates=1 - lamb
    )
    weights = tf.tensor_scatter_nd_add(
        weights, indices=ind1, updates=lamb
    )

    return weights

def get_weights_legendre(zeta, order):

    x = 2.0 * zeta - 1.0 
 
    P = [tf.ones_like(x)]
    if order > 1:
        P.append(x)
    for k in range(2, order):
        Pk = ((2 * k - 1) * x * P[-1] - (k - 1) * P[-2]) / k
        P.append(Pk)
     
    return tf.stack(P, axis=-2)