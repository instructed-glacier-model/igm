#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf 

def stag2(B):
    return (B[..., 1:] + B[..., :-1]) / 2

def stag4(B):
    return (
        B[..., 1:, 1:] + B[..., 1:, :-1] + B[..., :-1, 1:] + B[..., :-1, :-1]
    ) / 4

def stag8(B):
    if B.shape[-3] == 1:
        return tf.expand_dims(stag4(tf.squeeze(B, axis=-3)), axis=-3)
    else: 
        return (stag4(B[..., 1:, :, :]) + stag4(B[..., :-1, :, :])) / 2.0

def gauss_points_and_weights(ord_gauss):
    if ord_gauss == 3:
        n = tf.constant([0.11270, 0.5,     0.88730], dtype=tf.float32)
        w = tf.constant([0.27778, 0.44444, 0.27778], dtype=tf.float32)
    elif ord_gauss == 5:
        n = tf.constant([0.04691, 0.23077, 0.5,     0.76923, 0.95309], dtype=tf.float32)
        w = tf.constant([0.11847, 0.23932, 0.28444, 0.23932, 0.11847], dtype=tf.float32)
    elif ord_gauss == 7:
        n = tf.constant([0.025446, 0.129234, 0.297078, 0.5, 0.702922, 0.870766, 0.974554], dtype=tf.float32)
        w = tf.constant([0.064742, 0.139852, 0.190915, 0.208979, 0.190915, 0.139852, 0.064742], dtype=tf.float32)
    else:
        raise ValueError("Only Gauss orders 3, 5, and 7 are supported.")
    
    return n[None,:,None,None], w[None,:,None,None]

def psia(zeta,exp_glen):
    return ( 1 - (1 - zeta) ** (exp_glen + 1) )

def psiap(zeta,exp_glen):
    return (exp_glen + 1) * (1 - zeta) ** exp_glen
