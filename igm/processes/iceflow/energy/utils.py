#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np 
import tensorflow as tf 

from igm.processes.iceflow.vert_disc import compute_levels

@tf.function()
def compute_gradient_stag(s, dX, dY):
    """
    compute spatial gradient, outcome on stagerred grid
    """

    E = 2.0 * (s[:, :, 1:] - s[:, :, :-1]) / (dX[:, :, 1:] + dX[:, :, :-1])
    diffx = 0.5 * (E[:, 1:, :] + E[:, :-1, :])

    EE = 2.0 * (s[:, 1:, :] - s[:, :-1, :]) / (dY[:, 1:, :] + dY[:, :-1, :])
    diffy = 0.5 * (EE[:, :, 1:] + EE[:, :, :-1])

    return diffx, diffy
 

def stag2(B):
    return (B[..., 1:] + B[..., :-1]) / 2


def stag4(B):
    return (
        B[..., 1:, 1:] + B[..., 1:, :-1] + B[..., :-1, 1:] + B[..., :-1, :-1]
    ) / 4

def stag8(B):
    return (
        B[..., 1:, 1:, 1:]
        + B[..., 1:, 1:, :-1]
        + B[..., 1:, :-1, 1:]
        + B[..., 1:, :-1, :-1]
        + B[..., :-1, 1:, 1:]
        + B[..., :-1, 1:, :-1]
        + B[..., :-1, :-1, 1:]
        + B[..., :-1, :-1, :-1]
    ) / 8

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
    
    return n, w

def psia(zeta,exp_glen):
    return ( 1 - (1 - zeta) ** (exp_glen + 1) )

def psiap(zeta,exp_glen):
    return (exp_glen + 1) * (1 - zeta) ** exp_glen
