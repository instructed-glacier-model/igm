#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np 
import tensorflow as tf 

from igm.processes.iceflow.vert_disc import compute_levels
 
@tf.function()
def compute_average_velocity_twolayers_tf(U, V):

    Um = (U[:, :, 1:, 1:] + U[:, :, 1:, :-1] + U[:, :, :-1, 1:] + U[:, :, :-1, :-1]) / 4
    Vm = (V[:, :, 1:, 1:] + V[:, :, 1:, :-1] + V[:, :, :-1, 1:] + V[:, :, :-1, :-1]) / 4

    return Um, Vm

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
    return (B[:, 1:] + B[:, :-1]) / 2


def stag4(B):
    return (B[:, 1:, 1:] + B[:, 1:, :-1] + B[:, :-1, 1:] + B[:, :-1, :-1]) / 4


def stag4b(B):
    return (
        B[:, :, 1:, 1:] + B[:, :, 1:, :-1] + B[:, :, :-1, 1:] + B[:, :, :-1, :-1]
    ) / 4


def stag8(B):
    return (
        B[:, 1:, 1:, 1:]
        + B[:, 1:, 1:, :-1]
        + B[:, 1:, :-1, 1:]
        + B[:, 1:, :-1, :-1]
        + B[:, :-1, 1:, 1:]
        + B[:, :-1, 1:, :-1]
        + B[:, :-1, :-1, 1:]
        + B[:, :-1, :-1, :-1]
    ) / 8

tf.function()
def get_dz_COND(thk, Nz, vert_spacing):

    # Vertical discretization
    dz = tf.expand_dims(stag4(thk), axis=0)
    if Nz > 1:
        levels = compute_levels(Nz, vert_spacing)
        temd = levels[1:] - levels[:-1]
        dz = tf.expand_dims(tf.expand_dims(temd,axis=-1),axis=-1) * dz

    COND = (
        (thk[:, 1:, 1:] > 0)
        & (thk[:, 1:, :-1] > 0)
        & (thk[:, :-1, 1:] > 0)
        & (thk[:, :-1, :-1] > 0)
    )
    COND = tf.expand_dims(COND, axis=1)

    return dz, COND

def gauss_points_and_weigths(ord_gauss):

    if ord_gauss == 3:
        n = np.array([0.11270, 0.5,     0.88730], dtype=np.float32)
        w = np.array([0.27778, 0.44444, 0.27778], dtype=np.float32)
    elif ord_gauss == 5:
        n = np.array([0.04691, 0.23077, 0.5,     0.76923, 0.95309], dtype=np.float32)
        w = np.array([0.11847, 0.23932, 0.28444, 0.23932, 0.11847], dtype=np.float32)
    elif ord_gauss == 7:
        n = np.array([0.025446, 0.129234, 0.297078,      0.5, 0.702922, 0.870766, 0.974554], dtype=np.float32)
        w = np.array([0.064742, 0.139852, 0.190915, 0.208979, 0.190915, 0.139852, 0.064742], dtype=np.float32)

    return n, w 