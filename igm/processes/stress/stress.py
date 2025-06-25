#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 

from igm.processes.iceflow.vert_disc import compute_levels, compute_dz, compute_depth
 
def initialize(cfg, state):
    
    if "iceflow" not in cfg.processes:
        raise ValueError("The 'iceflow' module is required for the 'stress' module.")
    
    Ny, Nx = state.thk.shape

    state.sigma_xx = tf.Variable(tf.zeros_like(state.U), trainable=False)
    state.sigma_yy = tf.Variable(tf.zeros_like(state.U), trainable=False)
    state.sigma_zz = tf.Variable(tf.zeros_like(state.U), trainable=False)
    state.sigma_xy = tf.Variable(tf.zeros_like(state.U), trainable=False)
    state.sigma_xz = tf.Variable(tf.zeros_like(state.U), trainable=False)
    state.sigma_yz = tf.Variable(tf.zeros_like(state.U), trainable=False)
  
def update(cfg, state):
    if hasattr(state, "logger"):
        state.logger.info("Update STRESS at time : " + str(state.t.numpy()))
 
    # get the vertical discretization
    levels = compute_levels(
               cfg.processes.iceflow.numerics.Nz, 
               cfg.processes.iceflow.numerics.vert_spacing)
    dz = compute_dz(state.thk, levels)
 
    if cfg.processes.iceflow.physics.dim_arrhenius == 2:
       B = (tf.expand_dims(state.arrhenius, axis=0)) ** (-1.0 / cfg.processes.iceflow.physics.exp_glen) 
    else:
       B = state.arrhenius ** (-1.0 / cfg.processes.iceflow.physics.exp_glen)

    Exx, Eyy, Ezz, Exy, Exz, Eyz = compute_strainratetensor_tf(state.U, state.V, state.dx, dz)

    strainrate = 0.5 * ( Exx**2 + Exy**2 + Exz**2 + Exy**2 + Eyy**2 + Eyz**2 + Exz**2 + Eyz**2 + Ezz**2 ) ** 0.5

    mu = 0.5 * B * strainrate ** (1.0 /  cfg.processes.iceflow.physics.exp_glen - 1)

    state.sigma_xx = 2 * mu * Exx
    state.sigma_yy = 2 * mu * Eyy
    state.sigma_zz = 2 * mu * Ezz
    state.sigma_xy = 2 * mu * Exy
    state.sigma_xz = 2 * mu * Exz
    state.sigma_yz = 2 * mu * Eyz   
    
def finalize(cfg, state):
    pass


@tf.function()
def compute_strainratetensor_tf(U, V, dx, dz, thr):
 
    Ui = tf.pad(U[:, :, :], [[0, 0], [0, 0], [1, 1]], "SYMMETRIC")
    Uj = tf.pad(U[:, :, :], [[0, 0], [1, 1], [0, 0]], "SYMMETRIC")
    Uk = tf.pad(U[:, :, :], [[1, 1], [0, 0], [0, 0]], "SYMMETRIC")

    Vi = tf.pad(V[:, :, :], [[0, 0], [0, 0], [1, 1]], "SYMMETRIC")
    Vj = tf.pad(V[:, :, :], [[0, 0], [1, 1], [0, 0]], "SYMMETRIC")
    Vk = tf.pad(V[:, :, :], [[1, 1], [0, 0], [0, 0]], "SYMMETRIC")

    DZ2 = tf.concat([dz[0:1], dz[:-1] + dz[1:], dz[-1:]], axis=0)

    Exx = (Ui[:, :, 2:] - Ui[:, :, :-2]) / (2 * dx)
    Eyy = (Vj[:, 2:, :] - Vj[:, :-2, :]) / (2 * dx)
    Ezz = -Exx - Eyy

    Exy = 0.5 * (Vi[:, :, 2:] - Vi[:, :, :-2]) / (2 * dx) + 0.5 * (
        Uj[:, 2:, :] - Uj[:, :-2, :]
    ) / (2 * dx)
    Exz = 0.5 * (Uk[2:, :, :] - Uk[:-2, :, :]) / tf.maximum(DZ2, thr)
    Eyz = 0.5 * (Vk[2:, :, :] - Vk[:-2, :, :]) / tf.maximum(DZ2, thr)

    Exx = tf.where(DZ2 > 1, Exx, 0.0)
    Eyy = tf.where(DZ2 > 1, Eyy, 0.0)
    Ezz = tf.where(DZ2 > 1, Ezz, 0.0)
    Exy = tf.where(DZ2 > 1, Exy, 0.0)
    Exz = tf.where(DZ2 > 1, Exz, 0.0)
    Eyz = tf.where(DZ2 > 1, Eyz, 0.0)

    return Exx, Eyy, Ezz, Exy, Exz, Eyz


 