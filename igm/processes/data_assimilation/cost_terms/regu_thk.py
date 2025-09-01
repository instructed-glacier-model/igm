#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import tensorflow as tf

def regu_thk(cfg,state):

    areaicemask = tf.reduce_sum(tf.where(state.icemask>0.5,1.0,0.0))*state.dx**2

    # here we had factor 8*np.pi*0.04, which is equal to 1
    if cfg.processes.data_assimilation.cook.infer_params:
        gamma = tf.zeros_like(state.thk)
        gamma = state.convexity_weights * areaicemask**(cfg.processes.data_assimilation.regularization.convexity_power-2.0)
    else:
        gamma = cfg.processes.data_assimilation.regularization.convexity_weight * areaicemask**(cfg.processes.data_assimilation.regularization.convexity_power-2.0)

    if cfg.processes.data_assimilation.regularization.to_regularize == 'topg':
        field = state.usurf - state.thk
    elif cfg.processes.data_assimilation.regularization.to_regularize == 'thk':
        field = state.thk

    if cfg.processes.data_assimilation.regularization.smooth_anisotropy_factor == 1:
        dbdx = (field[:, 1:] - field[:, :-1])/state.dx
        dbdy = (field[1:, :] - field[:-1, :])/state.dx

        if cfg.processes.data_assimilation.optimization.sole_mask:
            dbdx = tf.where( (state.icemaskobs[:, 1:] > 0.5) & (state.icemaskobs[:, :-1] > 0.5) , dbdx, 0.0)
            dbdy = tf.where( (state.icemaskobs[1:, :] > 0.5) & (state.icemaskobs[:-1, :] > 0.5) , dbdy, 0.0)

        if cfg.processes.data_assimilation.optimization.fix_opti_normalization_issue:
            REGU_H = (cfg.processes.data_assimilation.regularization.thk) * 0.5 * (
                tf.math.reduce_mean(dbdx**2) + tf.math.reduce_mean(dbdy**2)
                - gamma * tf.math.reduce_mean(state.thk)
            )
        else:
            REGU_H = (cfg.processes.data_assimilation.regularization.thk) * (
                tf.nn.l2_loss(dbdx) + tf.nn.l2_loss(dbdy)
                - gamma * tf.math.reduce_sum(state.thk)
            )
    else:
        dbdx = (field[:, 1:] - field[:, :-1])/state.dx
        dbdx = (dbdx[1:, :] + dbdx[:-1, :]) / 2.0
        dbdy = (field[1:, :] - field[:-1, :])/state.dx
        dbdy = (dbdy[:, 1:] + dbdy[:, :-1]) / 2.0

        if cfg.processes.data_assimilation.optimization.sole_mask:
            MASK = (state.icemaskobs[1:, 1:] > 0.5) & (state.icemaskobs[1:, :-1] > 0.5) & (state.icemaskobs[:-1, 1:] > 0.5) & (state.icemaskobs[:-1, :-1] > 0.5)
            dbdx = tf.where( MASK, dbdx, 0.0)
            dbdy = tf.where( MASK, dbdy, 0.0)
 
        if cfg.processes.data_assimilation.optimization.fix_opti_normalization_issue:
            REGU_H = (cfg.processes.data_assimilation.regularization.thk) * 0.5 * (
                (1.0/np.sqrt(cfg.processes.data_assimilation.regularization.smooth_anisotropy_factor))
                * tf.math.reduce_mean((dbdx * state.flowdirx + dbdy * state.flowdiry)**2)
                + np.sqrt(cfg.processes.data_assimilation.regularization.smooth_anisotropy_factor)
                * tf.math.reduce_mean((dbdx * state.flowdiry - dbdy * state.flowdirx)**2)
                - tf.math.reduce_mean(gamma*state.thk)
            )
        else:
            REGU_H = (cfg.processes.data_assimilation.regularization.thk) * (
                (1.0/np.sqrt(cfg.processes.data_assimilation.regularization.smooth_anisotropy_factor))
                * tf.nn.l2_loss((dbdx * state.flowdirx + dbdy * state.flowdiry))
                + np.sqrt(cfg.processes.data_assimilation.regularization.smooth_anisotropy_factor)
                * tf.nn.l2_loss((dbdx * state.flowdiry - dbdy * state.flowdirx))
                - tf.math.reduce_sum(gamma*state.thk)
            )

    return REGU_H