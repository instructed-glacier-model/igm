#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import tensorflow as tf
from igm.processes.data_assimilation.utils import ave4

def regu_thk(cfg,state):
    if cfg.processes.data_assimilation.regularization.thk_version == 1:
        return regu_thk_v1(cfg,state)
    elif cfg.processes.data_assimilation.regularization.thk_version == 2:
        return regu_thk_v2(cfg,state)

#######################################

def regu_thk_v1(cfg,state):

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
        state.flowdirx = ave4(state.flowdirx)
        state.flowdiry = ave4(state.flowdiry)

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

#######################################

def regu_thk_v2(cfg,state):

    if cfg.processes.data_assimilation.regularization.to_regularize == 'topg':
        field = state.usurf - state.thk
    elif cfg.processes.data_assimilation.regularization.to_regularize == 'thk':
        field = state.thk

    # Comupute a rectification factor based on topg to favor 
    # deep ice in the ablation (against shallow in the accumulation) 
    if cfg.processes.data_assimilation.regularization.abl_acc_balance == 1:
        rect = 1
    else:
        ELA = np.percentile(state.usurf[state.usurf > 0], 66.7, method="linear")       
        r_acc = cfg.processes.data_assimilation.regularization.abl_acc_balance
        r_abl = 1/cfg.processes.data_assimilation.regularization.abl_acc_balance
        w_acc = 0.5 * (1.0 + tf.math.tanh((state.usurf - ELA) / 100.0)) 
        rect = (r_acc * w_acc + r_abl * (1.0 - w_acc))

    # Compute derivatives directly on 2D tensors
    kx, ky, kxx, kyy, kxy = _kernels(state.dx)           # Derivative stencils
    bx  = _conv2(field, kx);  by  = _conv2(field, ky)    # Derivatives of field
    bxx = _conv2(field, kxx); byy = _conv2(field, kyy); bxy = _conv2(field, kxy) 

    if cfg.processes.data_assimilation.optimization.sole_mask:
        bx  = tf.where( state.icemaskobs > 0.0, bx, 0.0)
        by  = tf.where( state.icemaskobs > 0.0, by, 0.0)
        bxx = tf.where( state.icemaskobs > 0.0, bxx, 0.0)
        byy = tf.where( state.icemaskobs > 0.0, byy, 0.0)
        bxy = tf.where( state.icemaskobs > 0.0, bxy, 0.0)

    if cfg.processes.data_assimilation.regularization.smooth_anisotropy_factor == 1:
        Dnn     = byy 
        Dtautau = bxx 
    else:
        tx, ty = state.flowdirx, state.flowdiry   # along-flow
        nx, ny = -ty, tx      
        Dtautau = (tx*tx)*bxx + 2.0*(tx*ty)*bxy + (ty*ty)*byy   # along-flow curvature
        Dnn     = (nx*nx)*bxx + 2.0*(nx*ny)*bxy + (ny*ny)*byy   # cross-flow curvature

    alpha = cfg.processes.data_assimilation.regularization.thk_2nd_der
    beta  = cfg.processes.data_assimilation.regularization.thk_1st_der
    anis_factor = cfg.processes.data_assimilation.regularization.smooth_anisotropy_factor
    gamma = cfg.processes.data_assimilation.regularization.convexity_weight
 
    if cfg.processes.data_assimilation.optimization.fix_opti_normalization_issue:
        J = alpha * rect * (tf.square(Dtautau) + anis_factor * tf.square(Dnn) ) \
          + beta * (tf.square(bx) + tf.square(by))  \
          - gamma * state.thk
        return tf.reduce_mean( J) 

    else:
        return alpha * (tf.nn.l2_loss( tf.square(Dtautau) + anis_factor * tf.square(Dnn) )) \
             + beta  * tf.nn.l2_loss( tf.square(bx) + tf.square(by) ) \
             - gamma * tf.reduce_sum(state.thk) 
    
def map_range(x,source_min, source_max, target_min, target_max): 
        return target_min + (x - source_min) / (source_max - source_min) * (target_max - target_min)
 
def _conv2(x, k):
    """Apply 2D convolution to a 2D tensor."""
    # Convert 2D tensor to 4D for conv2d: [batch, height, width, channels]
    x_4d = tf.expand_dims(tf.expand_dims(x, 0), -1)
    
    # Apply padding and convolution
    pad = [[0,0], [1,1], [1,1], [0,0]]  # 1 pixel on H and W
    x_pad = tf.pad(x_4d, pad, mode="SYMMETRIC")
    result = tf.nn.conv2d(x_pad, k, strides=1, padding='VALID')
    
    # Convert back to 2D
    return tf.squeeze(result, [0, 3])

def _kernels(dx: float):
    """3×3 central-difference stencils."""
    kx  = tf.constant([[0.,0.,0.],[-1.,0.,1.],[0.,0.,0.]], tf.float32) / (2.0*dx)
    ky  = tf.constant([[0.,-1.,0.],[0., 0.,0.],[0., 1.,0.]], tf.float32) / (2.0*dx)
    kxx = tf.constant([[0.,0.,0.],[1.,-2.,1.],[0.,0.,0.]], tf.float32) / (dx*dx)
    kyy = tf.constant([[0.,1.,0.],[0.,-2.,0.],[0.,1.,0.]], tf.float32) / (dx*dx)
    kxy = tf.constant([[ 1., 0.,-1.],[ 0., 0., 0.],[-1., 0., 1.]], tf.float32) / (4.0*dx*dx)
    expand = lambda K: tf.reshape(K, [3,3,1,1])
    return expand(kx), expand(ky), expand(kxx), expand(kyy), expand(kxy)
 
