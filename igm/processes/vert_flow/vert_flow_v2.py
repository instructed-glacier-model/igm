#!/usr/bin/env python3

"""
# Copyright (C) 2021-2025 IGM authors 
Published under the GNU GPL (Version 3), check at the LICENSE file
"""

import numpy as np
import tensorflow as tf

from igm.utils.gradient.compute_gradient_tf import compute_gradient_tf
from igm.utils.gradient.compute_divflux import compute_divflux

from igm.processes.iceflow.vert_disc import compute_levels, compute_dz
 

def compute_vertical_velocity_kinematic_v2(cfg, state):

    # implementation CMS

    levels = compute_levels(cfg.processes.iceflow.numerics.Nz, 
                            cfg.processes.iceflow.numerics.vert_spacing)
    dz = compute_dz(state.thk, levels)

    W = compute_w_kinematic_tf(state.U,state.V,state.topg,state.thk,dz,state.dx,state.vert_weight)

    return W

@tf.function()
def compute_w_kinematic_tf(U,V,topg,thk,dz,dx,vert_weight):
    cumdzCM = tf.concat([tf.expand_dims(tf.zeros_like(thk),axis=0), tf.cumsum(dz,axis=0)], axis=0)
    sxCM, syCM = compute_gradient_layers_tf(topg+ cumdzCM, dx, dx)
    ubCM =  tf.cumsum(vert_weight * U, axis=0)  / tf.cumsum(vert_weight, axis=0)
    vbCM =  tf.cumsum(vert_weight * V, axis=0)  / tf.cumsum(vert_weight, axis=0)
    divCM = compute_divflux_layers(ubCM, vbCM, cumdzCM,dx,dx)
    W =  - divCM + U * sxCM + V * syCM
    return W


def compute_vertical_velocity_incompressibility_v2(cfg, state):

    # implementation CMS
    levels = compute_levels(cfg.processes.iceflow.numerics.Nz, 
                            cfg.processes.iceflow.numerics.vert_spacing)
    dz = compute_dz(state.thk, levels)

    dz = tf.concat([tf.expand_dims(tf.zeros_like(state.thk),0),dz],axis=0)
    Z = tf.cumsum(dz) + state.topg
    sloptopgx, sloptopgy = compute_gradient_tf(state.topg,state.dX,state.dX)

    dudx = gradx_non_flat_layers_tf(state.U,state.dX,Z,state.vert_weight,state.thk) 
    dvdy = grady_non_flat_layers_tf(state.V,state.dX,Z,state.vert_weight,state.thk) 

    intdwdz = tf.cumsum(dz * (dudx + dvdy), axis=0)
    W =   sloptopgx * state.U[0] + sloptopgy * state.V[0] - intdwdz

    return W 



@tf.function()
def compute_gradient_layers_tf(s, dx, dy):
    """
    compute spatial 2D gradient on each layer of a given 3D field
    """

    EX = tf.concat(
        [
            1.5 * s[:,:, 0:1] - 0.5 * s[:,:, 1:2],
            0.5 * s[:,:, :-1] + 0.5 * s[:,:, 1:],
            1.5 * s[:,:, -1:] - 0.5 * s[:,:, -2:-1],
        ],
        2,
    )
    diffx = (EX[:,:, 1:] - EX[:,:, :-1]) / dx

    EY = tf.concat(
        [
            1.5 * s[:,0:1, :] - 0.5 * s[:,1:2, :],
            0.5 * s[:,:-1, :] + 0.5 * s[:,1:, :],
            1.5 * s[:,-1:, :] - 0.5 * s[:,-2:-1, :],
        ],
        1,
    )
    diffy = (EY[:,1:, :] - EY[:,:-1, :]) / dy

    return diffx, diffy

@tf.function()
def compute_gradx_layers_tf(s, dx):
    """
    compute spatial 2D gradient on each layer of a given 3D field along the x direction
    """

    EX = tf.concat(
        [
            1.5 * s[:,:, 0:1] - 0.5 * s[:,:, 1:2],
            0.5 * s[:,:, :-1] + 0.5 * s[:,:, 1:],
            1.5 * s[:,:, -1:] - 0.5 * s[:,:, -2:-1],
        ],
        2,
    )
    diffx = (EX[:,:, 1:] - EX[:,:, :-1]) / dx

    return diffx

@tf.function()
def compute_grady_layers_tf(s, dy):
    """
    compute spatial 2D gradient on each layer of a given 3D field along the direction y
    """

    EY = tf.concat(
        [
            1.5 * s[:,0:1, :] - 0.5 * s[:,1:2, :],
            0.5 * s[:,:-1, :] + 0.5 * s[:,1:, :],
            1.5 * s[:,-1:, :] - 0.5 * s[:,-2:-1, :],
        ],
        1,
    )
    diffy = (EY[:,1:, :] - EY[:,:-1, :]) / dy

    return diffy

@tf.function()
def compute_divflux_layers(u, v, h, dx, dy):
    '''
    Compute the divergence on each layer of a 3D field
    u and v are the vertical average velocity (integral along z from the bedrock to the given layer,
    divided by the height between the bedrock and the layer)
    '''
    #derivatives computed with centered method
    Qx = u * h  
    Qy = v * h  
    
    dQx_x = tf.concat(
        [Qx[:,:, 0:1], 0.5 * (Qx[:,:, :-1] + Qx[:,:, 1:]), Qx[:,:, -1:]], 2
    ) 
    
    dQy_y = tf.concat(
        [Qy[:,0:1, :], 0.5 * (Qy[:,:-1, :] + Qy[:,1:, :]), Qy[:,-1:, :]], 1
    ) 

    gradQx = (dQx_x[:,:, 1:] - dQx_x[:,:, :-1]) / dx
    gradQy = (dQy_y[:,1:, :] - dQy_y[:,:-1, :]) / dy

    return gradQx + gradQy


@tf.function()
def gradient_non_flat_layers_tf(U,dX,dY,Z,vert_weight,thk):
    '''
    U and Z are 3D fields
    dX and dY are scalars
    '''
    flat_gradx, flat_grady = compute_gradient_layers_tf(U,dX,dY)
    sx, sy = compute_gradient_layers_tf(Z, dX, dY) #Z = topg+ cumdzCM (cf kinematic)
    
    U = tf.concat([U[0:1,:, :], 0.5 * (U[:-1,:, :] + U[1:,:, :]), U[-1:,:, :]], 0)
    dUdz = tf.where(thk>0, (U[1:, :, :] - U[:-1, :, :]) / (vert_weight*thk), 0)
    
    gradx = flat_gradx - sx*dUdz
    grady = flat_grady - sy*dUdz

    return gradx, grady

@tf.function()
def gradx_non_flat_layers_tf(U,dX,Z,vert_weight,thk):
    '''
    U and Z are 3D fields
    dX and dY are scalars
    '''
    flat_gradx = compute_gradx_layers_tf(U,dX)
    sx = compute_gradx_layers_tf(Z, dX) #Z = topg+ cumdzCM (cf kinematic)
    
    U = tf.concat([U[0:1,:, :], 0.5 * (U[:-1,:, :] + U[1:,:, :]), U[-1:,:, :]], 0)
    dUdz = tf.where(thk>0, (U[1:, :, :] - U[:-1, :, :]) / (vert_weight*thk), 0)
    
    gradx = flat_gradx - sx*dUdz


    return gradx

@tf.function()
def grady_non_flat_layers_tf(U,dY,Z,vert_weight,thk):
    '''
    U and Z are 3D fields
    dX and dY are scalars
    '''
    flat_grady = compute_grady_layers_tf(U,dY)
    sy = compute_grady_layers_tf(Z,dY) #Z = topg+ cumdzCM (cf kinematic)
    
    U = tf.concat([U[0:1,:, :], 0.5 * (U[:-1,:, :] + U[1:,:, :]), U[-1:,:, :]], 0)
    dUdz = tf.where(thk>0, (U[1:, :, :] - U[:-1, :, :]) / (vert_weight*thk), 0)
    

    grady = flat_grady - sy*dUdz

    return grady


@tf.function()
def compute_divflux_d(u, v, h, dx, dy):
    
    #derivatives computed with centered method
    Qx = u * h  
    Qy = v * h  
    
    dQx_x = tf.concat(
        [Qx[:, 0:1], 0.5 * (Qx[:, :-1] + Qx[:, 1:]), Qx[:, -1:]], 1
    ) 
    
    dQy_y = tf.concat(
        [Qy[0:1, :], 0.5 * (Qy[:-1, :] + Qy[1:, :]), Qy[-1:, :]], 0
    ) 

    gradQx = (dQx_x[:, 1:] - dQx_x[:, :-1]) / dx
    gradQy = (dQy_y[1:, :] - dQy_y[:-1, :]) / dy

    return gradQx + gradQy

def compute_vertical_velocity_twolayers(cfg, state):
 
    sloptopgx, sloptopgy  = compute_gradient_tf(state.topg, state.dx, state.dx)

    slopusurfx,slopusurfy = compute_gradient_tf(state.usurf, state.dx, state.dx)

    div = compute_divflux_d(state.ubar, state.vbar, state.thk, state.dx, state.dx)

    wbase =  state.U[0] * sloptopgx + state.V[0] * sloptopgy

    wsurf = state.U[-1] * slopusurfx + state.V[-1] * slopusurfy - div[-1]

    return tf.stack([wbase,wsurf],axis=0)


