#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from igm.processes.iceflow.energy.utils import stag4h, stag2v
from igm.utils.gradient.compute_gradient import compute_gradient
from igm.processes.iceflow.energy.cost_shear import compute_horizontal_derivatives, compute_vertical_derivatives, compute_srx

def cost_penalty(cfg, U, V, W, P, fieldin, vert_disc, staggered_grid):

    thk, usurf, arrhenius, slidingco, dX = fieldin
    zeta, dzeta, Leg_P, Leg_dPdz = vert_disc
    
    thr_ice_thk = cfg.processes.iceflow.physics.thr_ice_thk 
    vert_basis = cfg.processes.iceflow.numerics.vert_basis

    return _cost_penalty(U, V, W, P, thk, dX, zeta, dzeta, thr_ice_thk, staggered_grid, vert_basis)

@tf.function()
def center_to_staggered_faces_tf(u, v, w):
    """Convert center-based u,v to face-based MAC-style velocity"""
    u_pad1 = tf.pad(u, [[0, 0], [0, 0], [0, 0], [1, 0]], mode='SYMMETRIC')
    u_pad2 = tf.pad(u, [[0, 0], [0, 0], [0, 0], [0, 1]], mode='SYMMETRIC')
    u_face = 0.5 * (u_pad1 + u_pad2)  

    v_pad1 = tf.pad(v, [[0, 0], [0, 0], [1, 0], [0, 0]], mode='SYMMETRIC')
    v_pad2 = tf.pad(v, [[0, 0], [0, 0], [0, 1], [0, 0]], mode='SYMMETRIC')
    v_face = 0.5 * (v_pad1 + v_pad2) 

    w_pad1 = tf.pad(w, [[0, 0], [1, 0], [0, 0], [0, 0]], mode='SYMMETRIC')
    w_pad2 = tf.pad(w, [[0, 0], [0, 1], [0, 0], [0, 0]], mode='SYMMETRIC')
    w_face = 0.5 * (w_pad1 + w_pad2)

    return u_face, v_face, w_face

@tf.function()
def compute_mac_divergence_tf(u_face, v_face, w_face, dx, dy, thk, dzeta, thr):
    """Divergence from face-centered velocities to cell centers"""

    d_pad1 = tf.pad(dzeta, [[1, 0]], mode='SYMMETRIC')
    d_pad2 = tf.pad(dzeta, [[0, 1]], mode='SYMMETRIC')
    ddzeta = 0.5 * (d_pad1 + d_pad2)  
    
    div = (u_face[:,:,:,1:] - u_face[:,:,:,:-1]) / dx \
        + (v_face[:,:,1:,:] - v_face[:,:,:-1,:]) / dy \
        + (w_face[:,1:,:,:] - w_face[:,:-1,:,:]) / (ddzeta[None, :, None, None] * tf.maximum(thk, thr))
    return div
 
@tf.function()
def _cost_penalty_0(U, V, W, P, thk, dX, zeta, dzeta, thr_ice_thk, staggered_grid, vert_basis):
 
    dUdx, dVdx, dWdx, dUdy, dVdy, dWdy = compute_horizontal_derivatives(U, V, W, dX[0,0,0], staggered_grid) 

    # compute the horizontal average, these quantitites will be used for vertical derivatives
    if staggered_grid:
        U = stag4h(U)
        V = stag4h(V)
        W = stag4h(W)
        P = stag4h(P)
        thk = stag4h(thk)

    if vert_basis == "Lagrange":

        dUdx = stag2v(dUdx) 
        dVdx = stag2v(dVdx) 
        dWdx = stag2v(dWdx)
        dUdy = stag2v(dUdy) 
        dVdy = stag2v(dVdy)
        dWdy = stag2v(dWdy)
        P = stag2v(P)

        dUdz, dVdz, dWdz = compute_vertical_derivatives(U, V, W, thk, dzeta, thr=thr_ice_thk) 

    elif vert_basis == "Legendre":
        raise ValueError(f"Unknown vertical basis: {vert_basis}")
    
    elif vert_basis == "SIA":
        raise ValueError(f"Unknown vertical basis: {vert_basis}")
    
    else:
        raise ValueError(f"Unknown vertical basis: {vert_basis}")

    div = dUdx + dVdy + dWdz
  
    # Unit  Mpa m/y
    return  - thk * tf.reduce_sum( dzeta[None, :, None, None] * div * P , axis=1 ) \
            + 10**10 * thk * tf.reduce_sum( dzeta[None, :, None, None] * div**2 , axis=1 ) \
            + 10**10 * thk * tf.reduce_sum( dzeta[None, :, None, None] * P[:,-1]**2 , axis=1 ) 

 

@tf.function()
def _cost_penalty(U, V, W, P, thk, dX, zeta, dzeta, thr_ice_thk, staggered_grid, vert_basis):

    # dUdx, dVdx, dWdx, dUdy, dVdy, dWdy = compute_horizontal_derivatives(U, V, W, dX[0,0,0], staggered_grid)

    # compute the horizontal average, these quantitites will be used for vertical derivatives
    # if staggered_grid:
    #     U = stag4h(U)
    #     V = stag4h(V)
    #     W = stag4h(W)
    #     P = stag4h(P)
    #     thk = stag4h(thk)

    if vert_basis == "Lagrange":

        UF, VF, WF = center_to_staggered_faces_tf(U, V, W)

        div = compute_mac_divergence_tf(UF, VF, WF, dX[0,0,0], dX[0,0,0], thk, dzeta, thr=thr_ice_thk)
 

    elif vert_basis == "Legendre":
        raise ValueError(f"Unknown vertical basis: {vert_basis}")
    
    elif vert_basis == "SIA":
        raise ValueError(f"Unknown vertical basis: {vert_basis}")
    
    else:
        raise ValueError(f"Unknown vertical basis: {vert_basis}")


    div = stag2v(div)
    P = stag2v(P)
  
    # Unit  Mpa m/y
    return  - thk * tf.reduce_sum( dzeta[None, :, None, None] * div * P , axis=1 ) 
#            + 10**5 * thk * tf.reduce_sum( dzeta[None, :, None, None] * div**2 , axis=1 ) 
#            + 10**5 * thk * tf.reduce_sum( dzeta[None, :, None, None] * P[:,-1]**2 , axis=1 ) 
