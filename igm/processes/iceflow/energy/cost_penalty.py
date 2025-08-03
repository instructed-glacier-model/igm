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
def center_to_staggered_faces_tf(u, v):
    """Convert center-based u,v to face-based MAC-style velocity"""
    u_pad1 = tf.pad(u, [[0, 0], [0, 0], [1, 0], [1, 0]], mode='SYMMETRIC')
    u_pad2 = tf.pad(u, [[0, 0], [0, 0], [0, 0], [0, 1]], mode='SYMMETRIC')
    u_face = 0.5 * (u_pad1 + u_pad2)  # u on vertical faces (Nx+1, Ny)

    v_pad1 = tf.pad(v, [[0, 0], [0, 0], [1, 0], [0, 0]], mode='SYMMETRIC')
    v_pad2 = tf.pad(v, [[0, 0], [0, 0], [0, 1], [0, 0]], mode='SYMMETRIC')
    v_face = 0.5 * (v_pad1 + v_pad2)  # v on horizontal faces (Nx, Ny+1)

    return u_face, v_face

@tf.function()
def compute_mac_divergence_tf(u_face, v_face, dx, dy):
    """Divergence from face-centered velocities to cell centers"""
    div = (u_face[:,:,1:, :] - u_face[:,:,:-1, :]) / dx \
        + (v_face[:,:,:, 1:] - v_face[:,:,:, :-1]) / dy
    return div

@tf.function()
def _cost_penalty(U, V, W, P, thk, dX, zeta, dzeta, thr_ice_thk, staggered_grid, vert_basis):

#    UF, VF = center_to_staggered_faces_tf(U, V)
#    div = compute_mac_divergence_tf(UF, VF, dX[0,0,0], dX[0,0,0])

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
