#!/usr/bin/env python3

"""
# Copyright (C) 2021-2025 IGM authors 
Published under the GNU GPL (Version 3), check at the LICENSE file
"""

import numpy as np
import tensorflow as tf
from igm.processes.iceflow.energy.cost_shear import compute_horizontal_derivatives
from igm.processes.iceflow.energy.utils import stag4h
from igm.utils.gradient.compute_gradient import compute_gradient
from igm.processes.iceflow.utils import get_velbase

def compute_vertical_velocity_legendre(cfg, state):
 
    sloptopgx, sloptopgy = compute_gradient(state.topg, state.dX, state.dX, staggered_grid=False)

    uvelbase, vvelbase = get_velbase(state.U, state.V, cfg.processes.iceflow.numerics.vert_basis) # Lagrange basis

    wvelbase = uvelbase * sloptopgx + vvelbase * sloptopgy # Lagrange basis

    dUdx, dVdx, dUdy, dVdy = compute_horizontal_derivatives(
                                state.U[None,...], state.V[None,...], 
                                state.dX[None,...], staggered_grid=False 
                                                           ) # Legendre basis

    # dUdx = tf.einsum('ij,bjkl->bikl', state.P, dUdx) 
    # dVdx = tf.einsum('ij,bjkl->bikl', state.P, dVdx)
    # dUdy = tf.einsum('ij,bjkl->bikl', state.P, dUdy)
    # dVdy = tf.einsum('ij,bjkl->bikl', state.P, dVdy)
    
    dUdx, dVdx, dUdy, dVdy = dUdx[0], dVdx[0], dUdy[0], dVdy[0]
 
    # Lagrange basis
    WLA = wvelbase[None,...] \
        - tf.tensordot(state.I, dUdx + dVdy, axes=[[1], [0]]) \
        * state.thk[None,...]  

    # Legendre basis
    return tf.einsum('ij,jkl->ikl', state.P, WLA * state.dzeta[0])