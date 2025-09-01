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

    uvelbase, vvelbase = get_velbase(state.U, state.V, cfg.processes.iceflow.numerics.vert_basis) 

    wvelbase = uvelbase * sloptopgx + vvelbase * sloptopgy # Lagrange basis

    dUdx, dVdx, dUdy, dVdy = compute_horizontal_derivatives(state.U, state.V, state.dX[0,0], staggered_grid=False) # Legendre basis
 
    # Lagrange basis
    WLA = wvelbase[None,...] \
        - tf.tensordot(state.Leg_I, dUdx + dVdy, axes=[[1], [0]]) \
        * state.thk[None,...]
    
    # Legendre basis
    return tf.einsum('ji,jkl->ikl', state.Leg_P, WLA * state.dzeta[:,None,None])