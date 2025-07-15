#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import matplotlib.pyplot as plt
import tensorflow as tf

from igm.utils.math.getmag import getmag
from igm.utils.gradient.compute_gradient_tf import compute_gradient_tf

def initialize(cfg, state):
    pass

def update(cfg, state):
    slopsurfx, slopsurfy = compute_gradient_tf(state.usurf, state.dx, state.dx)

    slop = getmag(slopsurfx, slopsurfy)

    dirx = -cfg.processes.rockflow.flow_speed * tf.where(
        tf.not_equal(slop, 0), slopsurfx / slop, 1
    )
    diry = -cfg.processes.rockflow.flow_speed * tf.where(
        tf.not_equal(slop, 0), slopsurfy / slop, 1
    )

    thkexp = tf.repeat(tf.expand_dims(state.thk, axis=0), state.U.shape[0], axis=0)

    if cfg.processes.iceflow.numerics.vert_basis in ["Lagrange","SIA"]:
        state.U = tf.where(thkexp > 0, state.U, dirx)
        state.V = tf.where(thkexp > 0, state.V, diry)
    elif cfg.processes.iceflow.numerics.vert_basis == "Legendre":
        state.U = tf.where(thkexp > 0, state.U, 
                           tf.concat([dirx[None,...] , 0.0*state.U[1:]], axis=0))
        state.V = tf.where(thkexp > 0, state.V, 
                           tf.concat([diry[None,...] , 0.0*state.V[1:]], axis=0))

def finalize(cfg, state):
    pass
