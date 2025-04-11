#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np 
import tensorflow as tf 
import math

def initialize_iceflow_fields(cfg, state):

    # here we initialize variable parmaetrizing ice flow
    if not hasattr(state, "arrhenius"):
        if cfg.processes.iceflow.physics.dim_arrhenius == 3:
            state.arrhenius = \
                tf.ones((cfg.processes.iceflow.numerics.Nz, state.thk.shape[0], state.thk.shape[1])) \
                * cfg.processes.iceflow.physics.init_arrhenius * cfg.processes.iceflow.physics.enhancement_factor
        else:
            state.arrhenius = tf.ones_like(state.thk) * cfg.processes.iceflow.physics.init_arrhenius * cfg.processes.iceflow.physics.enhancement_factor

    if not hasattr(state, "slidingco"):
        state.slidingco = tf.ones_like(state.thk) * cfg.processes.iceflow.physics.init_slidingco

    # here we create a new velocity field
    if not hasattr(state, "U"):
        state.U = tf.zeros((cfg.processes.iceflow.numerics.Nz, state.thk.shape[0], state.thk.shape[1])) 
        state.V = tf.zeros((cfg.processes.iceflow.numerics.Nz, state.thk.shape[0], state.thk.shape[1])) 

def define_vertical_weight(cfg, state):
    """
    define_vertical_weight
    """

    zeta = np.arange(cfg.processes.iceflow.numerics.Nz + 1) / cfg.processes.iceflow.numerics.Nz
    weight = (zeta / cfg.processes.iceflow.numerics.vert_spacing) * (
        1.0 + (cfg.processes.iceflow.numerics.vert_spacing - 1.0) * zeta
    )
    weight = tf.Variable(weight[1:] - weight[:-1], dtype=tf.float32, trainable=False)
    state.vert_weight = tf.expand_dims(tf.expand_dims(weight, axis=-1), axis=-1)


def update_2d_iceflow_variables(cfg, state):
    state.uvelbase = state.U[0, :, :]
    state.vvelbase = state.V[0, :, :]
    state.ubar = tf.reduce_sum(state.U * state.vert_weight, axis=0)
    state.vbar = tf.reduce_sum(state.V * state.vert_weight, axis=0)
    state.uvelsurf = state.U[-1, :, :]
    state.vvelsurf = state.V[-1, :, :]

def compute_PAD(cfg,Nx,Ny):

    # In case of a U-net, must make sure the I/O size is multiple of 2**N
    if cfg.processes.iceflow.emulator.network.multiple_window_size > 0:
        NNy = cfg.processes.iceflow.emulator.network.multiple_window_size * math.ceil(
            Ny / cfg.processes.iceflow.emulator.network.multiple_window_size
        )
        NNx = cfg.processes.iceflow.emulator.network.multiple_window_size * math.ceil(
            Nx / cfg.processes.iceflow.emulator.network.multiple_window_size
        )
        return [[0, 0], [0, 0], [0, NNy - Ny], [0, NNx - Nx], [0, 0]]
    else:
        return [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
    

@tf.function()
def base_surf_to_U(uvelbase, uvelsurf, Nz, vert_spacing, iflo_exp_glen):

    zeta = tf.cast(tf.range(Nz) / (Nz - 1), "float32")
    levels = (zeta / vert_spacing) * (1.0 + (vert_spacing - 1.0) * zeta)
    levels = tf.expand_dims(tf.expand_dims(levels, axis=-1), axis=-1)

    return tf.expand_dims(uvelbase, axis=0) \
         + tf.expand_dims(uvelsurf - uvelbase, axis=0) \
         * ( 1 - (1 - levels) ** (iflo_exp_glen + 1) )

class EarlyStopping:
    def __init__(self, relative_min_delta=1e-3, patience=10):
        """
        Args:
            relative_min_delta (float): Minimum relative improvement required.
            patience (int): Number of consecutive iterations with no significant improvement allowed.
        """
        self.relative_min_delta = relative_min_delta
        self.patience = patience
        self.wait = 0
        self.best_loss = None

    def should_stop(self, current_loss):
        if self.best_loss is None:
            # Initialize best_loss during the first call
            self.best_loss = current_loss
            return False
        
        # Compute relative improvement
        relative_improvement = (self.best_loss - current_loss) / abs(self.best_loss)

        if relative_improvement > self.relative_min_delta:
            # Significant improvement: update best_loss and reset wait
            self.best_loss = current_loss
            self.wait = 0
            return False
        else:
            # No significant improvement: increment wait
            self.wait += 1
            if self.wait >= self.patience:
                return True

# def Y_to_UV(cfg, Y):
#     N = cfg.processes.iceflow.numerics.Nz

#     U = tf.experimental.numpy.moveaxis(Y[:, :, :, :N], [-1], [1])
#     V = tf.experimental.numpy.moveaxis(Y[:, :, :, N:], [-1], [1])

#     return U, V

# def UV_to_Y(cfg, U, V):
#     UU = tf.experimental.numpy.moveaxis(U, [0], [-1])
#     VV = tf.experimental.numpy.moveaxis(V, [0], [-1])
#     RR = tf.expand_dims(
#         tf.concat(
#             [UU, VV],
#             axis=-1,
#         ),
#         axis=0,
#     )

#     return RR

# def fieldin_to_X(cfg, fieldin):
#     X = []

#     fieldin_dim = [0, 0, 1 * (cfg.processes.iceflow.physics.dim_arrhenius == 3), 0, 0]

#     for f, s in zip(fieldin, fieldin_dim):
#         if s == 0:
#             X.append(tf.expand_dims(f, axis=-1))
#         else:
#             X.append(tf.experimental.numpy.moveaxis(f, [0], [-1]))

#     return tf.expand_dims(tf.concat(X, axis=-1), axis=0)


# def X_to_fieldin(cfg, X):
#     i = 0

#     fieldin_dim = [0, 0, 1 * (cfg.processes.iceflow.physics.dim_arrhenius == 3), 0, 0]

#     fieldin = []

#     for f, s in zip(cfg.processes.iceflow.emulator.fieldin, fieldin_dim):
#         if s == 0:
#             fieldin.append(X[:, :, :, i])
#             i += 1
#         else:
#             fieldin.append(
#                 tf.experimental.numpy.moveaxis(
#                     X[:, :, :, i : i + cfg.processes.iceflow.numerics.Nz], [-1], [1]
#                 )
#             )
#             i += cfg.processes.iceflow.numerics.Nz

#     return fieldin

########################################################


def Y_to_UV(cfg, Y):

    return Y[:, :, :, :, 0], Y[:, :, :, :, 1]


def UV_to_Y(cfg, U, V):
    
    UU = tf.expand_dims(U, axis=-1)
    VV = tf.expand_dims(V, axis=-1)
 
    return tf.expand_dims(tf.concat([UU,VV],axis=-1),axis=0)


def fieldin_to_X(cfg, fieldin):
    X = []

    fieldin_dim = [0, 0, 1 * (cfg.processes.iceflow.physics.dim_arrhenius == 3), 0, 0]

    for f, s in zip(fieldin, fieldin_dim):
        if s == 0:
            X.append(tf.expand_dims(f, axis=-1))
        else:
            X.append(tf.experimental.numpy.moveaxis(f, [0], [-1]))

    return tf.expand_dims(tf.expand_dims(tf.concat(X, axis=-1), axis=0), axis=0)


def X_to_fieldin(cfg, X):
    i = 0

    fieldin_dim = [0, 0, 1 * (cfg.processes.iceflow.physics.dim_arrhenius == 3), 0, 0]

    fieldin = []

    for f, s in zip(cfg.processes.iceflow.emulator.fieldin, fieldin_dim):
        if s == 0:
            fieldin.append(X[:, 0, :, :, i])
            i += 1
        else:
            fieldin.append(
                tf.experimental.numpy.moveaxis(
                    X[:, 0, :, :, i : i + cfg.processes.iceflow.numerics.Nz], [-1], [1]
                )
            )
            i += cfg.processes.iceflow.numerics.Nz

    return fieldin
