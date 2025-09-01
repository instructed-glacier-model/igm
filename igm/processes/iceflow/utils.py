#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np 
import tensorflow as tf 
import math
from tqdm import tqdm
import datetime

from igm.processes.iceflow.vert_disc import compute_levels
from igm.utils.math.getmag import getmag 

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
    
def get_velbase_1(U, vert_basis):
    if vert_basis in ["Lagrange","SIA"]:
        return U[...,0,:,:]
    elif vert_basis == "Legendre":
        pm = tf.pow(-1.0, tf.range(U.shape[-3], dtype=tf.float32))
        return tf.tensordot(pm, U, axes=[[0], [-3]]) 

def get_velbase(U, V, vert_basis):
    return get_velbase_1(U, vert_basis), get_velbase_1(V, vert_basis)

def get_velsurf_1(U, vert_basis):
    if vert_basis in ["Lagrange","SIA"]:
        return U[...,-1,:,:]
    elif vert_basis == "Legendre":
        pm = tf.pow(1.0, tf.range(U.shape[-3], dtype=tf.float32))
        return tf.tensordot(pm, U, axes=[[0], [-3]])
    
def get_velsurf(U, V, vert_basis):
    return get_velsurf_1(U, vert_basis), get_velsurf_1(V, vert_basis)

def get_velbar_1(U, vert_weight, vert_basis):
    if vert_basis == "Lagrange":
        return tf.reduce_sum(U * vert_weight, axis=-3)
    elif vert_basis == "Legendre":
        return U[...,0,:,:]
    elif vert_basis == "SIA":
        return U[...,0,:,:]+0.8*(U[...,-1,:,:]-U[...,0,:,:])

def get_velbar(U, V, vert_weight, vert_basis):
    return get_velbar_1(U, vert_weight, vert_basis), \
           get_velbar_1(V, vert_weight, vert_basis)

def compute_PAD(cfg,Nx,Ny):

    # In case of a U-net, must make sure the I/O size is multiple of 2**N
    if cfg.processes.iceflow.emulator.network.multiple_window_size > 0:
        NNy = cfg.processes.iceflow.emulator.network.multiple_window_size * math.ceil(
            Ny / cfg.processes.iceflow.emulator.network.multiple_window_size
        )
        NNx = cfg.processes.iceflow.emulator.network.multiple_window_size * math.ceil(
            Nx / cfg.processes.iceflow.emulator.network.multiple_window_size
        )
        return [[0, 0], [0, NNy - Ny], [0, NNx - Nx], [0, 0]]
    else:
        return [[0, 0], [0, 0], [0, 0], [0, 0]]
    
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
            

def print_info(state, it, cfg, energy_mean_list, velsurf_mag):
 
    if it % 100 == 1:
        if hasattr(state, "pbar_train"):
            state.pbar_train.close()
        state.pbar_train = tqdm(desc=f" Phys assim.", ascii=False, dynamic_ncols=True, bar_format="{desc} {postfix}")

    if hasattr(state, "pbar_train"):
        dic_postfix = {}
        dic_postfix["🕒"] = datetime.datetime.now().strftime("%H:%M:%S")
        dic_postfix["🔄"] = f"{it:04.0f}"
        for i, f in enumerate(cfg.processes.iceflow.physics.energy_components):
            dic_postfix[f] = f"{energy_mean_list[i]:06.3f}"
        dic_postfix["glen"] = f"{np.sum(energy_mean_list):06.3f}"
        dic_postfix["Max vel"] = f"{velsurf_mag:06.1f}"
#       dic_postfix["💾 GPU Mem (MB)"] = tf.config.experimental.get_memory_info("GPU:0")['current'] / 1024**2

        state.pbar_train.set_postfix(dic_postfix)
        state.pbar_train.update(1)

def Y_to_UV(cfg, Y):
    N = cfg.processes.iceflow.numerics.Nz

    U = tf.experimental.numpy.moveaxis(Y[..., :N], [-1], [1])
    V = tf.experimental.numpy.moveaxis(Y[..., N:], [-1], [1])

    return U, V

def UV_to_Y(cfg, U, V):
    UU = tf.experimental.numpy.moveaxis(U, [0], [-1])
    VV = tf.experimental.numpy.moveaxis(V, [0], [-1])

    return tf.concat([UU, VV], axis=-1)[None,...]

def fieldin_to_X(cfg, fieldin):
    X = []

    fieldin_dim = [0, 0, 1 * (cfg.processes.iceflow.physics.dim_arrhenius == 3), 0, 0]

    for f, s in zip(fieldin, fieldin_dim):
        if s == 0:
            X.append(tf.expand_dims(f, axis=-1))
        else:
            X.append(tf.experimental.numpy.moveaxis(f, [0], [-1]))

    return tf.expand_dims(tf.concat(X, axis=-1), axis=0)


def X_to_fieldin(cfg, X):
    i = 0

    fieldin_dim = [0, 0, 1 * (cfg.processes.iceflow.physics.dim_arrhenius == 3), 0, 0]

    fieldin = []

    for f, s in zip(cfg.processes.iceflow.emulator.fieldin, fieldin_dim):
        if s == 0:
            fieldin.append(X[..., i])
            i += 1
        else:
            fieldin.append(
                tf.experimental.numpy.moveaxis(
                    X[..., i : i + cfg.processes.iceflow.numerics.Nz], [-1], [1]
                )
            )
            i += cfg.processes.iceflow.numerics.Nz

    return fieldin

def boundvel(velbar_mag, VEL, force_max_velbar):
    return tf.where(velbar_mag >= force_max_velbar, force_max_velbar * (VEL / velbar_mag), VEL)

def force_max_velbar(cfg, state):

    force_max_velbar = cfg.processes.iceflow.force_max_velbar
    vert_basis = cfg.processes.iceflow.numerics.vert_basis

    if vert_basis in ["Lagrange","SIA"]:
        velbar_mag = getmag(state.U, state.V)
        state.U = boundvel(velbar_mag, state.U, force_max_velbar)
        state.V = boundvel(velbar_mag, state.V, force_max_velbar)

    elif vert_basis == "Legendre":
        velbar_mag = getmag(*get_velbar(state.U, state.V, \
                                        state.vert_weight, vert_basis))
        uvelbar = boundvel(velbar_mag, state.U[0], force_max_velbar)
        vvelbar = boundvel(velbar_mag, state.V[0], force_max_velbar)
        state.U = tf.concat([uvelbar[None,...] , state.U[1:]], axis=0)
        state.V = tf.concat([vvelbar[None,...] , state.V[1:]], axis=0)
        
    else:
        raise ValueError("Unknown vertical basis: " + cfg.processes.iceflow.numerics.vert_basis)