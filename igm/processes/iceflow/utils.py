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
    if vert_basis.lower() in ["lagrange","sia"]:
        return U[...,0,:,:]
    elif vert_basis.lower() == "legendre":
        pm = tf.pow(-1.0, tf.range(U.shape[-3], dtype=tf.float32))
        return tf.tensordot(pm, U, axes=[[0], [-3]]) 

@tf.function(jit_compile=True)
def get_velbase(U, V, vert_basis):
    return get_velbase_1(U, vert_basis), get_velbase_1(V, vert_basis)

def get_velsurf_1(U, vert_basis):
    if vert_basis.lower() in ["lagrange","sia"]:
        return U[...,-1,:,:]
    elif vert_basis.lower() == "legendre":
        pm = tf.pow(1.0, tf.range(U.shape[-3], dtype=tf.float32))
        return tf.tensordot(pm, U, axes=[[0], [-3]])

@tf.function(jit_compile=True)
def get_velsurf(U, V, vert_basis):
    return get_velsurf_1(U, vert_basis), get_velsurf_1(V, vert_basis)

def get_velbar_1(U, vert_weight, vert_basis):
    if vert_basis.lower() == "lagrange":
        return tf.reduce_sum(U * vert_weight, axis=-3)
    elif vert_basis.lower() == "legendre":
        return U[...,0,:,:]
    elif vert_basis.lower() == "sia":
        return U[...,0,:,:]+0.8*(U[...,-1,:,:]-U[...,0,:,:])

@tf.function(jit_compile=True)
def get_velbar(U, V, vert_weight, vert_basis):
    return get_velbar_1(U, vert_weight, vert_basis), \
           get_velbar_1(V, vert_weight, vert_basis)

def compute_PAD(multiple_window_size,Nx,Ny):

    # In case of a U-net, must make sure the I/O size is multiple of 2**N
    if multiple_window_size > 0:
        NNy = multiple_window_size * math.ceil(
            Ny / multiple_window_size
        )
        NNx = multiple_window_size * math.ceil(
            Nx / multiple_window_size
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

@tf.function(jit_compile=True)
def Y_to_UV(Nz, Y):

    U = tf.experimental.numpy.moveaxis(Y[..., :Nz], [-1], [1])
    V = tf.experimental.numpy.moveaxis(Y[..., Nz:], [-1], [1])

    return U, V

def UV_to_Y(cfg, U, V):
    UU = tf.experimental.numpy.moveaxis(U, [0], [-1])
    VV = tf.experimental.numpy.moveaxis(V, [0], [-1])

    return tf.concat([UU, VV], axis=-1)[None,...]

@tf.function(jit_compile=True)
def fieldin_to_X_2d(fieldin):

    return tf.expand_dims(fieldin, axis=0)


def fieldin_to_X_3d(dim_arrhenius, fieldin):
    
    return fieldin

from typing import List

@tf.function(jit_compile=True)
def X_to_fieldin(X: tf.Tensor, fieldin_names: List, dim_arrhenius: int, Nz: int):

    thk = X[..., 0]
    usurf = X[..., 1]
    
    if dim_arrhenius == 3:
        arrhenius = tf.experimental.numpy.moveaxis(X[..., 2 : 2 + Nz], [-1], [1])
        slidingco = X[..., 2 + Nz]
        dX = X[..., 3 + Nz]
    elif dim_arrhenius == 2:
        arrhenius = X[..., 2]
        slidingco = X[..., 3]
        dX = X[..., 4]
    else:
        raise ValueError("dim_arrhenius must be 2 or 3") # issue inside of jit?
    
    return dict(
        thk=thk,
        usurf=usurf,
        arrhenius=arrhenius,
        slidingco=slidingco,
        dX=dX
    )

@tf.function(jit_compile=True)
def boundvel(velbar_mag, VEL, force_max_velbar):
    return tf.where(velbar_mag >= force_max_velbar, force_max_velbar * (VEL / velbar_mag), VEL)

@tf.function(jit_compile=True)
def clip_max_velbar(U, V, force_max_velbar, vert_basis, vert_weight):

    if vert_basis.lower() in ["lagrange","sia"]:
        velbar_mag = getmag(U, V)
        U_clipped = boundvel(velbar_mag, U, force_max_velbar)
        V_clipped = boundvel(velbar_mag, V, force_max_velbar)

    elif vert_basis.lower() == "legendre":
        velbar_mag = getmag(*get_velbar(U, V, \
                                        vert_weight, vert_basis))
        uvelbar = boundvel(velbar_mag, U[0], force_max_velbar)
        vvelbar = boundvel(velbar_mag, V[0], force_max_velbar)
        U_clipped = tf.concat([uvelbar[None,...] , U[1:]], axis=0)
        V_clipped = tf.concat([vvelbar[None,...] , V[1:]], axis=0)
        
    else:
        raise ValueError("Unknown vertical basis: " + vert_basis)
    
    return U_clipped, V_clipped

def split_into_patches(X, nbmax, split_patch_method):
    """
    This function splits the input tensor into patches of size nbmax x nbmax.
    The patches are then stacked together to form a new tensor.
    If stack along axis 0, the adata will be streammed in a sequential way
    If stack along axis 1, the adata will be streammed in a parallel way by baches
    """
    XX = []
    ny = X.shape[1]
    nx = X.shape[2]
    sy = ny // nbmax + 1
    sx = nx // nbmax + 1
    ly = int(ny / sy)
    lx = int(nx / sx)

    for i in range(sx):
        for j in range(sy):
            #            if tf.reduce_max(X[:, j * ly : (j + 1) * ly, i * lx : (i + 1) * lx, :]) > 0:
            XX.append(X[:, j * ly : (j + 1) * ly, i * lx : (i + 1) * lx, :])

    if split_patch_method.lower() == "sequential":
        XXX = tf.stack(XX, axis=0)
    elif split_patch_method.lower() == "parallel":
        XXX = tf.expand_dims(tf.concat(XX, axis=0), axis=0)

    return XXX


def pertubate_X(cfg, X):

    XX = [X]

    for i, f in enumerate(cfg.processes.iceflow.emulator.fieldin):

        vec = [tf.ones_like(X[:, :, :, i]) * (i == j) for j in range(X.shape[3])]
        vec = tf.stack(vec, axis=-1)

        if hasattr(cfg.processes, "data_assimilation"):
            if f in cfg.processes.data_assimilation.control_list:
                XX.append(X + X * vec * 0.2)
                XX.append(X - X * vec * 0.2)
        else:
            if f in ["thk", "usurf"]:
                XX.append(X + X * vec * 0.2)
                XX.append(X - X * vec * 0.2)

    return tf.concat(XX, axis=0)

from typing import Tuple, Dict
from omegaconf import DictConfig
class TrainingParams(tf.experimental.ExtensionType):
    lr_decay: float
    Nx: int
    Ny: int
    Nz: int
    iz: int
    multiple_window_size: int
    framesizemax: int
    split_patch_method: str
    arrhenius_dimension: int
    staggered_grid: int
    fieldin_names: Tuple[str, ...]
    print_cost: bool

def get_emulator_data(state, nbit, lr) -> Dict:
    
    return dict({
        "iceflow_model_inference": state.iceflow_model_inference,
        "iceflow_model": state.iceflow_model,
        "sliding_law": state.iceflow.sliding_law,
        "energy_components": state.iceflow.energy_components,
        "opti_retrain": state.opti_retrain,
        "nbit": nbit,
        "effective_pressure": state.effective_pressure,
        "lr": lr,
})


def is_retrain(iteration, cfg: DictConfig) -> bool:

    # run_it = False
    if cfg.processes.iceflow.emulator.retrain_freq > 0:
        run_it = iteration % cfg.processes.iceflow.emulator.retrain_freq == 0

    warm_up = int(iteration <= cfg.processes.iceflow.emulator.warm_up_it)
    
    return run_it or warm_up

def prepare_data(cfg, fieldin, pertubate=False) -> Tuple[tf.Tensor, List[List[int]], int, int, int]:
    arrhenius_dimesnion = cfg.processes.iceflow.physics.dim_arrhenius
    iz = cfg.processes.iceflow.emulator.exclude_borders
    
    if arrhenius_dimesnion == 3:
        X = fieldin_to_X_3d(arrhenius_dimesnion, fieldin)
    elif arrhenius_dimesnion == 2:
        X = fieldin_to_X_2d(fieldin)

    if pertubate:
        X = pertubate_X(cfg, X)

    X = split_into_patches(
        X,
        cfg.processes.iceflow.emulator.framesizemax,
        cfg.processes.iceflow.emulator.split_patch_method,
    )
    
    Ny = X.shape[-3]
    Nx = X.shape[-2]

    padding = compute_PAD(cfg.processes.iceflow.emulator.network.multiple_window_size, Nx, Ny)
    
    return X, padding, Ny, Nx, iz