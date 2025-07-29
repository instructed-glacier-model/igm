#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import tensorflow as tf
import os

from igm.processes.iceflow.utils import (
    fieldin_to_X_2d,
    fieldin_to_X_3d,
    Y_to_UV,
    compute_PAD,
    print_info,
)
from igm.processes.iceflow.utils import (
    get_velbase,
    get_velsurf,
    get_velbar,
    clip_max_velbar,
)
from igm.processes.iceflow.energy.energy import iceflow_energy_XY
from igm.processes.iceflow.energy.sliding_laws.sliding_law import sliding_law_XY
from igm.processes.iceflow.energy.sliding_laws import sliding_law_XY, Weertman, WeertmanParams
from igm.processes.iceflow.emulate.neural_network import *
from igm.processes.iceflow.emulate import emulators
from igm.utils.math.getmag import getmag
import importlib_resources
import igm
import matplotlib.pyplot as plt
import matplotlib

from igm.processes.iceflow.energy import cost_floating, cost_shear, cost_gravity, cost_sliding_weertman


def initialize_iceflow_emulator(cfg, state):

    if (int(tf.__version__.split(".")[1]) <= 10) | (
        int(tf.__version__.split(".")[1]) >= 16
    ):
        state.opti_retrain = getattr(
            tf.keras.optimizers, cfg.processes.iceflow.emulator.optimizer
        )(
            learning_rate=cfg.processes.iceflow.emulator.lr,
            epsilon=cfg.processes.iceflow.emulator.optimizer_epsilon,
            clipnorm=cfg.processes.iceflow.emulator.optimizer_clipnorm,
        )
    else:
        state.opti_retrain = getattr(
            tf.keras.optimizers.legacy, cfg.processes.iceflow.emulator.optimizer
        )(
            learning_rate=cfg.processes.iceflow.emulator.lr,
            epsilon=cfg.processes.iceflow.emulator.optimizer_epsilon,
            clipnorm=cfg.processes.iceflow.emulator.optimizer_clipnorm,
        )

    L = (cfg.processes.iceflow.numerics.vert_basis=="Legendre")*'e' + \
        (not cfg.processes.iceflow.numerics.vert_basis=="Legendre")*'a'

    direct_name = (
        "pinnbp"
        + "_"
        + str(cfg.processes.iceflow.numerics.Nz)
        + "_"
        + str(int(cfg.processes.iceflow.numerics.vert_spacing))
        + "_"
    )
    direct_name += (
        cfg.processes.iceflow.emulator.network.architecture
        + "_"
        + str(cfg.processes.iceflow.emulator.network.nb_layers)
        + "_"
        + str(cfg.processes.iceflow.emulator.network.nb_out_filter)
        + "_"
    )
    direct_name += str(cfg.processes.iceflow.physics.dim_arrhenius) + "_" + str(int(1))

    if cfg.processes.iceflow.emulator.pretrained:
        dirpath = ''
        if cfg.processes.iceflow.emulator.name == "":
            if os.path.exists(
                importlib_resources.files(emulators).joinpath(direct_name)
            ):
                dirpath = importlib_resources.files(emulators).joinpath(direct_name)
                print("Found pretrained emulator in the igm package: " + direct_name)
            else:
                print("No pretrained emulator found in the igm package")
        else:
            dirpath = os.path.join(
                state.original_cwd, cfg.processes.iceflow.emulator.name
            )
            if os.path.exists(dirpath):
                print(
                    "----------------------------------> Found pretrained emulator: "
                    + cfg.processes.iceflow.emulator.name
                )
            else:
                print(
                    "----------------------------------> No pretrained emulator found "
                )

        fieldin = []
        fid = open(os.path.join(dirpath, "fieldin.dat"), "r")
        for fileline in fid:
            part = fileline.split()
            fieldin.append(part[0])
        fid.close()
        assert cfg.processes.iceflow.emulator.fieldin == fieldin
        state.iceflow_model = tf.keras.models.load_model(
            os.path.join(dirpath, "model.h5"), compile=False
        )
        state.iceflow_model.compile(jit_compile=True)
    else:
        print(
            "----------------------------------> No pretrained emulator, start from scratch."
        )
        nb_inputs = len(cfg.processes.iceflow.emulator.fieldin) + (
            cfg.processes.iceflow.physics.dim_arrhenius == 3
        ) * (cfg.processes.iceflow.numerics.Nz - 1)
        nb_outputs = 2 * cfg.processes.iceflow.numerics.Nz
        state.iceflow_model = getattr(
            igm.processes.iceflow.emulate.emulate,
            cfg.processes.iceflow.emulator.network.architecture,
        )(cfg, nb_inputs, nb_outputs)

    @tf.function(jit_compile=True)
    def fast_inference(x):
        return state.iceflow_model(x)

    # Holds the callable TF concrete function - not the model itself. This allows us to update the weights
    # for the graph but keep the XLA compiled function (check!)
    state.iceflow_model_inference = fast_inference
    
    gravity_params_dict = {
        "exp_glen": cfg.processes.iceflow.physics.exp_glen,
        "ice_density": cfg.processes.iceflow.physics.ice_density,
        "gravity_cst": cfg.processes.iceflow.physics.gravity_cst,
        "force_negative_gravitational_energy": cfg.processes.iceflow.physics.force_negative_gravitational_energy,
        "vert_basis": cfg.processes.iceflow.numerics.vert_basis,
    }
        # exp_glen = cfg.processes.iceflow.physics.exp_glen
    # ice_density = cfg.processes.iceflow.physics.ice_density
    # gravity_cst = cfg.processes.iceflow.physics.gravity_cst
    # fnge = cfg.processes.iceflow.physics.force_negative_gravitational_energy
    # vert_basis = cfg.processes.iceflow.numerics.vert_basis
    
    shear_params_dict = {
        "exp_glen": cfg.processes.iceflow.physics.exp_glen,
        "regu_glen": cfg.processes.iceflow.physics.regu_glen,
        "thr_ice_thk": cfg.processes.iceflow.physics.thr_ice_thk,
        "min_sr": cfg.processes.iceflow.physics.min_sr,
        "max_sr": cfg.processes.iceflow.physics.max_sr,
        "vert_basis": cfg.processes.iceflow.numerics.vert_basis,
    }
    
    sliding_weertman_params_dict = {
        "exp_weertman": cfg.processes.iceflow.physics.exp_weertman,
        "regu_weertman": cfg.processes.iceflow.physics.regu_weertman,
        "vert_basis": cfg.processes.iceflow.numerics.vert_basis,
    }


    from abc import ABC, abstractmethod
    class EnergyComponent(ABC):
        @abstractmethod
        def cost():
            pass
    
    class GravityComponent(EnergyComponent):
        def __init__(self, params):
            self.params = params
        def cost(self, U, V, fieldin, vert_disc, staggered_grid):
            return cost_gravity(
                U, V, fieldin, vert_disc, staggered_grid, self.params
            )
    class ShearComponent(EnergyComponent):
        def __init__(self, params):
            self.params = params
        def cost(self, U, V, fieldin, vert_disc, staggered_grid):
            return cost_shear(
                U, V, fieldin, vert_disc, staggered_grid, self.params
            )
    class SlidingWeertmanComponent(EnergyComponent):
        def __init__(self, params):
            self.params = params
        def cost(self, U, V, fieldin, vert_disc, staggered_grid):
            return cost_sliding_weertman(
                U, V, fieldin, vert_disc, staggered_grid, self.params
            )
    
    EnergyComponents = {
        "gravity": GravityComponent,
        "shear": ShearComponent,
        # "sliding_weertman": SlidingWeertmanComponent,
    }
    
        
    EnergyParams = {
        "gravity": gravity_params_dict,
        "shear": shear_params_dict,
        # "sliding_weertman": sliding_weertman_params_dict,
    }
    
    state.iceflow.energy_components = []
    for component in cfg.processes.iceflow.physics.energy_components:
        if component not in EnergyComponents:
            raise ValueError(f"Unknown energy component: {component}")

        component_class = EnergyComponents[component]
        params = EnergyParams[component]
        state.iceflow.energy_components.append(
            component_class(params)
        )
    
        # exp_glen = cfg.processes.iceflow.physics.exp_glen
    # regu_glen = cfg.processes.iceflow.physics.regu_glen
    # thr_ice_thk = cfg.processes.iceflow.physics.thr_ice_thk
    # min_sr = cfg.processes.iceflow.physics.min_sr
    # max_sr = cfg.processes.iceflow.physics.max_sr
    # vert_basis = cfg.processes.iceflow.numerics.vert_basis

    
        # exp_weertman = cfg.processes.iceflow.physics.exp_weertman
    # regu_weertman = cfg.processes.iceflow.physics.regu_weertman
    # vert_basis = cfg.processes.iceflow.numerics.vert_basis




from typing import Dict, List, Any


def extract_state_for_emulated(state) -> Dict[str, Any]:
    """
    Extract necessary state variables for emulated functon.

    Args:
        state: The original state object

    Returns:
        Dictionary containing all necessary state variables
    """
    return dict(
        {
            "thk": state.thk,
            "PAD": state.PAD,
            "vert_weight": state.vert_weight,
            "U": state.U,
            "V": state.V,
            "iceflow_model_inference": state.iceflow_model_inference,
        }
    )


class UpdatedIceflowEmulatedParams(tf.experimental.ExtensionType):
    Nz: int
    arrhenius_dimension: int
    exclude_borders: int
    multiple_window_size: int
    force_max_velbar: float
    vertical_basis: str


@tf.function(jit_compile=True)
def update_iceflow_emulated(data: Dict, fieldin: tf.Tensor, parameters: UpdatedIceflowEmulatedParams) -> Dict[str, tf.Tensor]:

    # Define the input of the NN, include scaling

    Ny, Nx = data["thk"].shape

    if parameters.arrhenius_dimension == 3:
        X = fieldin_to_X_3d(parameters.arrhenius_dimension, fieldin)
    elif parameters.arrhenius_dimension == 2:
        X = fieldin_to_X_2d(fieldin)

    if parameters.exclude_borders > 0:
        iz = parameters.exclude_borders
        X = tf.pad(X, [[0, 0], [iz, iz], [iz, iz], [0, 0]], "SYMMETRIC")

    if parameters.multiple_window_size == 0:
        Y = data["iceflow_model_inference"](X)
    else:
        Y = data["iceflow_model_inference"](tf.pad(X, data["PAD"], "CONSTANT"))[
            :, :Ny, :Nx, :
        ]

    if parameters.exclude_borders > 0:
        iz = parameters.exclude_borders
        Y = Y[:, iz:-iz, iz:-iz, :]

    U, V = Y_to_UV(parameters.Nz, Y)
    U = U[0]
    V = V[0]

    U = tf.where(data["thk"] > 0, U, 0)
    V = tf.where(data["thk"] > 0, V, 0)

    # If requested, the speeds are artifically upper-bounded
    if parameters.force_max_velbar > 0:
        U, V = clip_max_velbar(
            U,
            V,
            parameters.force_max_velbar,
            parameters.vertical_basis,
            data["vert_weight"],
        )

    uvelbase, vvelbase = get_velbase(data["U"], data["V"], parameters.vertical_basis)
    uvelsurf, vvelsurf = get_velsurf(data["U"], data["V"], parameters.vertical_basis)
    ubar, vbar = get_velbar(
        data["U"], data["V"], data["vert_weight"], parameters.vertical_basis
    )

    return {
        "U": U,
        "V": V,
        "uvelbase": uvelbase,
        "vvelbase": vvelbase,
        "uvelsurf": uvelsurf,
        "vvelsurf": vvelsurf,
        "ubar": ubar,
        "vbar": vbar,
    }

def match_fieldin_dimensions(fieldin):

    for i in tf.range(len(fieldin)):
        field = fieldin[i]

        if tf.rank(field) == 2:
            field = tf.expand_dims(field, axis=0)
        if i == 0:
            fieldin_matched = field
        else:

            fieldin_matched = tf.concat([fieldin_matched, field], axis=0)

    fieldin_matched = tf.expand_dims(fieldin_matched, axis=0)
    fieldin_matched = tf.transpose(fieldin_matched, perm=[0, 2, 3, 1])
    return fieldin_matched

# tf.config.optimizer.set_jit(True)

@tf.function(jit_compile=True)
def apply_gradients_xla(optimizer, grads_and_vars):
    optimizer.apply_gradients(grads_and_vars) # does this work?
    
    return 0

def update_iceflow_emulator(cfg, data, fieldin, vert_disc, pertubate, parameters):

    iz = parameters.iz
    
    if parameters.arrhenius_dimension == 3:
        X = fieldin_to_X_3d(parameters.arrhenius_dimension, fieldin)
    elif parameters.arrhenius_dimension == 2:
        X = fieldin_to_X_2d(fieldin)

    if pertubate:
        X = pertubate_X(cfg, X)

    X = split_into_patches(
        X,
        parameters.framesizemax,
        parameters.split_patch_method,
    )

    Ny = X.shape[-3]
    Nx = X.shape[-2]

    padding = compute_PAD(parameters.multiple_window_size, Nx, Ny)

    rng_outer = igm.utils.profiling.srange("Training Loop", "red")
    
    for _ in tf.range(data["nbit"]):
        cost_emulator = 0.0

        # print(X.shape[0])
        for i in tf.range(tf.constant(X.shape[0])):
            with tf.GradientTape(persistent=True) as tape:

                # if training_loop_params["lr_decay"] < 1:
                #     training_loop_state_objects["opti_retrain"].lr = lr * (
                #         training_loop_params["lr_decay"] ** (i / 1000)
                #     )

                # rng = igm.utils.profiling.srange("Running Forward Model", "White")
                Y = data["iceflow_model_inference"](tf.pad(X[i, :, :, :, :], padding, "CONSTANT"))[ # ! CHECK THAT CALLING THE INFERENCE MODEL HERE STILL PASSES THE GRADIENTS BACK CORRECTLY
                    :, :Ny, :Nx, :
                ]

                energy_list = iceflow_energy_XY(
                    Nz=parameters.Nz,
                    dim_arrhenius=parameters.arrhenius_dimension,
                    staggered_grid=parameters.staggered_grid,
                    fieldin_names=parameters.fieldin_names,
                    X=X[i, :, iz : Ny - iz, iz : Nx - iz, :],
                    Y=Y[:, iz : Ny - iz, iz : Nx - iz, :],
                    vert_disc=vert_disc,
                    energy_components=data["energy_components"],
                )
                
                basis_vectors, sliding_shear_stress = sliding_law_XY(
                    X=X[i, :, iz : Ny - iz, iz : Nx - iz, :],
                    Y=Y[:, iz : Ny - iz, iz : Nx - iz, :],
                    Nz=parameters.Nz,
                    fieldin_list=parameters.fieldin_names,
                    dim_arrhenius=parameters.arrhenius_dimension,
                    sliding_law=data["sliding_law"],
                )
                
                energy_mean_list = tf.reduce_mean(energy_list, axis=[1, 2, 3]) # mean over the spatial dimensions
                total_energy = tf.reduce_sum(energy_mean_list, axis=0) # axis is right?
                cost_emulator += total_energy
            
            nonsliding_gradients = tape.gradient(total_energy, data["iceflow_model"].trainable_variables)
            sliding_gradients = tape.gradient(target=basis_vectors,
                                              sources=data["iceflow_model"].trainable_variables,
                                              output_gradients=sliding_shear_stress,
            )
            
            total_gradients = [
                grad + (sgrad / tf.cast(Nx * Ny, tf.float32))
                for grad, sgrad in zip(nonsliding_gradients, sliding_gradients)
            ]
                

            # rng = igm.utils.profiling.srange("Applying the gradients", "green")
            data["opti_retrain"].apply_gradients(
                zip(total_gradients, data["iceflow_model"].trainable_variables)
            )


            del tape
            

            # state.emulator_cost = state.emulator_cost.write(epoch, cost_emulator)
            # state.emulator_grad = state.emulator_grad.write(epoch, grad_emulator)

    igm.utils.profiling.erange(rng_outer)


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

    if split_patch_method == "sequential":
        XXX = tf.stack(XX, axis=0)
    elif split_patch_method == "parallel":
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


def save_iceflow_model(cfg, state):
    directory = "iceflow-model"

    import shutil

    if os.path.exists(directory):
        shutil.rmtree(directory)

    os.mkdir(directory)

    state.iceflow_model.save(os.path.join(directory, "model.h5"))

    #    fieldin_dim=[0,0,1*(cfg.processes.iceflow.physics.dim_arrhenius==3),0,0]

    fid = open(os.path.join(directory, "fieldin.dat"), "w")
    #    for key,gg in zip(cfg.processes.iceflow.emulator.fieldin,fieldin_dim):
    #        fid.write("%s %.1f \n" % (key, gg))
    for key in cfg.processes.iceflow.emulator.fieldin:
        print(key)
        fid.write("%s \n" % (key))
    fid.close()

    fid = open(os.path.join(directory, "vert_grid.dat"), "w")
    fid.write(
        "%4.0f  %s \n"
        % (cfg.processes.iceflow.numerics.Nz, "# number of vertical grid point (Nz)")
    )
    fid.write(
        "%2.2f  %s \n"
        % (
            cfg.processes.iceflow.numerics.vert_spacing,
            "# param for vertical spacing (vert_spacing)",
        )
    )
    fid.close()
