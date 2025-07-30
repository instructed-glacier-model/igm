#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
import os

from igm.processes.iceflow.utils import (
    fieldin_to_X_2d,
    fieldin_to_X_3d,
    Y_to_UV,
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
import importlib_resources
import igm
from igm.processes.iceflow.utils import TrainingParams
import warnings
from igm.processes.iceflow.energy import EnergyComponents, GravityParams, ViscosityParams, SlidingWeertmanParams, FloatingParams
from omegaconf import DictConfig

def get_emulator_path(cfg: DictConfig):
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
    
    return direct_name

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

    direct_name = get_emulator_path(cfg)

    if cfg.processes.iceflow.emulator.pretrained:
        dirpath = ''
        if cfg.processes.iceflow.emulator.name == "":
            if os.path.exists(
                importlib_resources.files(emulators).joinpath(direct_name)
            ):
                dirpath = importlib_resources.files(emulators).joinpath(direct_name)
                print("Found pretrained emulator in the igm package: " + direct_name)
            else:
                warnings.warn("No pretrained emulator found in the igm package")
        else:
            dirpath = os.path.join(
                state.original_cwd, cfg.processes.iceflow.emulator.name
            )
            if os.path.exists(dirpath):
                print(f"'-'*40 Found pretrained emulator: {cfg.processes.iceflow.emulator.name} ")
                #     "----------------------------------> Found pretrained emulator: "
                #     + 
                # )
            else:
                warnings.warn("No pretrained emulator found")

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
        warnings.warn("No pretrained emulator found. Starting from scratch.")

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

    # ! Have a separate function that takes care of this
    # Todo: Lets try to find a convention so we can reliably use dictionary unpacking to keep this tidy
    gravity_params = GravityParams(
        exp_glen=cfg.processes.iceflow.physics.exp_glen,
        ice_density=cfg.processes.iceflow.physics.ice_density,
        gravity_cst=cfg.processes.iceflow.physics.gravity_cst,
        force_negative_gravitational_energy=cfg.processes.iceflow.physics.force_negative_gravitational_energy,
        vert_basis=cfg.processes.iceflow.numerics.vert_basis,
    )
    
    viscosity_params = ViscosityParams(
        exp_glen=cfg.processes.iceflow.physics.exp_glen,
        regu_glen=cfg.processes.iceflow.physics.regu_glen,
        thr_ice_thk=cfg.processes.iceflow.physics.thr_ice_thk,
        min_sr=cfg.processes.iceflow.physics.min_sr,
        max_sr=cfg.processes.iceflow.physics.max_sr,
        vert_basis=cfg.processes.iceflow.numerics.vert_basis,
    )
    
    sliding_weertman_params = SlidingWeertmanParams(
        exp_weertman=cfg.processes.iceflow.physics.exp_weertman,
        regu_weertman=cfg.processes.iceflow.physics.regu_weertman,
        vert_basis=cfg.processes.iceflow.numerics.vert_basis,
    )


    floating_params = FloatingParams(
        Nz = cfg.processes.iceflow.numerics.Nz,
        vert_spacing = cfg.processes.iceflow.numerics.vert_spacing,
        cf_eswn = cfg.processes.iceflow.physics.cf_eswn,
        vert_basis = cfg.processes.iceflow.numerics.vert_basis
    )

    EnergyParams = {
        "gravity": gravity_params,
        "viscosity": viscosity_params,
        "sliding_weertman": sliding_weertman_params,
        "floating": floating_params
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

    emulator_params = TrainingParams(
        lr_decay=cfg.processes.iceflow.emulator.lr_decay,
        Nx=state.thk.shape[1],
        Ny=state.thk.shape[0],
        Nz=cfg.processes.iceflow.numerics.Nz,
        iz=cfg.processes.iceflow.emulator.exclude_borders,
        multiple_window_size=cfg.processes.iceflow.emulator.network.multiple_window_size,
        framesizemax=cfg.processes.iceflow.emulator.framesizemax,
        split_patch_method=cfg.processes.iceflow.emulator.split_patch_method,
        arrhenius_dimension = cfg.processes.iceflow.physics.dim_arrhenius,
        staggered_grid=cfg.processes.iceflow.numerics.staggered_grid,
        fieldin_names=tuple(cfg.processes.iceflow.emulator.fieldin),
    )
    
    emulated_params = UpdatedIceflowEmulatedParams(
            Nz = cfg.processes.iceflow.numerics.Nz,
            arrhenius_dimension = cfg.processes.iceflow.physics.dim_arrhenius,
            exclude_borders = cfg.processes.iceflow.emulator.exclude_borders,
            multiple_window_size = cfg.processes.iceflow.emulator.network.multiple_window_size,
            force_max_velbar = cfg.processes.iceflow.force_max_velbar,
            vertical_basis  = cfg.processes.iceflow.numerics.vert_basis,
    )

    state.iceflow.emulated_params = emulated_params
    state.iceflow.emulator_params = emulator_params

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

tf.config.optimizer.set_jit(True)

@tf.function(jit_compile=True)
def apply_gradients_xla(optimizer, grads_and_vars):
    optimizer.apply_gradients(grads_and_vars) # does this work?
    
    return 0

@tf.function(jit_compile=False)
def update_iceflow_emulator(data, X, padding, Ny, Nx, iz, vert_disc, parameters):
    
    for _ in tf.range(data["nbit"]):
        cost_emulator = 0.0

        for i in tf.range(tf.constant(X.shape[0])):
            with tf.GradientTape(persistent=True) as tape:

                # if training_loop_params["lr_decay"] < 1:
                #     training_loop_state_objects["opti_retrain"].lr = lr * (
                #         training_loop_params["lr_decay"] ** (i / 1000)
                #     )

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
