from typing import Dict, Tuple
import tensorflow as tf
import os
import warnings
import logging
import igm
import importlib_resources


import igm.processes.iceflow.emulate.emulators as emulators
from igm.processes.iceflow.emulate import EmulatedParams
from igm.processes.iceflow.emulate.utils.misc import get_effective_pressure_precentage, get_emulator_path
from igm.processes.iceflow.energy import EnergyComponents

from igm.processes.iceflow.energy import (
    GravityEnergyParams,
    ViscosityEnergyParams,
    FloatingEnergyParams,
    SlidingWeertmanEnergyParams,
)

from igm.processes.iceflow.sliding import sliding_law_XY
from igm.processes.iceflow.energy.energy import iceflow_energy_XY

class EmulatorParams(tf.experimental.ExtensionType):
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

def get_emulator_bag(state, nbit, lr) -> Dict:
    
    return dict({
        "iceflow_model_inference": state.iceflow_model_inference,
        "iceflow_model": state.iceflow_model,
        "sliding_law": state.iceflow.sliding_law,
        "energy_components": state.iceflow.energy_components,
        "opti_retrain": state.opti_retrain,
        "nbit": nbit,
        "effective_pressure": state.effective_pressure,
        "lr": lr,
        "PAD": state.PAD,
        "vert_disc": state.vert_disc
})
    
tf.config.optimizer.set_jit(True)
@tf.function(jit_compile=False)
def update_iceflow_emulator(bag, X, parameters):

    emulator_cost_tensor = tf.TensorArray(
        dtype=tf.float32, size=bag["nbit"]
    )
    # emulator_grad_tensor = tf.TensorArray(
    #     dtype=tf.float32, size=parameters.nbit
    # )

    Nx = parameters.Nx
    Ny = parameters.Ny
    iz = parameters.iz
    
    for iteration in tf.range(bag["nbit"]):
        cost_emulator = 0.0

        for i in tf.range(tf.constant(X.shape[0])):
            with tf.GradientTape(persistent=True) as tape:

                if parameters.lr_decay < 1:
                    new_lr = bag["lr"] * (
                        parameters.lr_decay ** (i / 1000)
                    )
                    bag["opti_retrain"].learning_rate.assign(tf.cast(new_lr, tf.float32))

                Y = bag["iceflow_model_inference"](
                    tf.pad(X[i, :, :, :, :], bag["PAD"], "CONSTANT")
                )[:, :Ny, :Nx, :]

                nonstaggered_energy, staggered_energy = iceflow_energy_XY(
                    Nz=parameters.Nz,
                    dim_arrhenius=parameters.arrhenius_dimension,
                    staggered_grid=parameters.staggered_grid,
                    fieldin_names=parameters.fieldin_names,
                    X=X[i, :, iz : Ny - iz, iz : Nx - iz, :],
                    Y=Y[:, iz : Ny - iz, iz : Nx - iz, :],
                    vert_disc=bag['vert_disc'],
                    energy_components=bag["energy_components"],
                )

                basis_vectors, sliding_shear_stress = sliding_law_XY(
                    Y=Y[:, iz : Ny - iz, iz : Nx - iz, :],
                    effective_pressure=bag["effective_pressure"],
                    sliding_law=bag["sliding_law"],
                )

                energy_mean_staggered = tf.reduce_mean(staggered_energy, axis=[1, 2, 3])
                energy_mean_nonstaggered = tf.reduce_mean(
                    nonstaggered_energy, axis=[1, 2, 3]
                )
                total_energy = tf.reduce_sum(
                    energy_mean_nonstaggered, axis=0
                ) + tf.reduce_sum(energy_mean_staggered, axis=0)
                cost_emulator += total_energy

            nonsliding_gradients = tape.gradient(
                total_energy, bag["iceflow_model"].trainable_variables
            )
            sliding_gradients = tape.gradient(
                target=basis_vectors,
                sources=bag["iceflow_model"].trainable_variables,
                output_gradients=sliding_shear_stress,
            )

            total_gradients = [
                grad + (sgrad / tf.cast(Nx * Ny, tf.float32))
                for grad, sgrad in zip(nonsliding_gradients, sliding_gradients)
            ]

            bag["opti_retrain"].apply_gradients(
                zip(total_gradients, bag["iceflow_model"].trainable_variables)
            )

            if parameters.print_cost:
                tf.print("Iteration", iteration + 1, "/", bag["nbit"], end=" ")
                tf.print(": Cost =", cost_emulator)

            del tape

        emulator_cost_tensor = emulator_cost_tensor.write(iteration, cost_emulator)
            # emulator_grad_tensor = emulator_grad_tensor.write(iteration, total_gradients)
            
    return emulator_cost_tensor.stack()

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
        dirpath = ""
        if cfg.processes.iceflow.emulator.name == "":
            print(importlib_resources.files(emulators).joinpath(direct_name))
            if os.path.exists(
                importlib_resources.files(emulators).joinpath(direct_name)
            ):
                dirpath = importlib_resources.files(emulators).joinpath(direct_name)
                logging.info(
                    "Found pretrained emulator in the igm package: " + direct_name
                )
            else:
                raise ImportError(f"No pretrained emulator found in the igm package with name {direct_name}")
        else:
            dirpath = os.path.join(
                state.original_cwd, cfg.processes.iceflow.emulator.name
            )
            if os.path.exists(dirpath):
                logging.info(
                    f"'-'*40 Found pretrained emulator: {cfg.processes.iceflow.emulator.name} "
                )
            else:
                raise ImportError(f"No pretrained emulator found with path {dirpath}")

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
    gravity_params = GravityEnergyParams(
        exp_glen=cfg.processes.iceflow.physics.exp_glen,
        ice_density=cfg.processes.iceflow.physics.ice_density,
        gravity_cst=cfg.processes.iceflow.physics.gravity_cst,
        force_negative_gravitational_energy=cfg.processes.iceflow.physics.force_negative_gravitational_energy,
        vert_basis=cfg.processes.iceflow.numerics.vert_basis,
    )

    viscosity_params = ViscosityEnergyParams(
        exp_glen=cfg.processes.iceflow.physics.exp_glen,
        regu_glen=cfg.processes.iceflow.physics.regu_glen,
        thr_ice_thk=cfg.processes.iceflow.physics.thr_ice_thk,
        min_sr=cfg.processes.iceflow.physics.min_sr,
        max_sr=cfg.processes.iceflow.physics.max_sr,
        vert_basis=cfg.processes.iceflow.numerics.vert_basis,
    )

    sliding_weertman_params = SlidingWeertmanEnergyParams(
        exp_weertman=cfg.processes.iceflow.physics.sliding.weertman.exponent,
        regu_weertman=cfg.processes.iceflow.physics.sliding.weertman.regu_weertman,
        vert_basis=cfg.processes.iceflow.numerics.vert_basis,
    )

    floating_params = FloatingEnergyParams(
        Nz=cfg.processes.iceflow.numerics.Nz,
        vert_spacing=cfg.processes.iceflow.numerics.vert_spacing,
        cf_eswn=cfg.processes.iceflow.physics.cf_eswn,
        vert_basis=cfg.processes.iceflow.numerics.vert_basis,
    )

    EnergyParams = {
        "gravity": gravity_params,
        "viscosity": viscosity_params,
        "sliding_weertman": sliding_weertman_params,
        "floating": floating_params,
    }

    state.iceflow.energy_components = []
    for component in cfg.processes.iceflow.physics.energy_components:
        if component not in EnergyComponents:
            raise ValueError(f"Unknown energy component: {component}")

        component_class = EnergyComponents[component]
        params = EnergyParams[component]
        state.iceflow.energy_components.append(component_class(params))

    emulator_params = EmulatorParams(
        lr_decay=cfg.processes.iceflow.emulator.lr_decay,
        Nx=state.thk.shape[1],
        Ny=state.thk.shape[0],
        Nz=cfg.processes.iceflow.numerics.Nz,
        iz=cfg.processes.iceflow.emulator.exclude_borders,
        multiple_window_size=cfg.processes.iceflow.emulator.network.multiple_window_size,
        framesizemax=cfg.processes.iceflow.emulator.framesizemax,
        split_patch_method=cfg.processes.iceflow.emulator.split_patch_method,
        arrhenius_dimension=cfg.processes.iceflow.physics.dim_arrhenius,
        staggered_grid=cfg.processes.iceflow.numerics.staggered_grid,
        fieldin_names=tuple(cfg.processes.iceflow.emulator.fieldin),
        print_cost=cfg.processes.iceflow.emulator.print_cost,
    )

    emulated_params = EmulatedParams(
        Nz=cfg.processes.iceflow.numerics.Nz,
        arrhenius_dimension=cfg.processes.iceflow.physics.dim_arrhenius,
        exclude_borders=cfg.processes.iceflow.emulator.exclude_borders,
        multiple_window_size=cfg.processes.iceflow.emulator.network.multiple_window_size,
        force_max_velbar=cfg.processes.iceflow.force_max_velbar,
        vertical_basis=cfg.processes.iceflow.numerics.vert_basis,
    )

    state.iceflow.emulated_params = emulated_params
    state.iceflow.emulator_params = emulator_params

    if not hasattr(
        state, "effective_pressure"
    ):  # temporarly putting this here but should put in budd / coulomb
        warnings.warn(
            f"Effective pressure not provided for sliding law {state.iceflow.sliding_law.name}. Using 0% of ice overburden pressure as default."
        )

        state.effective_pressure = get_effective_pressure_precentage(
            state.thk, percentage=0.0
        )
        state.effective_pressure = tf.where(
            state.effective_pressure < 1e-3, 1e-3, state.effective_pressure
        )