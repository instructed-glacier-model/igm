#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from omegaconf import DictConfig
from typing import Any, Dict, Tuple
import tensorflow as tf
import warnings

import igm
from igm.common import State
from igm.processes.iceflow.energy.energy import iceflow_energy_XY
from igm.processes.iceflow.emulate import EmulatedParams
from igm.processes.iceflow.emulate.emulated import get_emulated_params_args
from igm.processes.iceflow.emulate.utils.misc import (
    get_pretrained_emulator_path,
    load_model_from_path,
)

from igm.processes.iceflow.utils.data_preprocessing import (
    prepare_X,
)

from igm.processes.iceflow.utils.data_preprocessing import (
    fieldin_state_to_X,
    compute_PAD,
)
from igm.processes.iceflow.energy.utils import get_energy_components
from .emulated import update_iceflow_emulated


class EmulatorParams(tf.experimental.ExtensionType):
    lr_decay: float
    Nx: int
    Ny: int
    Nz: int
    iz: int
    multiple_window_size: int
    fieldin_names: Tuple[str, ...]
    print_cost: bool


def get_emulator_params_args(cfg: DictConfig, Nx: int, Ny: int) -> Dict[str, Any]:

    cfg_emulator = cfg.processes.iceflow.emulator
    cfg_numerics = cfg.processes.iceflow.numerics

    return {
        "lr_decay": cfg_emulator.lr_decay,
        "Nx": Nx,
        "Ny": Ny,
        "Nz": cfg_numerics.Nz,
        "iz": cfg_emulator.exclude_borders,
        "multiple_window_size": cfg_emulator.network.multiple_window_size,
        "fieldin_names": tuple(cfg_emulator.fieldin),
        "print_cost": cfg_emulator.print_cost,
    }


def get_emulator_bag(
    state: State, nbit: int, lr: float, batch_size: int
) -> Dict[str, Any]:

    return {
        "iceflow_model_inference": state.iceflow_model_inference,
        "iceflow_model": state.iceflow_model,
        "energy_components": state.iceflow.energy_components,
        "opti_retrain": state.opti_retrain,
        "nbit": nbit,
        "lr": lr,
        "PAD": state.PAD,
        "discr_h": state.iceflow.discr_h,
        "discr_v": state.iceflow.discr_v,
        "batch_size": batch_size,
    }


def update_iceflow_emulator(
    cfg: DictConfig, state: State, initial: bool = False
) -> None:
    cfg_emulator = cfg.processes.iceflow.emulator
    it = 0 if initial else state.it
    warm_up = it <= cfg_emulator.warm_up_it
    run_it = cfg_emulator.retrain_freq > 0 and it % cfg_emulator.retrain_freq == 0

    if initial or run_it or warm_up:
        nbit = cfg_emulator.nbit_init if warm_up else cfg_emulator.nbit
        lr = cfg_emulator.lr_init if warm_up else cfg_emulator.lr

        fieldin = fieldin_state_to_X(cfg, state)
        X = prepare_X(
            cfg,
            fieldin,
            pertubate=cfg.processes.iceflow.emulator.pertubate,
            split_into_patches=True,
        )

        batch_size = X.shape[1]

        bag = get_emulator_bag(state, nbit, lr, batch_size)
        state.cost_emulator = update_emulator(bag, X, state.iceflow.emulator_params)

    update_iceflow_emulated(cfg, state)


tf.config.optimizer.set_jit(True)


@tf.function(jit_compile=False)
def update_emulator(
    bag: Dict[str, Any], X: tf.Tensor, parameters: EmulatorParams
) -> tf.Tensor:

    emulator_cost_tensor = tf.TensorArray(dtype=tf.float32, size=bag["nbit"])

    Nx = parameters.Nx
    Ny = parameters.Ny
    iz = parameters.iz

    for iteration in tf.range(bag["nbit"]):
        cost_emulator = 0.0

        for i in tf.range(X.shape[0]):

            with tf.GradientTape(persistent=True) as tape:

                if parameters.lr_decay < 1:
                    new_lr = bag["lr"] * (
                        parameters.lr_decay ** (int(iteration) / 1000)
                    )
                    bag["opti_retrain"].learning_rate.assign(
                        tf.cast(new_lr, tf.float32)
                    )

                Y = bag["iceflow_model_inference"](
                    tf.pad(X[i, :, :, :, :], bag["PAD"], "CONSTANT")
                )[:, :Ny, :Nx, :]

                energy = iceflow_energy_XY(
                    Nz=parameters.Nz,
                    fieldin_names=parameters.fieldin_names,
                    X=X[i, :, iz : Ny - iz, iz : Nx - iz, :],
                    Y=Y[:, iz : Ny - iz, iz : Nx - iz, :],
                    discr_h=bag["discr_h"],
                    discr_v=bag["discr_v"],
                    energy_components=bag["energy_components"],
                    batch_size=bag["batch_size"],
                    Ny=Ny - 2 * iz,
                    Nx=Nx - 2 * iz,
                )

                energy_mean = tf.reduce_mean(energy, axis=[1, 2, 3])
                total_energy = tf.reduce_sum(energy_mean, axis=0)
                cost_emulator += total_energy

            gradients = tape.gradient(
                total_energy, bag["iceflow_model"].trainable_variables
            )

            bag["opti_retrain"].apply_gradients(
                zip(gradients, bag["iceflow_model"].trainable_variables)
            )

            del tape

            if parameters.print_cost:
                tf.print("Iteration", iteration + 1, "/", bag["nbit"], end=" ")
                tf.print(": Cost =", cost_emulator)

        emulator_cost_tensor = emulator_cost_tensor.write(iteration, cost_emulator)

    return emulator_cost_tensor.stack()


def initialize_iceflow_emulator(cfg: Dict, state: State) -> None:

    if not hasattr(cfg, "processes"):
        raise AttributeError("❌ <cfg.processes> does not exist.")
    if not hasattr(cfg.processes, "iceflow"):
        raise AttributeError("❌ <cfg.processes.iceflow> does not exist.")
    if not hasattr(state, "thk"):
        raise AttributeError("❌ <state.thk> does not exist.")

    cfg_emulator = cfg.processes.iceflow.emulator
    cfg_numerics = cfg.processes.iceflow.numerics
    cfg_physics = cfg.processes.iceflow.physics

    # need to do this dummy call to prepare_X to get the dimensions right
    fieldin = fieldin_state_to_X(cfg, state)
    X = prepare_X(
        cfg,
        fieldin,
        pertubate=cfg.processes.iceflow.emulator.pertubate,
        split_into_patches=True,
    )

    Nx = X.shape[-2]
    Ny = X.shape[-3]

    # padding is necessary when using U-net emulator
    state.PAD = compute_PAD(
        cfg_emulator.network.multiple_window_size,
        Nx,
        Ny,
    )

    # Retraining option
    version_tf = int(tf.__version__.split(".")[1])
    if (version_tf <= 10) | (version_tf >= 16):
        module_optimizer = tf.keras.optimizers
    else:
        module_optimizer = tf.keras.optimizers.legacy

    state.opti_retrain = getattr(module_optimizer, cfg_emulator.optimizer)(
        learning_rate=cfg_emulator.lr,
        epsilon=cfg_emulator.optimizer_epsilon,
        clipnorm=cfg_emulator.optimizer_clipnorm,
    )

    if cfg_emulator.pretrained:
        dir_path = get_pretrained_emulator_path(cfg, state)
        state.iceflow_model = load_model_from_path(dir_path, cfg_emulator.fieldin)
        state.iceflow_model.compile(jit_compile=True)
    else:
        warnings.warn("No pretrained emulator found. Starting from scratch.")

        nb_inputs = len(cfg_emulator.fieldin)
        nb_outputs = 2 * cfg_numerics.Nz

        from igm.processes.iceflow.emulate.utils.architectures import Architectures
        arch_name = cfg_emulator.network.architecture.upper()
        state.iceflow_model = Architectures[arch_name](cfg, nb_inputs, nb_outputs)

    @tf.function(jit_compile=True)
    def fast_inference(x):
        return state.iceflow_model(x)

    # Holds the callable TF concrete function - not the model itself. This allows us to update the weights
    # for the graph but keep the XLA compiled function (check!)
    state.iceflow_model_inference = fast_inference

    # Initialize energy components
    state.iceflow.energy_components = get_energy_components(cfg)

    # Instantiate emulator params
    emulator_params_args = get_emulator_params_args(cfg, Nx, Ny)
    emulator_params = EmulatorParams(**emulator_params_args)

    # Instantiate emulated params
    emulated_params_args = get_emulated_params_args(cfg)
    emulated_params = EmulatedParams(**emulated_params_args)

    # Save emulator/emulated in the state
    state.iceflow.emulator_params = emulator_params
    state.iceflow.emulated_params = emulated_params

    # Update the emulator and evaluate it once
    update_iceflow_emulator(cfg, state, initial=True)
