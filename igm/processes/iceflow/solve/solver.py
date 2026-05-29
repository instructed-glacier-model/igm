#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from omegaconf import DictConfig
from typing import Any, Dict, Tuple
import tensorflow as tf
import tensorflow_probability as tfp

from igm.common import State
from igm.processes.iceflow.energy.energy import iceflow_energy


class SolverParams(tf.experimental.ExtensionType):
    Nx: int
    Ny: int
    batch_size: int
    print_cost: bool


def get_solver_params_args(
    cfg: DictConfig, Nx: int, Ny: int, batch_size: int
) -> Dict[str, Any]:

    cfg_solver = cfg.processes.iceflow.solver
    cfg_numerics = cfg.processes.iceflow.numerics

    return {
        "Nx": Nx,
        "Ny": Ny,
        "batch_size": batch_size,
        "print_cost": cfg_solver.print_cost,
    }


def get_solver_bag(cfg: DictConfig, state: State) -> Dict[str, Any]:

    cfg_solver = cfg.processes.iceflow.solver

    fieldin = {name: getattr(state, name)[None, ...] for name in cfg_solver.fieldin}

    return {
        "fieldin": fieldin,
        "energy_components": state.iceflow.energy_components,
        "optimizer": state.optimizer,
        "nbit": cfg_solver.nbitmax,
        "discr_h": state.iceflow.discr_h,
        "discr_v": state.iceflow.discr_v,
    }


tf.config.optimizer.set_jit(True)


@tf.function(jit_compile=False)
def optimize_adam(
    bag: Dict[str, Any], U: tf.Tensor, V: tf.Tensor, parameters: SolverParams
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

    solver_cost_tensor = tf.TensorArray(dtype=tf.float32, size=bag["nbit"])

    for iteration in tf.range(bag["nbit"]):

        with tf.GradientTape(persistent=True) as tape:

            energy = iceflow_energy(
                U[None, :, :, :],
                V[None, :, :, :],
                bag["fieldin"],
                bag["discr_h"],
                bag["discr_v"],
                bag["energy_components"],
                parameters.batch_size,
                parameters.Ny,
                parameters.Nx,
            )

            energy_mean = tf.reduce_mean(energy, axis=[1, 2, 3])
            total_energy = tf.reduce_sum(energy_mean, axis=0)

        gradients = tape.gradient(total_energy, [U, V])

        bag["optimizer"].apply_gradients(zip(gradients, [U, V]))

        del tape

        if parameters.print_cost:
            tf.print("Iteration", iteration + 1, "/", bag["nbit"], end=" ")
            tf.print(": Cost =", total_energy)

        solver_cost_tensor = solver_cost_tensor.write(iteration, total_energy)

    return U, V, solver_cost_tensor.stack()


def optimize_lbfgs(
    bag: Dict[str, Any], U: tf.Tensor, V: tf.Tensor, parameters: SolverParams
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

    def cost_fn(UV: tf.Tensor) -> tf.Tensor:
        U = UV[0]
        V = UV[1]

        energy = iceflow_energy(
            U[None, :, :, :],
            V[None, :, :, :],
            bag["fieldin"],
            bag["discr_h"],
            bag["discr_v"],
            bag["energy_components"],
            parameters.batch_size,
            parameters.Ny,
            parameters.Nx,
        )

        energy_mean = tf.reduce_mean(energy, axis=[1, 2, 3])
        total_energy = tf.reduce_sum(energy_mean, axis=0)

        return total_energy

    cost_list = []

    def loss_and_gradients_function(UV: tf.Tensor):
        with tf.GradientTape() as tape:
            tape.watch(UV)
            loss = cost_fn(UV)
            cost_list.append(loss)
            gradients = tape.gradient(loss, UV)
        return loss, gradients

    solution = tfp.optimizer.lbfgs_minimize(
        value_and_gradients_function=loss_and_gradients_function,
        initial_position=tf.stack([U, V], axis=0),
        max_iterations=bag["nbit"],
        tolerance=1e-8,
    )

    U = solution.position[0]
    V = solution.position[0]

    return U, V, tf.convert_to_tensor(cost_list)


def solve_iceflow(cfg: DictConfig, state: State, init: bool = False) -> None:

    # Optimize and save cost
    bag = get_solver_bag(cfg, state)

    U = tf.Variable(state.U)
    V = tf.Variable(state.V)

    optimizer_name = cfg.processes.iceflow.solver.optimizer.lower()

    if optimizer_name == "adam":
        optimize = optimize_adam
    elif optimizer_name == "lbfgs":
        optimize = optimize_lbfgs

    state.U, state.V, state.cost_solver = optimize(
        bag, U, V, state.iceflow.solver_params
    )
