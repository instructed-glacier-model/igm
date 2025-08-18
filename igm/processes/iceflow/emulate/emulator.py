from typing import Any, Dict, Tuple
import tensorflow as tf
import os
import warnings
import igm


from igm.processes.iceflow.emulate import EmulatedParams
from igm.processes.iceflow.emulate.emulated import get_emulated_params_args
from igm.processes.iceflow.emulate.utils.misc import (
    get_effective_pressure_precentage,
    get_pretrained_emulator_path,
)
from igm.processes.iceflow.utils.data_preprocessing import compute_PAD
from igm.processes.iceflow.energy import (
    EnergyComponents,
    EnergyParams,
    get_energy_params_args,
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


def get_emulator_params_args(cfg, Nx: int, Ny: int) -> Dict[str, Any]:

    cfg_emulator = cfg.processes.iceflow.emulator
    cfg_numerics = cfg.processes.iceflow.numerics
    cfg_physics = cfg.processes.iceflow.physics

    return {
        "lr_decay": cfg_emulator.lr_decay,
        "Nx": Nx,
        "Ny": Ny,
        "Nz": cfg_numerics.Nz,
        "iz": cfg_emulator.exclude_borders,
        "multiple_window_size": cfg_emulator.network.multiple_window_size,
        "framesizemax": cfg_emulator.framesizemax,
        "split_patch_method": cfg_emulator.split_patch_method,
        "arrhenius_dimension": cfg_physics.dim_arrhenius,
        "staggered_grid": cfg_numerics.staggered_grid,
        "fieldin_names": tuple(cfg_emulator.fieldin),
        "print_cost": cfg_emulator.print_cost,
    }


def get_emulator_inputs(state, nbit, lr) -> Dict:

    return {
        "iceflow_model_inference": state.iceflow_model_inference,
        "iceflow_model": state.iceflow_model,
        "sliding_law": state.iceflow.sliding_law,
        "energy_components": state.iceflow.energy_components,
        "opti_retrain": state.opti_retrain,
        "nbit": nbit,
        "effective_pressure": state.effective_pressure,
        "lr": lr,
    }


tf.config.optimizer.set_jit(True)


@tf.function(jit_compile=False)
def update_iceflow_emulator(data, X, padding, Ny, Nx, iz, vert_disc, parameters):

    emulator_cost_tensor = tf.TensorArray(dtype=tf.float32, size=data["nbit"])
    # emulator_grad_tensor = tf.TensorArray(
    #     dtype=tf.float32, size=parameters.nbit
    # )

    for iteration in tf.range(data["nbit"]):
        cost_emulator = 0.0

        for i in tf.range(tf.constant(X.shape[0])):
            with tf.GradientTape(persistent=True) as tape:

                if parameters.lr_decay < 1:
                    new_lr = data["lr"] * (parameters.lr_decay ** (i / 1000))
                    data["opti_retrain"].learning_rate.assign(
                        tf.cast(new_lr, tf.float32)
                    )

                Y = data["iceflow_model_inference"](
                    tf.pad(X[i, :, :, :, :], padding, "CONSTANT")
                )[:, :Ny, :Nx, :]

                nonstaggered_energy, staggered_energy = iceflow_energy_XY(
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
                    Y=Y[:, iz : Ny - iz, iz : Nx - iz, :],
                    effective_pressure=data["effective_pressure"],
                    sliding_law=data["sliding_law"],
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
                total_energy, data["iceflow_model"].trainable_variables
            )
            sliding_gradients = tape.gradient(
                target=basis_vectors,
                sources=data["iceflow_model"].trainable_variables,
                output_gradients=sliding_shear_stress,
            )

            total_gradients = [
                grad + (sgrad / tf.cast(Nx * Ny, tf.float32))
                for grad, sgrad in zip(nonsliding_gradients, sliding_gradients)
            ]

            data["opti_retrain"].apply_gradients(
                zip(total_gradients, data["iceflow_model"].trainable_variables)
            )

            if parameters.print_cost:
                tf.print("Iteration", iteration + 1, "/", data["nbit"], end=" ")
                tf.print(": Cost =", cost_emulator)

            del tape

        emulator_cost_tensor = emulator_cost_tensor.write(iteration, cost_emulator)
        # emulator_grad_tensor = emulator_grad_tensor.write(iteration, total_gradients)

    return emulator_cost_tensor.stack()


def initialize_iceflow_emulator(cfg, state):

    if not hasattr(cfg, "processes"):
        raise AttributeError("❌ <cfg.processes> does not exist")
    if not hasattr(cfg.processes, "iceflow"):
        raise AttributeError("❌ <cfg.processes.iceflow> does not exist")
    if not hasattr(state, "thk"):
        raise AttributeError("❌ <state.thk> does not exist.")

    cfg_emulator = cfg.processes.iceflow.emulator
    cfg_numerics = cfg.processes.iceflow.numerics
    cfg_physics = cfg.processes.iceflow.physics

    Nx = state.thk.shape[1]
    Ny = state.thk.shape[0]

    # Retraining option
    if (int(tf.__version__.split(".")[1]) <= 10) | (
        int(tf.__version__.split(".")[1]) >= 16
    ):
        state.opti_retrain = getattr(tf.keras.optimizers, cfg_emulator.optimizer)(
            learning_rate=cfg_emulator.lr,
            epsilon=cfg_emulator.optimizer_epsilon,
            clipnorm=cfg_emulator.optimizer_clipnorm,
        )
    else:
        state.opti_retrain = getattr(
            tf.keras.optimizers.legacy, cfg_emulator.optimizer
        )(
            learning_rate=cfg_emulator.lr,
            epsilon=cfg_emulator.optimizer_epsilon,
            clipnorm=cfg_emulator.optimizer_clipnorm,
        )

    if cfg_emulator.pretrained:
        dir_path = get_pretrained_emulator_path(cfg, state)

        fieldin = []
        fid = open(os.path.join(dir_path, "fieldin.dat"), "r")
        for fileline in fid:
            part = fileline.split()
            fieldin.append(part[0])
        fid.close()
        assert cfg_emulator.fieldin == fieldin
        state.iceflow_model = tf.keras.models.load_model(
            os.path.join(dir_path, "model.h5"), compile=False
        )
        state.iceflow_model.compile(jit_compile=True)
    else:
        warnings.warn("No pretrained emulator found. Starting from scratch.")

        nb_inputs = len(cfg_emulator.fieldin) + (cfg_physics.dim_arrhenius == 3) * (
            cfg_numerics.Nz - 1
        )
        nb_outputs = 2 * cfg_numerics.Nz

        state.iceflow_model = getattr(
            igm.processes.iceflow.emulate.utils.networks,
            cfg_emulator.network.architecture,
        )(cfg, nb_inputs, nb_outputs)

    state.PAD = compute_PAD(
        cfg_emulator.network.multiple_window_size,
        Nx,
        Ny,
    )

    @tf.function(jit_compile=True)
    def fast_inference(x):
        return state.iceflow_model(x)

    # Holds the callable TF concrete function - not the model itself. This allows us to update the weights
    # for the graph but keep the XLA compiled function (check!)
    state.iceflow_model_inference = fast_inference

    # Initialize energy components
    state.iceflow.energy_components = []
    for component in cfg_physics.energy_components:
        if component not in EnergyComponents:
            raise ValueError(f"❌ Unknown energy component: <{component}>.")

        # Get component class, params class, and argument extractor
        component_class = EnergyComponents[component]
        params_class = EnergyParams[component]
        get_params_args = get_energy_params_args[component]

        # Instantiate component and params classes
        params_args = get_params_args(cfg)
        params = params_class(**params_args)
        component = component_class(params)

        # Add component to the list of components
        state.iceflow.energy_components.append(component)

    # Instantiate emulator params
    emulator_params_args = get_emulator_params_args(cfg, Nx, Ny)
    emulator_params = EmulatorParams(**emulator_params_args)

    # Instantiate emulated params
    emulated_params_args = get_emulated_params_args(cfg)
    emulated_params = EmulatedParams(**emulated_params_args)

    # Save emulator/emulated in the state
    state.iceflow.emulator_params = emulator_params
    state.iceflow.emulated_params = emulated_params

    # Temporary fix for the effective pressure
    if not hasattr(state, "effective_pressure"):
        warnings.warn(
            f"Effective pressure not provided for sliding law {state.iceflow.sliding_law.name}. Using 0% of ice overburden pressure as default."
        )

        state.effective_pressure = get_effective_pressure_precentage(
            state.thk, percentage=0.0
        )
        state.effective_pressure = tf.where(
            state.effective_pressure < 1e-3, 1e-3, state.effective_pressure
        )
