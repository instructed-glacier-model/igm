from typing import Any, Dict, Tuple
import tensorflow as tf
import os
import warnings
import igm


from igm.processes.iceflow.emulate import EmulatedParams
from igm.processes.iceflow.emulate.emulated import get_emulated_params_args
from igm.processes.iceflow.emulate.utils.misc import (
    get_pretrained_emulator_path,
)
from igm.processes.iceflow.utils.data_preprocessing import prepare_X
from igm.processes.iceflow.energy import (
    EnergyComponents,
    EnergyParams,
    get_energy_params_args,
)

from igm.processes.iceflow.energy.energy import iceflow_energy_XY
from igm.processes.iceflow.vertical import VerticalDiscrs


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


def get_emulator_bag(state, nbit, lr) -> Dict:

    return {
        "iceflow_model_inference": state.iceflow_model_inference,
        "iceflow_model": state.iceflow_model,
        "energy_components": state.iceflow.energy_components,
        "opti_retrain": state.opti_retrain,
        "nbit": nbit,
        "effective_pressure": state.effective_pressure,
        "lr": lr,
        "PAD": state.PAD,
        "vert_disc": state.vert_disc,
    }


def update_iceflow_emulator(cfg, state, fieldin, initial, it, pertubate):

    cfg_emulator = cfg.processes.iceflow.emulator

    warm_up = int(it <= cfg_emulator.warm_up_it)
    run_it = (
        (cfg_emulator.retrain_freq > 0)
        & (it > 0)
        & (it % cfg_emulator.retrain_freq == 0)
    )

    if initial or run_it or warm_up:
        nbit = cfg_emulator.nbit_init if warm_up else cfg_emulator.nbit
        lr = cfg_emulator.lr_init if warm_up else cfg_emulator.lr
        X = prepare_X(cfg, fieldin, pertubate=pertubate, split_into_patches=True)
        bag = get_emulator_bag(state, nbit, lr)
        state.cost_emulator = update_emulator(bag, X, state.iceflow.emulator_params)


tf.config.optimizer.set_jit(True)


@tf.function(jit_compile=False)
def update_emulator(bag, X, parameters):

    emulator_cost_tensor = tf.TensorArray(dtype=tf.float32, size=bag["nbit"])
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
                    new_lr = bag["lr"] * (parameters.lr_decay ** (i / 1000))
                    bag["opti_retrain"].learning_rate.assign(
                        tf.cast(new_lr, tf.float32)
                    )

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
                    vert_disc=bag["vert_disc"],
                    energy_components=bag["energy_components"],
                )

                energy_mean_staggered = tf.reduce_mean(staggered_energy, axis=[1, 2, 3])
                energy_mean_nonstaggered = tf.reduce_mean(
                    nonstaggered_energy, axis=[1, 2, 3]
                )

                total_energy = tf.reduce_sum(
                    energy_mean_nonstaggered, axis=0
                ) + tf.reduce_sum(energy_mean_staggered, axis=0)
                cost_emulator += total_energy

            gradients = tape.gradient(
                total_energy, bag["iceflow_model"].trainable_variables
            )

            bag["opti_retrain"].apply_gradients(
                zip(gradients, bag["iceflow_model"].trainable_variables)
            )

            if parameters.print_cost:
                tf.print("Iteration", iteration + 1, "/", bag["nbit"], end=" ")
                tf.print(": Cost =", cost_emulator)

            del tape

        emulator_cost_tensor = emulator_cost_tensor.write(iteration, cost_emulator)
        # emulator_grad_tensor = emulator_grad_tensor.write(iteration, total_gradients)

    return emulator_cost_tensor.stack()


def initialize_iceflow_emulator(cfg, state):

    if not hasattr(cfg, "processes"):
        raise AttributeError("❌ <cfg.processes> does not exist.")
    if not hasattr(cfg.processes, "iceflow"):
        raise AttributeError("❌ <cfg.processes.iceflow> does not exist.")
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

    @tf.function(jit_compile=True)
    def fast_inference(x):
        return state.iceflow_model(x)

    # Holds the callable TF concrete function - not the model itself. This allows us to update the weights
    # for the graph but keep the XLA compiled function (check!)
    state.iceflow_model_inference = fast_inference

    # Initialize vertical discretization
    vertical_basis = cfg_numerics.vert_basis.lower()
    vertical_discr = VerticalDiscrs[vertical_basis](cfg)
    state.iceflow.vertical_discr = vertical_discr

    # Initialize energy components
    state.iceflow.energy_components = []
    for component in cfg_physics.energy_components:
        if component not in EnergyComponents:
            raise ValueError(f"❌ Unknown energy component: <{component}>.")

        # Get component and params class
        if component == "sliding":
            law = cfg_physics.sliding.law
            component_class = EnergyComponents[component][law]
            params_class = EnergyParams[component][law]
        else:
            component_class = EnergyComponents[component]
            params_class = EnergyParams[component]

        # Get args extractor
        get_params_args = get_energy_params_args[component]

        # Instantiate params and component classes
        params_args = get_params_args(cfg)
        params = params_class(**params_args)
        component_obj = component_class(params)

        # Add component to the list of components
        state.iceflow.energy_components.append(component_obj)

    # Instantiate emulator params
    emulator_params_args = get_emulator_params_args(cfg, Nx, Ny)
    emulator_params = EmulatorParams(**emulator_params_args)

    # Instantiate emulated params
    emulated_params_args = get_emulated_params_args(cfg)
    emulated_params = EmulatedParams(**emulated_params_args)

    # Save emulator/emulated in the state
    state.iceflow.emulator_params = emulator_params
    state.iceflow.emulated_params = emulated_params
