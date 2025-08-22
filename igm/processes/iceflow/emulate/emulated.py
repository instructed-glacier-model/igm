from typing import Dict, Any
import tensorflow as tf

from igm.processes.iceflow.utils.data_preprocessing import (
    fieldin_to_X_2d,
    fieldin_to_X_3d,
    Y_to_UV,
)

from igm.processes.iceflow.utils.velocities import (
    get_velbase,
    get_velsurf,
    get_velbar,
    clip_max_velbar,
)


class EmulatedParams(tf.experimental.ExtensionType):
    Nz: int
    arrhenius_dimension: int
    exclude_borders: int
    multiple_window_size: int
    force_max_velbar: float
    vertical_basis: str


def get_emulated_params_args(cfg) -> Dict[str, Any]:

    cfg_emulator = cfg.processes.iceflow.emulator
    cfg_numerics = cfg.processes.iceflow.numerics
    cfg_physics = cfg.processes.iceflow.physics

    return {
        "Nz": cfg_numerics.Nz,
        "arrhenius_dimension": cfg_physics.dim_arrhenius,
        "exclude_borders": cfg_emulator.exclude_borders,
        "multiple_window_size": cfg_emulator.network.multiple_window_size,
        "force_max_velbar": cfg.processes.iceflow.force_max_velbar,
        "vertical_basis": cfg_numerics.vert_basis,
    }


def get_emulated_inputs(state) -> Dict[str, Any]:

    return {
        "thk": state.thk,
        "PAD": state.PAD,
        "vert_weight": state.vert_weight,
        "U": state.U,
        "V": state.V,
        "iceflow_model_inference": state.iceflow_model_inference,
    }


@tf.function(jit_compile=True)
def update_iceflow_emulated(
    data: Dict, fieldin: tf.Tensor, parameters: EmulatedParams
) -> Dict[str, tf.Tensor]:

    # Define input of neural network: X
    if parameters.arrhenius_dimension == 3:
        X = fieldin_to_X_3d(parameters.arrhenius_dimension, fieldin)
    elif parameters.arrhenius_dimension == 2:
        X = fieldin_to_X_2d(fieldin)

    if parameters.exclude_borders > 0:
        iz = parameters.exclude_borders
        X = tf.pad(X, [[0, 0], [iz, iz], [iz, iz], [0, 0]], "SYMMETRIC")

    if parameters.multiple_window_size > 0:
        Ny, Nx = data["thk"].shape
        X = (tf.pad(X, data["PAD"], "CONSTANT"))[:, :Ny, :Nx, :]

    # Compute output of neural network: Y
    Y = data["iceflow_model_inference"](X)

    # Post-processing of output of neural network
    if parameters.exclude_borders > 0:
        iz = parameters.exclude_borders
        Y = Y[:, iz:-iz, iz:-iz, :]

    # Compute velocity fields: U, V
    U, V = Y_to_UV(parameters.Nz, Y)
    U = U[0]
    V = V[0]

    # Post-processing of velocity fields
    U = tf.where(data["thk"] > 0.0, U, 0.0)
    V = tf.where(data["thk"] > 0.0, V, 0.0)

    if parameters.force_max_velbar > 0.0:
        U, V = clip_max_velbar(
            U,
            V,
            parameters.force_max_velbar,
            parameters.vertical_basis,
            data["vert_weight"],
        )

    # Retrieve derived quantities from velocity fields
    uvelbase, vvelbase = get_velbase(U, V, parameters.vertical_basis)
    uvelsurf, vvelsurf = get_velsurf(U, V, parameters.vertical_basis)
    ubar, vbar = get_velbar(U, V, data["vert_weight"], parameters.vertical_basis)

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
