from typing import Dict, Any
import tensorflow as tf

from igm.processes.iceflow.utils.data_preprocessing import (
    fieldin_to_X_2d,
    fieldin_to_X_3d,
    Y_to_UV,
    prepare_X
)

from igm.processes.iceflow.utils.velocities import (
    get_velbase,
    get_velsurf,
    get_velbar,
    clip_max_velbar,
) 

class EmulatedParams(tf.experimental.ExtensionType):
    Nz: int
    exclude_borders: int
    multiple_window_size: int
    force_max_velbar: float
    vertical_basis: str


def get_emulated_bag(state) -> Dict[str, Any]:

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

def update_iceflow_emulated(cfg, state, fieldin):

    X = prepare_X(cfg, fieldin, pertubate=False, split_into_patches=False)
    bag = get_emulated_bag(state)
    updated_variable_dict = update_emulated(bag, X, state.iceflow.emulated_params)
    
    for key, value in updated_variable_dict.items():
        setattr(state, key, value)

@tf.function(jit_compile=True)
def update_emulated(
    bag: Dict, X: tf.Tensor, parameters: EmulatedParams
) -> Dict[str, tf.Tensor]:

    # Define the input of the NN, include scaling

    Ny, Nx = bag["thk"].shape

    if parameters.exclude_borders > 0:
        iz = parameters.exclude_borders
        X = tf.pad(X, [[0, 0], [iz, iz], [iz, iz], [0, 0]], "SYMMETRIC")

    if parameters.multiple_window_size == 0:
        Y = bag["iceflow_model_inference"](X)
    else:
        Y = bag["iceflow_model_inference"](tf.pad(X, bag["PAD"], "CONSTANT"))[
            :, :Ny, :Nx, :
        ]

    if parameters.exclude_borders > 0:
        iz = parameters.exclude_borders
        Y = Y[:, iz:-iz, iz:-iz, :]

    U, V = Y_to_UV(parameters.Nz, Y)
    U = U[0]
    V = V[0]

    U = tf.where(bag["thk"] > 0, U, 0)
    V = tf.where(bag["thk"] > 0, V, 0)

    # If requested, the speeds are artifically upper-bounded
    if parameters.force_max_velbar > 0:
        U, V = clip_max_velbar(
            U,
            V,
            parameters.force_max_velbar,
            parameters.vertical_basis,
            bag["vert_weight"],
        )

    uvelbase, vvelbase = get_velbase(U, V, parameters.vertical_basis)
    uvelsurf, vvelsurf = get_velsurf(U, V, parameters.vertical_basis)
    ubar, vbar = get_velbar(U, V, bag["vert_weight"], parameters.vertical_basis)

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