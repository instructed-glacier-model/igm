import tensorflow as tf
import math
from typing import Any, Dict, Tuple, List


def prepare_X(
    cfg, fieldin, pertubate=False, split_into_patches=True
) -> Tuple[tf.Tensor, List[List[int]]]:
    """General preprocessing of the data for the emulator: includes setting up the dimensions, perturbation, patching, and padding."""

    X = fieldin_to_X_2d(fieldin)

    if pertubate:
        X = pertubate_X(cfg, X)

    if split_into_patches:
        X = split_into_patches_X(
            X,
            cfg.processes.iceflow.emulator.framesizemax,
            cfg.processes.iceflow.emulator.split_patch_method,
        )

    return X


def split_into_patches_X(X, nbmax, split_patch_method):
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


def split_field_into_patches(X: tf.Tensor, framesizemax: int) -> tf.Tensor:
    """
    Split a [H, W, C] field into non-overlapping patches [N, ly, lx, C].

    Uses the same grid-division strategy as the emulated split_into_patches_X:
    - sy = H // framesizemax + 1 strips in y
    - sx = W // framesizemax + 1 strips in x
    - each tile is ly = H // sy  by  lx = W // sx  pixels
    - patches are guaranteed to be smaller than framesizemax
    """
    ny = int(X.shape[0])
    nx = int(X.shape[1])
    sy = ny // framesizemax + 1
    sx = nx // framesizemax + 1
    ly = ny // sy
    lx = nx // sx

    patches = []
    for j in range(sy):
        for i in range(sx):
            patches.append(X[j * ly : (j + 1) * ly, i * lx : (i + 1) * lx, :])

    return tf.stack(patches, axis=0)  # [sy*sx, ly, lx, C]


def fieldin_state_to_X(cfg, state) -> tf.Tensor:
    """This is a bit confusing variable naming. Essentially, it takes the inputs specified in the config files, checks they are in state, and then returns a stacked tensor.
    Previously, this was called 'get_fieldin' but typically field_in is a dictionary - not a stacked tensor - hence the confusion.
    """

    fieldin = [vars(state)[f] for f in cfg.processes.iceflow.unified.inputs]
    fieldin = tf.stack(fieldin, axis=-1)

    return fieldin


@tf.function(jit_compile=True)
def fieldin_to_X_2d(fieldin):
    """Converts the fieldin variables to X (2D). This X is used as input to the emulator."""

    return tf.expand_dims(fieldin, axis=0)


@tf.function(jit_compile=True)
def X_to_fieldin(X: tf.Tensor, fieldin_names: List) -> Dict[str, tf.Tensor]:
    """Converts the input tensor X to a dictionary of fieldin variables."""
    return {name: X[..., i] for i, name in enumerate(fieldin_names)}


@tf.function(jit_compile=True)
def Y_to_UV(Nz, Y):
    """Converts the output of the emulator (Y) to the horizontal velocities (U, V)."""

    U = tf.experimental.numpy.moveaxis(Y[..., :Nz], [-1], [1])
    V = tf.experimental.numpy.moveaxis(Y[..., Nz:], [-1], [1])

    return U, V


def UV_to_Y(cfg, U, V):
    """Stacks horizontal velocities (U, V) to match the output of the emulator (Y)."""
    U = tf.experimental.numpy.moveaxis(U, [0], [-1])
    V = tf.experimental.numpy.moveaxis(V, [0], [-1])

    return tf.concat([U, V], axis=-1)[None, ...]


def compute_PAD(multiple_window_size, Nx, Ny):

    # In case of a U-net, must make sure the I/O size is multiple of 2**N
    if multiple_window_size > 0:
        NNy = multiple_window_size * math.ceil(Ny / multiple_window_size)
        NNx = multiple_window_size * math.ceil(Nx / multiple_window_size)
        return [[0, 0], [0, NNy - Ny], [0, NNx - Nx], [0, 0]]
    else:
        return [[0, 0], [0, 0], [0, 0], [0, 0]]
