from typing import Dict, List

import tensorflow as tf


def split_field_into_patches(X: tf.Tensor, framesizemax: int) -> tf.Tensor:
    """
    Split a [H, W, C] field into non-overlapping patches [N, ly, lx, C].

    Grid-division strategy:
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


def fieldin_state_to_X(state, inputs) -> tf.Tensor:
    """Stack the named input fields read from `state` into a [H, W, C]
    tensor, in the order given by `inputs`.

    The caller passes the channel list (typically
    `cfg.processes.iceflow.unified.inputs`). The function itself is
    name-agnostic; the dual-name handling for the friction field is localized in
    `sliding.get_friction_field`, on the consumer side.
    """
    return tf.stack([getattr(state, f) for f in inputs], axis=-1)


@tf.function(jit_compile=True)
def fieldin_to_X_2d(fieldin):
    """Converts the fieldin variables to X (2D). This X is used as input to the emulator."""

    return tf.expand_dims(fieldin, axis=0)


@tf.function(jit_compile=True)
def X_to_fieldin(X: tf.Tensor, fieldin_names: List) -> Dict[str, tf.Tensor]:
    """Converts the input tensor X to a dictionary of fieldin variables.

    Name-agnostic. The dual-name handling for the friction field is on
    the consumer side, in `sliding.get_friction_field`.
    """
    channels = tf.unstack(X, num=len(fieldin_names), axis=-1)
    return dict(zip(fieldin_names, channels))


@tf.function(jit_compile=True)
def Y_to_UV(Nz, Y):
    """Converts the output of the emulator (Y) to the horizontal velocities (U, V)."""

    U = tf.experimental.numpy.moveaxis(Y[..., :Nz], [-1], [1])
    V = tf.experimental.numpy.moveaxis(Y[..., Nz:], [-1], [1])

    return U, V
