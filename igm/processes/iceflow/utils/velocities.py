from typing import Optional, Tuple
import tensorflow as tf

from igm.utils.math.getmag import getmag


@tf.function(jit_compile=True)
def compute_cell_ice_mask(thk: tf.Tensor) -> tf.Tensor:
    """Return cells whose four corner nodes have positive thickness."""
    return (
        (thk[..., :-1, :-1] > 0.0)
        & (thk[..., :-1, 1:] > 0.0)
        & (thk[..., 1:, :-1] > 0.0)
        & (thk[..., 1:, 1:] > 0.0)
    )


def _pad_cell_mask_to_nodes(
    cell_mask: tf.Tensor,
    top: int,
    bottom: int,
    left: int,
    right: int,
) -> tf.Tensor:
    rank = cell_mask.shape.rank
    if rank is None:
        raise ValueError("cell_mask rank must be statically known.")
    paddings = [[0, 0]] * (rank - 2) + [[top, bottom], [left, right]]
    return tf.pad(cell_mask, paddings)


@tf.function(jit_compile=True)
def compute_node_ice_mask(thk: tf.Tensor) -> tf.Tensor:
    """Return ice nodes belonging to at least one active Q1 cell."""
    cell_mask = compute_cell_ice_mask(thk)
    node_support = (
        _pad_cell_mask_to_nodes(cell_mask, 0, 1, 0, 1)
        | _pad_cell_mask_to_nodes(cell_mask, 0, 1, 1, 0)
        | _pad_cell_mask_to_nodes(cell_mask, 1, 0, 0, 1)
        | _pad_cell_mask_to_nodes(cell_mask, 1, 0, 1, 0)
    )
    return (thk > 0.0) & node_support


def get_velbase_1(U: tf.Tensor, V_b: tf.Tensor) -> tf.Tensor:
    """Get the basal velocity of the velocity component U."""
    return tf.einsum("j,...jkl->...kl", V_b, U)


@tf.function(jit_compile=True)
def get_velbase(
    U: tf.Tensor, V: tf.Tensor, V_b: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Get the basal velocity of the velocity vector (U, V)."""
    return get_velbase_1(U, V_b), get_velbase_1(V, V_b)


def get_velsurf_1(U: tf.Tensor, V_s: tf.Tensor) -> tf.Tensor:
    """Get the surface velocity of the velocity component U."""
    return tf.einsum("j,...jkl->...kl", V_s, U)


@tf.function(jit_compile=True)
def get_velsurf(
    U: tf.Tensor, V: tf.Tensor, V_s: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Get the surface velocity of the velocity vector (U, V)."""
    return get_velsurf_1(U, V_s), get_velsurf_1(V, V_s)


def get_velbar_1(U: tf.Tensor, V_bar: tf.Tensor) -> tf.Tensor:
    """Get the vertically-averaged velocity of the velocity component U."""
    return tf.einsum("j,...jkl->...kl", V_bar, U)


@tf.function(jit_compile=True)
def get_velbar(
    U: tf.Tensor, V: tf.Tensor, V_bar: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Get the vertically-averaged velocity of the velocity vector (U, V)."""
    return get_velbar_1(U, V_bar), get_velbar_1(V, V_bar)


@tf.function(jit_compile=True)
def boundvel(velbar_mag: tf.Tensor, U: tf.Tensor, velbar_mag_max: float) -> tf.Tensor:
    """Bound the velocity component U."""
    return tf.where(velbar_mag >= velbar_mag_max, velbar_mag_max * (U / velbar_mag), U)


@tf.function(jit_compile=True)
def clip_max_velbar(
    U: tf.Tensor, V: tf.Tensor, V_bar: tf.Tensor, velbar_mag_max: float
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Bound the velocity vector (U, V)."""

    velbar_x, velbar_y = get_velbar(U, V, V_bar)
    velbar_mag = getmag(velbar_x, velbar_y)

    U_clipped = boundvel(velbar_mag, U, velbar_mag_max)
    V_clipped = boundvel(velbar_mag, V, velbar_mag_max)

    return U_clipped, V_clipped


@tf.function(jit_compile=True)
def get_misfit(
    U1: tf.Tensor,
    V1: tf.Tensor,
    U2: tf.Tensor,
    V2: tf.Tensor,
    V_bar: tf.Tensor,
    thk: Optional[tf.Tensor] = None,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Get L1 and L2 misfits between (U1, V1) and (U2, V2)."""

    delta_velbar_x, delta_velbar_y = get_velbar(U1 - U2, V1 - V2, V_bar)

    delta_velbar_mag = getmag(delta_velbar_x, delta_velbar_y)

    mask = tf.ones_like(delta_velbar_mag)
    if thk is not None:
        mask = tf.cast(thk > 0.0, delta_velbar_mag.dtype)

    mask_sum = tf.reduce_sum(mask)
    error_L1 = tf.reduce_sum(mask * delta_velbar_mag) / mask_sum
    error_L2 = tf.sqrt(tf.reduce_sum(mask * tf.square(delta_velbar_mag)) / mask_sum)

    return error_L1, error_L2
