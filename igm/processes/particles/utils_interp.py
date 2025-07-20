#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf

def scatter_to_3d_grid(particle, field, dx, grid_shape, levels, vert_spacing, neighbor_radius=1):
    """
    Interpolates particle values back to a 3D Eulerian grid with contributions from neighboring cells.
    Ensures full coverage by spreading particle influence to nearby cells.

    Args:
        particle: Dictionary containing particle data (e.g., positions and field values).
        field: Name of the field to scatter.
        dx: Grid spacing.
        grid_shape: Shape of the 3D grid (Nz, Ny, Nx).
        levels: Vertical levels for the grid.
        vert_spacing: Vertical spacing type.
        method: Extrapolation method ("weighted_average" or "kriging").
        neighbor_radius: Radius (in grid cells) to consider for neighboring contributions.

    Returns:
        A 3D tensor representing the interpolated field on the Eulerian grid.
    """

    Nz, Ny, Nx = grid_shape

    # Compute grid indices for each particle
    i = tf.clip_by_value(tf.cast(particle["x"] / dx, tf.int32), 0, Nx - 1)
    j = tf.clip_by_value(tf.cast(particle["y"] / dx, tf.int32), 0, Ny - 1)

    if vert_spacing == 1:
        k = tf.clip_by_value(tf.cast(particle["r"] * Nz, tf.int32), 0, Nz - 1)
    else:
        k = fast_closest_index_tensor(particle["r"], levels)

    # Stack indices for scatter operation
    indices = tf.stack([k, j, i], axis=-1)

    # Initialize grid and counts
    grid = tf.zeros(grid_shape, dtype=particle[field].dtype)
    counts = tf.zeros(grid_shape, dtype=tf.float32)

    # Scatter particle values to the grid and neighboring cells
    for di in range(-neighbor_radius, neighbor_radius + 1):
        for dj in range(-neighbor_radius, neighbor_radius + 1):
            for dk in range(-neighbor_radius, neighbor_radius + 1):
                # Compute neighbor indices
                neighbor_indices = tf.stack([
                    tf.clip_by_value(k + dk, 0, Nz - 1),
                    tf.clip_by_value(j + dj, 0, Ny - 1),
                    tf.clip_by_value(i + di, 0, Nx - 1)
                ], axis=-1)

                # Compute weights based on distance
                distance = tf.sqrt(tf.cast(di**2 + dj**2 + dk**2, tf.float32))
                weight = tf.exp(-distance)  # Exponential decay with distance

                # Scatter values to neighbors
                grid = tf.tensor_scatter_nd_add(
                    grid, neighbor_indices, particle[field] * weight
                )
                counts = tf.tensor_scatter_nd_add(
                    counts, neighbor_indices, tf.ones_like(particle[field], dtype=tf.float32) * weight
                )

    # Avoid division by zero
    counts = tf.where(counts == 0, tf.ones_like(counts), counts)

    # Compute the average value for each grid cell
    grid = grid / counts

    return grid

def fast_closest_index_tensor(levels_target, levels):
    """
    Finds the closest index in `levels` for each value in `zeta_target`.

    Args:
        zeta_target: Tensor of any shape containing target values.
        levels: 1D tensor of sorted levels.

    Returns:
        Tensor of the same shape as `zeta_target` containing the closest indices.
    """
    # Flatten zeta_target to 1D
    levels_target_flat = tf.reshape(levels_target, [-1])

    # Perform searchsorted on the flattened tensor
    idx_flat = tf.searchsorted(levels, levels_target_flat, side="left")

    # Clamp to bounds
    idx_flat = tf.clip_by_value(idx_flat, 1, tf.shape(levels)[0] - 1)

    # Choose the closer of idx-1 and idx
    left = tf.gather(levels, idx_flat - 1)
    right = tf.gather(levels, idx_flat)
    cond = tf.abs(levels_target_flat - left) < tf.abs(levels_target_flat - right)
    idx_flat = tf.where(cond, idx_flat - 1, idx_flat)

    # Reshape back to the original shape of levels_target
    return tf.reshape(idx_flat, tf.shape(levels_target))