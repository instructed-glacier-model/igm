
#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf      
import math

# ! Needed for optmized particles!
try:
    import cupy as cp
    from numba import cuda 
except ImportError:
    raise ImportError(
        "The 'particles' module requires the 'cupy', 'numba'  packages. Please install them."
    )

@cuda.jit(cache=True)
def interpolate_2d(interpolated_grid, grid_values, array_particles, depth):
    particle_id = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    if particle_id < array_particles.shape[0]:

        for depth_layer in range(depth):

            y_pos = array_particles[particle_id, 0]
            x_pos = array_particles[particle_id, 1]

            x_1 = int(x_pos)  # left x coordinate
            y_1 = int(y_pos)  # bottom y coordinate
            x_2 = x_1 + 1  # right x coordinate
            y_2 = y_1 + 1  # top y coordinate

            Q_11 = grid_values[depth_layer, y_1, x_1]  # bottom left corner
            Q_12 = grid_values[depth_layer, y_2, x_1]  # top left corner
            Q_21 = grid_values[depth_layer, y_1, x_2]  # bottom right corner
            Q_22 = grid_values[depth_layer, y_2, x_2]  # top right corner

            # Interpolating on x
            dx = x_2 - x_1
            x_left_weight = (x_pos - x_1) / dx
            x_right_weight = (x_2 - x_pos) / dx
            R_1 = (
                x_left_weight * Q_21 + x_right_weight * Q_11
            )  # bottom x interpolation for fixed y_1 (f(x, y_1))
            R_2 = (
                x_left_weight * Q_22 + x_right_weight * Q_12
            )  # top x interpolation for fixed y_2 (f(x, y_2))

            # Interpolating on y
            dy = y_2 - y_1
            y_bottom_weight = (y_pos - y_1) / dy
            y_top_weight = (y_2 - y_pos) / dy

            P = (
                y_bottom_weight * R_2 + y_top_weight * R_1
            )  # final interpolation for fixed x (f(x, y))
            interpolated_grid[depth_layer, particle_id] = P


def interpolate_particles_2d(U, V, W, smb, thk, topg, indices):

    # True for all variables (maybe make it not dependent on U...)
    depth = U.shape[0]
    number_of_particles = indices.shape[1]

    # Convert TF -> CuPy / Numba
    indices_numba = tf.squeeze(
        indices
    )  # (N, 2) instead of (1, N, 2) - we can remove this before the function to simply the code and make it faster
    particles = tf.experimental.dlpack.to_dlpack(indices_numba)
    array_particles = cp.from_dlpack(particles)

    # Setup cuda block - maybe we can play around with different axes or grid-stipe loops
    threadsperblock = 32
    blockspergrid = math.ceil(number_of_particles / threadsperblock)

    U_numba = tf.experimental.dlpack.to_dlpack(U)
    U_numba = cp.from_dlpack(U_numba)

    V_numba = tf.experimental.dlpack.to_dlpack(V)
    V_numba = cp.from_dlpack(V_numba)

    W_numba = tf.experimental.dlpack.to_dlpack(W)
    W_numba = cp.from_dlpack(W_numba)

    thk_numba = tf.experimental.dlpack.to_dlpack(
        tf.expand_dims(tf.constant(thk), axis=0)
    )
    thk_numba = cp.from_dlpack(thk_numba)

    smb_numba = tf.experimental.dlpack.to_dlpack(
        tf.expand_dims(tf.constant(smb), axis=0)
    )
    smb_numba = cp.from_dlpack(smb_numba)

    topg_numba = tf.experimental.dlpack.to_dlpack(
        tf.expand_dims(tf.constant(topg), axis=0)
    )  # had to use tf.constant since topg is a tf variable and not tensor
    topg_numba = cp.from_dlpack(topg_numba)

    # smb_numba = tf.experimental.dlpack.to_dlpack(
    #     tf.expand_dims(tf.constant(smb), axis=0)
    # )
    # smb_numba = cp.from_dlpack(smb_numba)

    # Creating different streams as computations are independent and
    # will help with latency hiding / avoiding default stream and cuda memfree
    stream_u = cuda.stream()
    stream_v = cuda.stream()
    stream_w = cuda.stream()
    stream_thk = cuda.stream()
    stream_topg = cuda.stream()
    stream_smb = cuda.stream()

    u_device = cuda.device_array(
        shape=(depth, number_of_particles), dtype="float32", stream=stream_u
    )
    v_device = cuda.device_array(
        shape=(depth, number_of_particles), dtype="float32", stream=stream_v
    )
    w_device = cuda.device_array(
        shape=(depth, number_of_particles), dtype="float32", stream=stream_w
    )
    thk_device = cuda.device_array(
        shape=(1, number_of_particles), dtype="float32", stream=stream_thk
    )
    topg_device = cuda.device_array(
        shape=(1, number_of_particles), dtype="float32", stream=stream_topg
    )
    smb_device = cuda.device_array(
         shape=(1, number_of_particles), dtype="float32", stream=stream_smb
    )

    
    interpolate_2d[blockspergrid, threadsperblock, stream_u](
        u_device, U_numba, array_particles, depth
    )
    interpolate_2d[blockspergrid, threadsperblock, stream_v](
        v_device, V_numba, array_particles, depth
    )
    # stream_v.synchronize()
    interpolate_2d[blockspergrid, threadsperblock, stream_w](
        w_device, W_numba, array_particles, depth
    )
    # stream_w.synchronize()
    interpolate_2d[blockspergrid, threadsperblock, stream_thk](
        thk_device, thk_numba, array_particles, 1
    )
    # stream_thk.synchronize()
    interpolate_2d[blockspergrid, threadsperblock, stream_topg](
        topg_device, topg_numba, array_particles, 1
    )
    #stream_topg.synchronize()
    interpolate_2d[blockspergrid, threadsperblock, stream_smb](
         smb_device, smb_numba, array_particles, 1
    )

    u = cp.asarray(u_device)
    u = tf.experimental.dlpack.from_dlpack(u.toDlpack())

    v = cp.asarray(v_device)
    v = tf.experimental.dlpack.from_dlpack(v.toDlpack())

    w = cp.asarray(w_device)
    w = tf.experimental.dlpack.from_dlpack(w.toDlpack())

    thk = cp.asarray(thk_device)
    thk = tf.experimental.dlpack.from_dlpack(thk.toDlpack())
    thk = tf.squeeze(thk, axis=0)

    topg = cp.asarray(topg_device)
    topg = tf.experimental.dlpack.from_dlpack(topg.toDlpack())
    topg = tf.squeeze(topg, axis=0)

    smb = cp.asarray(smb_device)
    smb = tf.experimental.dlpack.from_dlpack(smb.toDlpack())
    smb = tf.squeeze(smb, axis=0)

    return u, v, w, smb, thk, topg


