import tensorflow as tf


def compute_divflux(u, v, h, dx, dy):
    """
    Upwind computation of the divergence of the flux: d(u h)/dx + d(v h)/dy.

    Parameters:
    -----------
    u : tf.Tensor
        x-component of velocity (ny, nx)
    v : tf.Tensor
        y-component of velocity (ny, nx)
    h : tf.Tensor
        Ice thickness (ny, nx)
    dx : float
        Grid spacing in x direction
    dy : float
        Grid spacing in y direction

    Returns:
    --------
    divflux : tf.Tensor
        Divergence of flux (ny, nx)
    """
    # Compute u and v on the staggered grid
    u = tf.concat([u[:, :1], 0.5 * (u[:, :-1] + u[:, 1:]), u[:, -1:]], axis=1)  # shape (ny, nx+1)
    v = tf.concat([v[:1, :], 0.5 * (v[:-1, :] + v[1:, :]), v[-1:, :]], axis=0)  # shape (ny+1, nx)

    # Extend h with constant value at the domain boundaries
    Hx = tf.pad(h, [[0, 0], [1, 1]], mode="CONSTANT")  # shape (ny, nx+2)
    Hy = tf.pad(h, [[1, 1], [0, 0]], mode="CONSTANT")  # shape (ny+2, nx)

    # Compute fluxes by selecting the upwind quantities
    Qx = u * tf.where(u > 0, Hx[:, :-1], Hx[:, 1:])  # shape (ny, nx+1)
    Qy = v * tf.where(v > 0, Hy[:-1, :], Hy[1:, :])  # shape (ny+1, nx)

    # Compute the divergence, final shape is (ny, nx)
    divflux = (Qx[:, 1:] - Qx[:, :-1]) / dx + (Qy[1:, :] - Qy[:-1, :]) / dy
    return divflux


def compute_gradient(s, dx, dy):
    """
    Compute spatial 2D gradient of a given field.

    Parameters:
    -----------
    s : tf.Tensor
        Input field (ny, nx)
    dx : float
        Grid spacing in x direction
    dy : float
        Grid spacing in y direction

    Returns:
    --------
    diffx : tf.Tensor
        Gradient in x direction (ny, nx)
    diffy : tf.Tensor
        Gradient in y direction (ny, nx)
    """
    EX = tf.concat([1.5 * s[:, :1] - 0.5 * s[:, 1:2],
                    0.5 * (s[:, :-1] + s[:, 1:]),
                    1.5 * s[:, -1:] - 0.5 * s[:, -2:-1]], axis=1)
    diffx = (EX[:, 1:] - EX[:, :-1]) / dx

    EY = tf.concat([1.5 * s[:1, :] - 0.5 * s[1:2, :],
                    0.5 * (s[:-1, :] + s[1:, :]),
                    1.5 * s[-1:, :] - 0.5 * s[-2:-1, :]], axis=0)
    diffy = (EY[1:, :] - EY[:-1, :]) / dy

    return diffx, diffy


def apply_boundary_condition(H_ice, boundary_width=5):
    """
    Apply boundary condition to the ice thickness field `H_ice`.

    Parameters:
    -----------
    H_ice : tf.Tensor
        Ice thickness field (ny, nx)
    boundary_width : int
        Width of boundary zone for ramping

    Returns:
    --------
    H_ice : tf.Tensor
        Ice thickness with boundary conditions applied
    """
    ny, nx = H_ice.shape

    # Create linear ramps
    ramp = tf.linspace(1.0, 0.0, boundary_width)
    ramp_rev = tf.reverse(ramp, axis=[0])

    # Convert H_ice to a variable for in-place-like operations
    H_ice = tf.Variable(H_ice, trainable=False)

    # Apply boundary condition to the left and right boundaries
    left_mask = tf.reshape(ramp_rev, [1, boundary_width])
    right_mask = tf.reshape(ramp, [1, boundary_width])

    # Create full masks
    left_full = tf.ones([ny, nx], dtype=H_ice.dtype)
    right_full = tf.ones([ny, nx], dtype=H_ice.dtype)

    # Build masks using tensor operations
    indices_left = tf.range(boundary_width)
    indices_right = tf.range(nx - boundary_width, nx)

    # Apply using scatter or masking approach
    H_ice_np = H_ice.numpy()
    H_ice_np[:, :boundary_width] *= ramp_rev.numpy()
    H_ice_np[:, -boundary_width:] *= ramp.numpy()
    H_ice_np[:boundary_width, :] *= ramp_rev.numpy().reshape(-1, 1)
    H_ice_np[-boundary_width:, :] *= ramp.numpy().reshape(-1, 1)

    return tf.constant(H_ice_np, dtype=H_ice.dtype)


@tf.function
def apply_boundary_condition_tf(H_ice, boundary_width=5):
    """
    Apply boundary condition to the ice thickness field `H_ice`.
    Pure TensorFlow implementation for use in gradient tape.

    Parameters:
    -----------
    H_ice : tf.Tensor
        Ice thickness field (ny, nx)
    boundary_width : int
        Width of boundary zone for ramping

    Returns:
    --------
    H_ice : tf.Tensor
        Ice thickness with boundary conditions applied
    """
    ny = tf.shape(H_ice)[0]
    nx = tf.shape(H_ice)[1]

    # Create linear ramps
    ramp = tf.linspace(1.0, 0.0, boundary_width)
    ramp_rev = tf.reverse(ramp, axis=[0])

    # Create mask tensors for each boundary
    # Left boundary mask
    left_ramp = tf.reshape(ramp_rev, [1, boundary_width])
    left_ones = tf.ones([1, nx - boundary_width], dtype=H_ice.dtype)
    left_mask_row = tf.concat([left_ramp, left_ones], axis=1)
    left_mask = tf.tile(left_mask_row, [ny, 1])

    # Right boundary mask
    right_ramp = tf.reshape(ramp, [1, boundary_width])
    right_ones = tf.ones([1, nx - boundary_width], dtype=H_ice.dtype)
    right_mask_row = tf.concat([right_ones, right_ramp], axis=1)
    right_mask = tf.tile(right_mask_row, [ny, 1])

    # Top boundary mask
    top_ramp = tf.reshape(ramp_rev, [boundary_width, 1])
    top_ones = tf.ones([ny - boundary_width, 1], dtype=H_ice.dtype)
    top_mask_col = tf.concat([top_ramp, top_ones], axis=0)
    top_mask = tf.tile(top_mask_col, [1, nx])

    # Bottom boundary mask
    bottom_ramp = tf.reshape(ramp, [boundary_width, 1])
    bottom_ones = tf.ones([ny - boundary_width, 1], dtype=H_ice.dtype)
    bottom_mask_col = tf.concat([bottom_ones, bottom_ramp], axis=0)
    bottom_mask = tf.tile(bottom_mask_col, [1, nx])

    # Combine all masks (multiply to get combined effect)
    combined_mask = left_mask * right_mask * top_mask * bottom_mask

    return H_ice * combined_mask
