import tensorflow as tf

@tf.function()
def compute_upwind_gradient_tf(u, v, s, dx):
    #  upwind computation of u ds/dx + v ds/dy

    # Extend E with constant value at the domain boundaries
    Ex = tf.pad(s, [[0, 0], [1, 1]], "SYMMETRIC")  # has shape (ny,nx+2)
    Ey = tf.pad(s, [[1, 1], [0, 0]], "SYMMETRIC")  # has shape (ny+2,nx)

    ## Compute the product selcting the upwind quantities  :-2, 1:-1 , 2:
    Rx = u * tf.where(
        u > 0,
        (Ex[:, 1:-1] - Ex[:, :-2]) / dx,
        (Ex[:, 2:] - Ex[:, 1:-1]) / dx,
    )  # has shape (ny,nx+1)
    Ry = v * tf.where(
        v > 0,
        (Ey[1:-1:, :] - Ey[:-2, :]) / dx,
        (Ey[2:, :] - Ey[1:-1, :]) / dx,
    )  # has shape (ny+1,nx)

    ##  Final shape is (ny,nx)
    return Rx + Ry