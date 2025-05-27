import tensorflow as tf

from igm.utils.math.gaussian_filter_tf import gaussian_filter_tf

@tf.function()
def compute_divflux(u, v, h, dx, dy, method='upwind'):
    """
    upwind computation of the divergence of the flux : d(u h)/dx + d(v h)/dy
    First, u and v are computed on the staggered grid (i.e. cell edges)
    Second, one extend h horizontally by a cell layer on any bords (assuming same value)
    Third, one compute the flux on the staggered grid slecting upwind quantities
    Last, computing the divergence on the staggered grid yields values def on the original grid
    """

    if method == 'upwind':

        ## Compute u and v on the staggered grid
        u = tf.concat(
            [u[:, 0:1], 0.5 * (u[:, :-1] + u[:, 1:]), u[:, -1:]], 1
        )  # has shape (ny,nx+1)
        v = tf.concat(
            [v[0:1, :], 0.5 * (v[:-1, :] + v[1:, :]), v[-1:, :]], 0
        )  # has shape (ny+1,nx)

        # Extend h with constant value at the domain boundaries
        Hx = tf.pad(h, [[0, 0], [1, 1]], "CONSTANT")  # has shape (ny,nx+2)
        Hy = tf.pad(h, [[1, 1], [0, 0]], "CONSTANT")  # has shape (ny+2,nx)

        ## Compute fluxes by selcting the upwind quantities
        Qx = u * tf.where(u > 0, Hx[:, :-1], Hx[:, 1:])  # has shape (ny,nx+1)
        Qy = v * tf.where(v > 0, Hy[:-1, :], Hy[1:, :])  # has shape (ny+1,nx)

    elif method == 'centered':

        Qx = u * h  
        Qy = v * h  
        
        Qx = tf.concat(
            [Qx[:, 0:1], 0.5 * (Qx[:, :-1] + Qx[:, 1:]), Qx[:, -1:]], 1
        )  # has shape (ny,nx+1) 
        
        Qy = tf.concat(
            [Qy[0:1, :], 0.5 * (Qy[:-1, :] + Qy[1:, :]), Qy[-1:, :]], 0
        )  # has shape (ny+1,nx)
        

#    Qx = gaussian_filter_tf(Qx, sigma=2.0, kernel_size=13)
#    Qy = gaussian_filter_tf(Qy, sigma=2.0, kernel_size=13)

    ## Computation of the divergence, final shape is (ny,nx)
    return (Qx[:, 1:] - Qx[:, :-1]) / dx + (Qy[1:, :] - Qy[:-1, :]) / dy
