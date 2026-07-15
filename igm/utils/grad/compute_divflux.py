import tensorflow as tf

from igm.utils.math.gaussian_filter_tf import gaussian_filter_tf

@tf.function()
def compute_divflux(u, v, h, dx, dy, method='upwind', smooth_sigma=0.0):
    """
    upwind computation of the divergence of the flux : d(u h)/dx + d(v h)/dy
    First, u and v are computed on the staggered grid (i.e. cell edges)
    Second, one extend h horizontally by a cell layer on any bords (assuming same value)
    Third, one compute the flux on the staggered grid slecting upwind quantities
    Last, computing the divergence on the staggered grid yields values def on the original grid

    smooth_sigma > 0 applies a Gaussian filter (std smooth_sigma, in cells) to the
    staggered fluxes Qx, Qy BEFORE taking the divergence. The result is the exact
    discrete divergence of a (filtered) flux field, so it is conservative: the
    global budget sum(divflux)*dx*dy is preserved up to boundary effects, unlike
    smoothing divflux after the fact. This tames the grid-scale noise that the
    derivative promotes to leading order (emulator jitter, upwind switching,
    rough data geometry) while leaving long wavelengths nearly untouched.
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

    if smooth_sigma > 0.0:
        kernel_size = 2 * int(3 * smooth_sigma) + 1  # spans +/- 3 sigma
        Qx = gaussian_filter_tf(Qx, sigma=smooth_sigma, kernel_size=kernel_size)
        Qy = gaussian_filter_tf(Qy, sigma=smooth_sigma, kernel_size=kernel_size)

    ## Computation of the divergence, final shape is (ny,nx)
    return (Qx[:, 1:] - Qx[:, :-1]) / dx + (Qy[1:, :] - Qy[:-1, :]) / dy
