import tensorflow as tf

from .grad import pad_x, pad_y
from ..stag.stag import stag2x, stag2y


def minmod(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    return tf.where(
        (tf.abs(a) < tf.abs(b)) & (a * b > 0.0),
        a,
        tf.where((tf.abs(a) > tf.abs(b)) & (a * b > 0.0), b, 0),
    )


def maxmod(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    return tf.where(
        (tf.abs(a) < tf.abs(b)) & (a * b > 0.0),
        b,
        tf.where((tf.abs(a) > tf.abs(b)) & (a * b > 0.0), a, 0),
    )


@tf.function()
def compute_divflux_slope_limiter(
    u: tf.Tensor,
    v: tf.Tensor,
    h: tf.Tensor,
    dx: tf.Tensor,
    dy: tf.Tensor,
    dt: tf.Tensor,
    slope_type: str = "superbee",
    mode_h: str = "zero",
    mode_u: str = "symmetric",
):
    """
    upwind computation of the divergence of the flux : d(u h)/dx + d(v h)/dy
    propose a slope limiter for the upwind scheme with 3 options : godunov, minmod, superbee

    References :
    - Numerical Methods for Engineers, Leif Rune Hellevik, book
      https://folk.ntnu.no/leifh/teaching/tkt4140/._main074.html

    - hydro_examples github page, Michael Zingale, Ian Hawke
     collection of simple python codes that demonstrate some basic techniques used in hydrodynamics codes.
     https://github.com/python-hydro/hydro_examples
    """

    # Stagger velocities to cell faces after padding
    u = stag2x(pad_x(u, mode_u))  # (ny,nx+1)
    v = stag2y(pad_y(v, mode_u))  # (ny+1,nx)

    # Pad h with 2 ghost cells for slope computation
    Hx = pad_x(pad_x(h, mode_h), mode_h)  # (ny,nx+4)
    Hy = pad_y(pad_y(h, mode_h), mode_h)  # (ny+4,nx)

    sigpx = (Hx[:, 2:] - Hx[:, 1:-1]) / dx  # (ny,nx+2)
    sigmx = (Hx[:, 1:-1] - Hx[:, :-2]) / dx  # (ny,nx+2)

    sigpy = (Hy[2:, :] - Hy[1:-1, :]) / dy  # (ny+2,nx)
    sigmy = (Hy[1:-1, :] - Hy[:-2, :]) / dy  # (ny+2,nx)

    if slope_type == "godunov":

        slopex = tf.zeros_like(sigpx)
        slopey = tf.zeros_like(sigpy)

    elif slope_type == "minmod":

        slopex = minmod(sigmx, sigpx)
        slopey = minmod(sigmy, sigpy)

    elif slope_type == "superbee":

        sig1x = minmod(sigpx, 2.0 * sigmx)
        sig2x = minmod(sigmx, 2.0 * sigpx)
        slopex = maxmod(sig1x, sig2x)

        sig1y = minmod(sigpy, 2.0 * sigmy)
        sig2y = minmod(sigmy, 2.0 * sigpy)
        slopey = maxmod(sig1y, sig2y)

    w = Hx[:, 1:-2] + 0.5 * dx * (1.0 - u * dt / dx) * slopex[:, :-1]  #  (ny,nx+1)
    e = Hx[:, 2:-1] - 0.5 * dx * (1.0 + u * dt / dx) * slopex[:, 1:]  #  (ny,nx+1)

    s = Hy[1:-2, :] + 0.5 * dy * (1.0 - v * dt / dy) * slopey[:-1, :]  #  (ny+1,nx)
    n = Hy[2:-1, :] - 0.5 * dy * (1.0 + v * dt / dy) * slopey[1:, :]  #  (ny+1,nx)

    Qx = u * tf.where(u > 0, w, e)  #  (ny,nx+1)
    Qy = v * tf.where(v > 0, s, n)  #  (ny+1,nx)

    return (Qx[:, 1:] - Qx[:, :-1]) / dx + (Qy[1:, :] - Qy[:-1, :]) / dy
