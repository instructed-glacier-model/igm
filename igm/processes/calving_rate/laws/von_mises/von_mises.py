#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Von Mises tensile-stress calving (Morlighem et al., 2016).

    c = |u| sigma / sigma_max,  sigma = sqrt(3) B e^(1/n),
    e^2 = (max(e1, 0)^2 + max(e2, 0)^2) / 2,

with ``e1 >= e2`` the principal horizontal strain rates (1/yr) and
``B = A^(-1/n)`` the ice hardness (MPa yr^(1/n)) from the ice-flow rate
factor ``state.arrhenius`` (MPa^-n yr^-1), which already includes the
enhancement factor; a 3-D field is averaged over the column as ``B``, with
the vertical weights of the ice flow. ``sigma_max`` (MPa) differs for
floating and grounded ice (ISSM's defaults: 0.15 and 1 MPa). ``|u|`` is the
ice speed at the last ice node, as in PISM and ISSM: at ``sigma = sigma_max``
the rate falls short of the sub-grid front's advance speed by the speed-up
over the partial cell (a few percent, vanishing with the grid spacing).

Morlighem, M., Bondzio, J., Seroussi, H., Rignot, E., Larour, E., Humbert,
A., and Rebuffi, S.: Modeling of Store Gletscher's calving dynamics, West
Greenland, in response to ocean thermal forcing, Geophys. Res. Lett., 43,
2659-2666, 2016.
"""

import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State

from ...geometry import FrontGeometry
from ...strain_rates import principal_strain_rates

#: The rate is evaluated on the supported ice nodes and carried to the
#: front band by the process (see ``laws/__init__.py``).
ON_BAND = False


def glen_exponent(cfg: DictConfig) -> float:
    """Glen exponent n of the ice flow (3 without it)."""
    viscosity = cfg.processes.get("iceflow", {}).get("physics", {}).get("viscosity", {})
    return float(viscosity.get("exponent", 3.0))


def hardness(cfg: DictConfig, state: State, dtype: tf.DType) -> tf.Tensor:
    """Column-mean ice hardness B = A^(-1/n) (MPa yr^(1/n)).

    ``state.arrhenius`` already includes the enhancement factor. A 3-D field
    is averaged as ``B`` with the vertical weights of the ice flow (the
    ``arrhenius`` process convention), or uniformly when the weights are
    missing or do not match its layers.
    """
    n = glen_exponent(cfg)
    B = tf.pow(tf.cast(state.arrhenius, dtype), -1.0 / n)
    if B.shape.rank != 3:
        return B
    iceflow = getattr(state, "iceflow", None)
    weights = getattr(getattr(iceflow, "discr_v", None), "enthalpy", None)
    weights = getattr(weights, "weights", None)
    if weights is not None and weights.shape[0] == B.shape[0]:
        return tf.reduce_sum(B * tf.cast(weights, dtype), axis=0)
    return tf.reduce_mean(B, axis=0)


def calving_rate(cfg: DictConfig, state: State, geom: FrontGeometry) -> tf.Tensor:
    p = cfg.processes.calving_rate.von_mises
    dtype = geom.thk.dtype
    n = glen_exponent(cfg)
    e1, e2 = principal_strain_rates(
        tf.cast(state.ubar, dtype), tf.cast(state.vbar, dtype), geom.supported, geom.dx
    )
    tensile = tf.sqrt(
        0.5 * (tf.square(tf.maximum(e1, 0.0)) + tf.square(tf.maximum(e2, 0.0)))
    )
    sigma = 3.0**0.5 * hardness(cfg, state, dtype) * tf.pow(tensile, 1.0 / n)
    sigma_max = tf.where(
        geom.floating,
        float(p.sigma_max_floating) + tf.zeros_like(sigma),
        float(p.sigma_max_grounded) + tf.zeros_like(sigma),
    )
    rate = geom.speed * sigma / sigma_max
    return tf.where(geom.supported, rate, tf.zeros_like(rate))
