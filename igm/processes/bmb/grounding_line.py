#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Melt near the grounding line.

Each node stands for its dual cell, made of one quarter of each of its four
Q1 cells. The floating fraction ``f`` of that dual cell is estimated by
sampling the bilinear interpolant of the flotation function on n x n points
per quarter. Only the ice shelf floats: ``f`` is 0 at grounded nodes without
a shelf node among their 8 neighbours, such as the front of a tidewater
glacier next to ice-free water.

The ocean melt (weight ``w``) applies to the floating part of the cell and
the grounded, thermodynamic melt (weight ``g``) to its grounded part,
following the no-melt and full-melt treatments of Seroussi and Morlighem
(2018) and the partial-melt treatment of Leguy et al. (2021), the dual-cell
analogue of their sub-element melt:

    nmp   w = 1[f = 1]   g = 1 - f   no ocean melt where the grounding line
                                     crosses the cell
    fmp   w = 1[f > 0]   g = 1 - w   full ocean melt wherever the cell floats
    pmp   w = f          g = 1 - f   each melt in proportion to its area

so a cell never receives more than one melt in full. The ocean melt applies
to the shelf and its grounded neighbours only, not over subglacial lakes.
Each melt is known on its own side of the grounding line (the ocean melt on
the shelf, the thermodynamic melt on grounded ice) and is carried across it
as the mean over the 8 neighbours on that side, as PISM's
``extend_basal_melt_rates`` does for the ocean melt.
"""

from typing import Tuple

import tensorflow as tf

from .geometry import Geometry, floating_fraction
from .utils import DIAGONAL, EDGE, any_neighbour, neighbours

TREATMENTS = ("fmp", "nmp", "pmp")


def melt_weights(
    treatment: str, fraction: tf.Tensor, grounded: tf.Tensor, shelf: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Weights ``(w, g)`` of the ocean and of the grounded melt."""
    if treatment == "nmp":
        w = tf.cast(fraction >= 1.0, fraction.dtype)
        g = 1.0 - fraction
    elif treatment == "fmp":
        w = tf.cast(fraction > 0.0, fraction.dtype)
        g = 1.0 - w
    else:
        w, g = fraction, 1.0 - fraction
    return w * tf.cast(shelf | grounded, fraction.dtype), g


def extend(field: tf.Tensor, support: tf.Tensor) -> tf.Tensor:
    """``field`` on ``support``, elsewhere the mean over support 8-neighbours.

    Nodes without a neighbour on ``support`` get 0.
    """
    weight = tf.cast(support, field.dtype)
    offsets = EDGE + DIAGONAL
    total = tf.add_n(neighbours(tf.where(support, field, 0.0), offsets))
    count = tf.add_n(neighbours(weight, offsets))
    return tf.where(support, field, tf.math.divide_no_nan(total, count))


@tf.function(autograph=False, jit_compile=True)
def _assemble(
    phi: tf.Tensor,
    ice: tf.Tensor,
    grounded: tf.Tensor,
    shelf: tf.Tensor,
    ocean_melt: tf.Tensor,
    grounded_melt: tf.Tensor,
    treatment: str,
    samples: int,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    fraction = floating_fraction(phi, samples)
    near_shelf = shelf | any_neighbour(shelf, EDGE + DIAGONAL)
    fraction = tf.where(grounded & ~near_shelf, 0.0, fraction)
    w, g = melt_weights(treatment, fraction, grounded, shelf)
    ocean = tf.where(shelf, ocean_melt, extend(ocean_melt, shelf))
    thermal = extend(grounded_melt, grounded & ice)
    bmb = tf.where(ice, -(g * thermal + w * ocean), 0.0)
    return bmb, 1.0 - fraction, tf.where(shelf, ocean, 0.0)


def basal_mass_balance(
    treatment: str,
    samples: int,
    geom: Geometry,
    ocean_melt: tf.Tensor,
    grounded_melt: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Basal mass balance, grounded fraction and sub-shelf melt of each node.

    ``ocean_melt`` is the sub-shelf melt rate on the shelf (possibly as last
    evaluated, extended to its neighbours) and ``grounded_melt`` the
    thermodynamic melt rate, both positive for melt.
    """
    return _assemble(
        geom.phi,
        geom.ice,
        geom.grounded,
        geom.shelf,
        tf.convert_to_tensor(ocean_melt),
        tf.convert_to_tensor(grounded_melt),
        treatment,
        int(samples),
    )
