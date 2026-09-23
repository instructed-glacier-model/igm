#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Flotation: the water level and the grounded mask.

Every flotation test in IGM (thickness, ice flow, error estimator) goes
through :func:`compute_grounded_mask`,

    grounded  <=>  thk + (rho_water / rho_ice) * (topg - water_level) > 0,

and the lower ice surface is ``lsurf = max(topg, water_level - r * thk)``.
``state.water_level`` is therefore always present:

* ``inputs.<method>.water_level.include: true`` gives a uniform sea/lake
  level equal to ``value`` (e.g. ``0.0`` for present-day sea level);
* a ``water_level`` variable in the input file is used as is;
* otherwise the domain has **no ocean** and the field is filled with
  :data:`WATER_LEVEL_NO_OCEAN`.

The "no ocean" level is a finite value far below any bed, so every formula
above degenerates to the land-only case without a special branch: all ice is
grounded, ``lsurf == topg`` exactly, the ocean depth ``max(wl - topg, 0)`` is
zero (no marine calving, full overburden for ``ocean_connected``
hydrology). This is what a synthetic domain with a bed below 0 m and no
ocean (e.g. ISMIP-HOM) needs; a zero default would instead make such ice
float. Mountain glaciers (bed above 0 m) behave identically with either.
"""

from typing import Sequence

import tensorflow as tf

#: Water level (m) meaning "no ocean"
WATER_LEVEL_NO_OCEAN = -1.0e6


def no_ocean_like(field: tf.Tensor) -> tf.Tensor:
    """Return a "no ocean" water level with the shape and dtype of ``field``."""
    return tf.fill(tf.shape(field), tf.cast(WATER_LEVEL_NO_OCEAN, field.dtype))


def compute_grounded_mask(
    thk: tf.Tensor,
    topg: tf.Tensor,
    water_level: tf.Tensor,
    rho_ratio,
) -> tf.Tensor:
    """Return a boolean mask where ice is grounded.

    ``rho_ratio`` is water density divided by ice density. The threshold is
    strict: ice exactly at flotation counts as floating.
    """
    dtype = thk.dtype
    phi = thk + tf.cast(rho_ratio, dtype) * (
        tf.cast(topg, dtype) - tf.cast(water_level, dtype)
    )
    return phi > 0.0


def mask_gr(
    thk: tf.Tensor,
    topg: tf.Tensor,
    water_level: tf.Tensor,
    rho_ratio,
) -> tf.Tensor:
    """The grounded mask in ``thk`` dtype (1 grounded, 0 floating)."""
    return tf.cast(compute_grounded_mask(thk, topg, water_level, rho_ratio), thk.dtype)


def water_level_from_inputs(
    inputs: tf.Tensor, input_names: Sequence[str], like: tf.Tensor
) -> tf.Tensor:
    """Return the ``water_level`` channel of ``inputs``, or "no ocean".

    ``water_level`` is not a default iceflow input (it would be a network
    input channel); ``initialize_iceflow_fields`` warns when a domain with an
    ocean does not list it. ``like`` sets shape and dtype.
    """
    if "water_level" in input_names:
        return inputs[..., tuple(input_names).index("water_level")]
    return no_ocean_like(like)
