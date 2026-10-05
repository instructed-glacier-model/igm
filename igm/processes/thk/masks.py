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

from typing import Optional, Sequence

import tensorflow as tf

#: Water level (m) meaning "no ocean"
WATER_LEVEL_NO_OCEAN = -1.0e6


def no_ocean_like(field: tf.Tensor) -> tf.Tensor:
    """Return a "no ocean" water level with the shape and dtype of ``field``."""
    return tf.fill(tf.shape(field), tf.cast(WATER_LEVEL_NO_OCEAN, field.dtype))


def flotation_function(
    thk: tf.Tensor,
    topg: tf.Tensor,
    water_level: tf.Tensor,
    rho_ratio: float,
) -> tf.Tensor:
    """Return the flotation function, positive where ice is grounded.

    ``rho_ratio`` is water density divided by ice density. The function is
    ``thk + rho_ratio * (topg - water_level)`` minus a float32 round-off
    tolerance, so ice within round-off of flotation counts as floating: with
    ``topg = usurf - thk`` (the ice flow), floating ice is exactly at
    flotation.
    """
    dtype = thk.dtype
    topg = tf.cast(topg, dtype)
    phi = thk + tf.cast(rho_ratio, dtype) * (topg - tf.cast(water_level, dtype))
    tol = tf.cast(64.0 * 2.0**-23, dtype) * (thk + tf.abs(topg))  # 64 float32 ulps
    return phi - tol


def compute_grounded_mask(
    thk: tf.Tensor,
    topg: tf.Tensor,
    water_level: tf.Tensor,
    rho_ratio: float,
) -> tf.Tensor:
    """Return a boolean mask where ice is grounded (see ``flotation_function``)."""
    return flotation_function(thk, topg, water_level, rho_ratio) > 0.0


def _all_cell_corners(mask: tf.Tensor) -> tf.Tensor:
    """Return cells for which all four corner nodes are true."""
    return (
        mask[..., :-1, :-1]
        & mask[..., :-1, 1:]
        & mask[..., 1:, :-1]
        & mask[..., 1:, 1:]
    )


def _any_cell_corner(mask: tf.Tensor) -> tf.Tensor:
    """Return cells for which at least one corner node is true."""
    return (
        mask[..., :-1, :-1]
        | mask[..., :-1, 1:]
        | mask[..., 1:, :-1]
        | mask[..., 1:, 1:]
    )


@tf.function(jit_compile=True)
def compute_cell_ice_mask(
    thk: tf.Tensor, grounded: Optional[tf.Tensor] = None
) -> tf.Tensor:
    """Return Q1 cells with mechanically supported velocity degrees of freedom.

    A full four-ice-node cell is valid whether grounded or floating.  A
    partially covered cell is valid only when it contains ice and all four
    corner locations are grounded; this retains land margins without adding
    unsupported degrees of freedom at a floating front.  If ``grounded`` is
    omitted, only full ice cells are returned.
    """
    ice = thk > 0.0
    full_ice = _all_cell_corners(ice)
    if grounded is None:
        return full_ice
    full_grounded = _all_cell_corners(tf.cast(grounded, tf.bool))
    return full_ice | (_any_cell_corner(ice) & full_grounded)


def _pad_cell_mask_to_nodes(
    cell_mask: tf.Tensor,
    top: int,
    bottom: int,
    left: int,
    right: int,
) -> tf.Tensor:
    rank = cell_mask.shape.rank
    if rank is None:
        raise ValueError("cell_mask rank must be statically known.")
    paddings = [[0, 0]] * (rank - 2) + [[top, bottom], [left, right]]
    return tf.pad(cell_mask, paddings)


@tf.function(jit_compile=True)
def compute_node_ice_mask(
    thk: tf.Tensor, grounded: Optional[tf.Tensor] = None
) -> tf.Tensor:
    """Return ice nodes belonging to at least one active Q1 cell."""
    cell_mask = compute_cell_ice_mask(thk, grounded)
    node_support = (
        _pad_cell_mask_to_nodes(cell_mask, 0, 1, 0, 1)
        | _pad_cell_mask_to_nodes(cell_mask, 0, 1, 1, 0)
        | _pad_cell_mask_to_nodes(cell_mask, 1, 0, 0, 1)
        | _pad_cell_mask_to_nodes(cell_mask, 1, 0, 1, 0)
    )
    return (thk > 0.0) & node_support


def iceflow_node_mask(
    thk: tf.Tensor,
    usurf: tf.Tensor,
    water_level: tf.Tensor,
    rho_ratio: float,
) -> tf.Tensor:
    """Ice nodes that carry velocity in the ice flow.

    This is the unified evaluator's node mask: flotation is tested on the
    lower surface ``usurf - thk``, and a node is kept when it belongs to an
    active Q1 cell (:func:`compute_node_ice_mask`). A floating or marine
    front is therefore sharp on the boundary of full cells, while land
    margins keep their partially covered, fully grounded cells.
    """
    grounded = compute_grounded_mask(thk, usurf - thk, water_level, rho_ratio)
    return compute_node_ice_mask(thk, grounded)


def mask_gr(
    thk: tf.Tensor,
    topg: tf.Tensor,
    water_level: tf.Tensor,
    rho_ratio: float,
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
