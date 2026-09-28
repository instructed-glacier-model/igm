#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Naming of the vertical dimension(s) when writing 3-D fields to NetCDF.

A single run can carry several *independent* vertical grids: `iceflow` resolves the
velocity field on `processes.iceflow.numerics.Nz` layers (often just 2 with the MOLHO
basis), while `enthalpy` resolves the thermal state on its own, much finer
`processes.enthalpy.numerics.Nz`. Writing both into one file therefore needs more than
a single `z` dimension.

The convention here: the iceflow grid keeps the plain name `z` (so existing files and
post-processing scripts are unaffected), and any other vertical size gets its own
dimension named `z<N>` — e.g. `z30` for a 30-layer enthalpy grid.
"""


def vertical_size_reference(cfg):
    """Vertical size that owns the plain `z` dimension name, or None if unknown."""

    if hasattr(cfg, "processes") and hasattr(cfg.processes, "iceflow"):
        return cfg.processes.iceflow.numerics.Nz

    return None


def vertical_dim_name(size, reference=None):
    """Dimension name for a 3-D field whose first axis has length `size`."""

    if reference is None or size == reference:
        return "z"

    return f"z{size}"


def vertical_dims_in_use(data_vars):
    """Map each vertical dimension name appearing in `data_vars` to its length.

    `data_vars` is a dict of xarray DataArrays already expanded along `time`, so their
    dims read `(time, <vertical>, y, x)` for 3-D fields and `(time, y, x)` for 2-D ones.
    """

    dims = {}
    for data in data_vars.values():
        if data.ndim != 4:
            continue
        name = data.dims[1]
        dims[name] = data.shape[1]

    return dims
