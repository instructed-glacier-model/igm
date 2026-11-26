#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Load a reference surface velocity for validating the error estimator."""

from typing import Dict, Tuple

import numpy as np


def load_reference_surface_velocity(
    path: str, expected_shape: Tuple[int, int]
) -> Dict[str, np.ndarray]:
    """Read ``uvelsurf``/``vvelsurf`` (last time slice) from an IGM NetCDF.

    Returns float64 arrays of shape ``expected_shape`` with NaN replaced by 0.
    """
    from netCDF4 import Dataset

    with Dataset(path, "r") as ds:
        out = {}
        for name in ("uvelsurf", "vvelsurf"):
            if name not in ds.variables:
                raise KeyError(
                    f"❌ Reference file {path} has no variable '{name}'."
                )
            array = np.asarray(ds.variables[name][:], dtype=np.float64)
            if array.ndim == 3:
                array = array[-1]
            if array.ndim != 2:
                raise ValueError(
                    f"❌ Reference variable '{name}' must be 2-D or 3-D, got {array.shape}."
                )
            if tuple(array.shape) != tuple(expected_shape):
                raise ValueError(
                    f"❌ Reference '{name}' has shape {array.shape}, but the "
                    f"simulation grid is {tuple(expected_shape)}."
                )
            out[name] = np.nan_to_num(np.ma.filled(array, np.nan), nan=0.0)
    return {"u_ref": out["uvelsurf"], "v_ref": out["vvelsurf"]}
