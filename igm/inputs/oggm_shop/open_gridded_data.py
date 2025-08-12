#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file
 
import numpy as np
import os
import xarray as xr  

def open_gridded_data(cfg, path_RGI, state, flip_y=True):

    ncpath = os.path.join(path_RGI, "gridded_data.nc")
    if not os.path.exists(ncpath):
        msg = f'OGGM data issue with glacier {cfg.inputs.oggm_shop.RGI_ID}'
        if hasattr(state, "logger"):
            state.logger.info(msg)
        else:
            print(msg)
        return

    ds = xr.open_dataset(ncpath)
    attr = ds.attrs
    # Convert all data to float32
    ds = ds.map(lambda x: x.astype("float32") if hasattr(x, 'dtype') and np.issubdtype(x.dtype, np.floating) else x)
    ds.attrs = attr
    # Ensure coordinates are set
    ds = ds.assign_coords({
        "x": ds["x"].squeeze().astype("float32"),
        "y": ds["y"].squeeze().astype("float32")
    })

    # Flip y-axis and all 2D variables
    if flip_y:
        ds = ds.sortby("y", ascending=True)
        for name, da in ds.data_vars.items():
            if da.ndim == 2:
                ds[name] = da[::-1, :]

    return ds