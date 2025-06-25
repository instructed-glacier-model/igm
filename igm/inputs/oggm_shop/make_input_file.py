#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file
 
import xarray as xr 

# Variable metadata
def build_var_info(cfg):
    info = {
        "thk": ["Ice Thickness", "m"],
        "usurf": ["Surface Topography", "m"],
        "usurfobs": ["Surface Topography", "m"],
        "thkobs": ["Ice Thickness", "m"],
        "thkinit": ["Ice Thickness", "m"],
        "uvelsurfobs": ["x surface velocity of ice", "m/y"],
        "vvelsurfobs": ["y surface velocity of ice", "m/y"],
        "icemask": ["Ice mask", "no unit"],
        "icemaskobs": ["Accumulation Mask", "bool"],
        "dhdt": ["Ice thickness change", "m/y"]
    }
    if cfg.inputs.oggm_shop.sub_entity_mask:
        info["tidewatermask"] = ["Tidewater glacier mask", "no unit"]
    return info

def make_input_file(cfg, ds, ds_vars, path_file):

    # Build output dataset
    coords = {
        "x": ds["x"],
        "y": ds["y"]
    }
    var_info = build_var_info(cfg)

    pyproj_srs = ds.attrs.get("pyproj_srs", None)

    ds_out = xr.Dataset(
        {
            v: xr.DataArray(data=ds_vars[v], dims=("y", "x"), attrs={
                "long_name": var_info[v][0],
                "units": var_info[v][1],
                "standard_name": v
            }) for v in ds_vars
        },
        coords=coords,
        attrs={"pyproj_srs": pyproj_srs} if pyproj_srs else {}
    )

    # Save to disk
    ds_out.to_netcdf(path_file, format="NETCDF4")
