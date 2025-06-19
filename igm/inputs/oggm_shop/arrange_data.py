#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import os, glob
import json
import numpy as np
import xarray as xr
import pandas as pd
import scipy.signal

from igm.inputs.oggm_shop.read_glathida import read_glathida_v6, read_glathida_v7
from igm.inputs.oggm_shop.masks_subentities import process_masks_subentities
 
def arrange_data(cfg, state, path_RGI, ds, RGI_version, RGI_product):
 
    # Prepare output dictionary
    ds_vars = {}

    # Load surface topo
    ds_vars["usurf"] = ds["topo"] 

    # Load thickness
    thk_name = cfg.inputs.oggm_shop.thk_source
    if thk_name in ds:
        thk = ds[thk_name].fillna(0)
    else:
        thk = ds["topo"] * 0
    ds_vars["thk"] = thk

    # Process masks
    if cfg.inputs.oggm_shop.sub_entity_mask:
        mask_vars = process_masks_subentities(ds, cfg, RGI_product, path_RGI)
        ds_vars.update(mask_vars)
    else:
        ds_vars["icemask"]    = ds["glacier_mask"] 

    # Apply glacier mask
    ds_vars["usurf"] = ds_vars["usurf"] - xr.where(ds_vars["icemask"], 0, ds_vars["thk"])
    ds_vars["thk"]   = xr.where(ds_vars["icemask"], ds_vars["thk"], 0)

    # Optional thickness initialization
    if thk_name in ds:
        thkinit = ds[thk_name].fillna(0)
        thkinit = xr.where(ds_vars["icemask"], thkinit, 0)
        ds_vars["thkinit"] = thkinit

    # Optional dhdt
    if "hugonnet_dhdt" in ds:
        dhdt = ds["hugonnet_dhdt"].fillna(0)
        dhdt = xr.where(ds_vars["icemask"], dhdt, 0)
        ds_vars["dhdt"] = dhdt

    # Velocities
    vx = "millan_vx" if cfg.inputs.oggm_shop.vel_source == "millan_ice_velocity" else "itslive_vx"
    vy = "millan_vy" if cfg.inputs.oggm_shop.vel_source == "millan_ice_velocity" else "itslive_vy"
    for key, src in zip(["uvelsurfobs", "vvelsurfobs"], [vx, vy]):
        if src in ds:
            vel = ds[src]
            vel = xr.where(ds_vars["icemask"], vel, 0)
        else:
            vel = np.full(ds["topo"].shape, np.nan)
        if cfg.inputs.oggm_shop.smooth_obs_vel:
            vel.data = scipy.signal.medfilt2d(vel.values, kernel_size=3)
        ds_vars[key] = vel
 
    # Ice thickness observations from GlaThiDa
    if cfg.inputs.oggm_shop.incl_glathida:
        if RGI_version == 6:
            with open(os.path.join(path_RGI, "glacier_grid.json")) as f:
                proj = json.load(f)["proj"]
            thkobs = read_glathida_v6(ds.x.values, ds.y.values, ds_vars["usurfobs"].values, \
                                      proj, cfg.inputs.oggm_shop.path_glathida, state)
        else:
            path_glathida = os.path.join(path_RGI, "glathida_data.csv")
            thkobs = read_glathida_v7(ds.x.values, ds.y.values, path_glathida)
        thkobs = xr.DataArray(np.where(ds_vars["icemask"], thkobs, np.nan), dims=("y", "x"))
    else:
        thkobs = np.full(ds["topo"].shape, np.nan)   
    ds_vars["thkobs"] = thkobs

    # observed quantities
    ds_vars["usurfobs"] = ds_vars["usurf"]
    ds_vars["icemaskobs"] = ds_vars["icemask"]

    return ds_vars
