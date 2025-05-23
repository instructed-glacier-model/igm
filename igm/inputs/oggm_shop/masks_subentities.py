
#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file
 
import numpy as np
import os, glob, shutil 
import pandas as pd
import xarray as xr

# Subfunction to handle masks
def process_masks_subentities(ds, cfg, RGI_product):
    result = {} 
    if RGI_product == "C":
        icemask = ds["sub_entities"]
        icemask = xr.where(icemask > -1, icemask + 1, icemask)
    else:
        icemask = ds["glacier_mask"]
    result["icemask"] = icemask 
    result["tidewatermask"] = icemask.copy(deep=True)
    result["slopes"] = icemask.copy(deep=True)

    get_tidewater_termini_and_slopes(
        result["tidewatermask"].values, result["slopes"].values,[cfg.inputs.oggm_shop.RGI_ID], RGI_product)

    return result

def get_tidewater_termini_and_slopes(tidewatermask, slopes, RGIs, RGI_product):
    #Function written by Samuel Cook
    #Identify which glaciers in a complex are tidewater and also return average slope (both needed for infer_params in optimize)

    from oggm import utils, workflow, tasks, graphics
    import xarray as xr
    import matplotlib.pyplot as plt

    rgi_ids = RGIs
    base_url = ( "https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/exps/igm_v4" )
    gdirs = workflow.init_glacier_directories(
        # Start from level 3 if you want some climate data in them
        rgi_ids,
        prepro_border=40,
        from_prepro_level=3,
        prepro_rgi_version='70'+RGI_product,
        prepro_base_url=base_url,
    )
    if RGI_product == "C":
        tasks.rgi7g_to_complex(gdirs[0])
        gdf = gdirs[0].read_shapefile('complex_sub_entities')
        with xr.open_dataset(gdirs[0].get_filepath('gridded_data')) as ds:
            ds = ds.load()
            NumEntities = np.max(ds.sub_entities.values)+1
            for i in range(1,NumEntities+1):
                slopes[slopes==i] = gdf.loc[i-1].slope_deg
                if gdf.loc[i-1].term_type == 1:
                    tidewatermask[tidewatermask==i] = 1
                else:
                    tidewatermask[tidewatermask==i] = 0
    else:
        gdf = gdirs[0].read_shapefile('outlines')
        slopes[slopes==1] = gdf.loc[0].slope_deg
        if gdf.loc[0].term_type == '1':
            tidewatermask[tidewatermask==1] = 1
        else:
            tidewatermask[tidewatermask==1] = 0
