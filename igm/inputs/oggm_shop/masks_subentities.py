
#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file
 
import numpy as np
import os, glob, shutil 
import pandas as pd
import xarray as xr

# Subfunction to handle masks
def process_masks_subentities(ds, cfg, RGI_product, path_RGI):
    result = {} 
    if RGI_product == "C":
        icemask = ds["sub_entities"]
        icemask = xr.where(icemask > -1, icemask + 1, 0)
    else:
        icemask = ds["glacier_mask"]
    result["icemask"] = icemask 
    twmask = xr.open_dataset(path_RGI+'/tidewatermask.nc')
    if RGI_product == 'C':
        result["tidewatermask"] = twmask['sub_entities']
    else:
        result["tidewatermask"] = twmask['glacier_mask']

    #get_tidewater_termini(
    #    result["tidewatermask"].values,[cfg.inputs.oggm_shop.RGI_ID], RGI_product, path_RGI, ds)

    return result

def get_tidewater_termini(gdir, RGI_product, path_RGI):
    #Function written by Samuel Cook
    #Identify which glaciers in a complex are tidewater

    from oggm import utils, workflow, tasks, graphics
    import xarray as xr

    with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
        ds = ds.load()
        if RGI_product == "C":
            tidewatermask = ds.sub_entities.copy(deep=True)
            gdf = gdir.read_shapefile('complex_sub_entities')
            
            NumEntities = np.max(ds.sub_entities.values)+1
            for i in range(1,NumEntities+1):
                if gdf.loc[i-1].term_type == 1:
                    tidewatermask.values[tidewatermask.values==i] = 1
                else:
                    tidewatermask.values[tidewatermask.values==i] = 0
        else:
            tidewatermask = ds.glacier_mask.copy(deep=True)
            gdf = gdir.read_shapefile('outlines')
            if gdf.loc[0].term_type == '1':
                tidewatermask.values[tidewatermask.values==1] = 1
            else:
                tidewatermask.values[tidewatermask.values==1] = 0
            tidewatermask.values[ds.glacier_mask.values==0] = -1
          
    tidewatermask.to_netcdf(path_RGI[0]+'/tidewatermask.nc', format='NETCDF4')
