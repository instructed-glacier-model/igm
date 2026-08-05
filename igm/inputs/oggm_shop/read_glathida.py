
#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file
 
import numpy as np
import os, glob, shutil 
import pandas as pd
 
def read_glathida(x, y, path_glathida):
    #Function written by Samuel Cook

    #Read GlaThiDa file
    gdf = pd.read_csv(path_glathida)

    gdf_sel = gdf.loc[gdf.thickness > 0]  # you may not want to do that, but be aware of: https://gitlab.com/wgms/glathida/-/issues/25
    gdf_per_grid = gdf_sel.groupby(by='ij_grid')[['i_grid', 'j_grid', 'elevation', 'thickness', 'thickness_uncertainty']].mean()  # just average per grid point
    # Average does not preserve ints
    gdf_per_grid['i_grid'] = gdf_per_grid['i_grid'].astype(int)
    gdf_per_grid['j_grid'] = gdf_per_grid['j_grid'].astype(int)

    #Get GlaThiDa data onto model grid
    thkobs = np.full((y.shape[0], x.shape[0]), np.nan)
    thkobs[gdf_per_grid['j_grid'],gdf_per_grid['i_grid']] = gdf_per_grid['thickness']
    thkobs = np.flipud(thkobs)

    return thkobs