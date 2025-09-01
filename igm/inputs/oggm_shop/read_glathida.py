
#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file
 
import numpy as np
import os, glob, shutil 
import pandas as pd
 
def read_glathida_v6(x, y, usurf, proj, path_glathida, state):
    """
    Function written by Ethan Welthy, Guillaume Jouvet and Samuel Cook
    """

    from pyproj import Transformer
    from scipy.interpolate import RectBivariateSpline
    import pandas as pd

    if path_glathida == "":
        path_glathida = os.path.expanduser("~")

    if not os.path.exists(os.path.join(path_glathida, "glathida")):
        os.system("git clone https://gitlab.com/wgms/glathida " + path_glathida)
    else:
        if hasattr(state, "logger"):
            state.logger.info("glathida data already at " + path_glathida)

    files = glob.glob(os.path.join(path_glathida, "glathida", "data", "*", "point.csv"))
    files += glob.glob(os.path.join(path_glathida, "glathida", "data", "point.csv"))

    os.path.expanduser

    transformer = Transformer.from_crs(proj, "epsg:4326", always_xy=True)

    lonmin, latmin = transformer.transform(min(x), min(y))
    lonmax, latmax = transformer.transform(max(x), max(y))

    transformer = Transformer.from_crs("epsg:4326", proj, always_xy=True)

    #    print(x.shape, y.shape, usurf.shape)

    fsurf = RectBivariateSpline(x, y, np.transpose(usurf))

    df = pd.concat(
        [pd.read_csv(file, low_memory=False) for file in files], ignore_index=True
    )


    mask = (
        (lonmin <= df["longitude"])
        & (df["longitude"] <= lonmax)
        & (latmin <= df["latitude"])
        & (df["latitude"] <= latmax)
        & df["elevation"].notnull()
        & df["date"].notnull()
        & df["elevation_date"].notnull()
    )
    df = df[mask]

    # Filter by date gap in second step for speed
    mask = (
        (
            df["date"].str.slice(0, 4).astype(int)
            - df["elevation_date"].str.slice(0, 4).astype(int)
        )
        .abs()
        .le(1)
    )
    df = df[mask]

    if df.index.shape[0] == 0:
        print("No ice thickness profiles found")
        thkobs = np.ones_like(usurf)
        thkobs[:] = np.nan

    else:
        if hasattr(state, "logger"):
            state.logger.info("Nb of profiles found : " + str(df.index.shape[0]))

        xx, yy = transformer.transform(df["longitude"], df["latitude"])
        bedrock = df["elevation"] - df["thickness"]
        elevation_normalized = fsurf(xx, yy, grid=False)
        thickness_normalized = np.maximum(elevation_normalized - bedrock, 0)

        dx = x[1]-x[0]
        dy = y[1]-y[0]
        
        # Rasterize thickness
        thickness_gridded = (
            pd.DataFrame(
                {
                    "col": np.floor((xx - np.min(x) + dx/2) / dx).astype(int),
                    "row": np.floor((yy - np.min(y) + dy/2) / dy).astype(int),
                    "thickness": thickness_normalized,
                }
            )
            .groupby(["row", "col"])["thickness"]
            .mean()
        )
        thkobs = np.full((y.shape[0], x.shape[0]), np.nan)
        thickness_gridded[thickness_gridded == 0] = np.nan
        thkobs[tuple(zip(*thickness_gridded.index))] = thickness_gridded

    return thkobs

def read_glathida_v7(x, y, path_glathida):
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