#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import matplotlib.pyplot as plt
import os, glob, shutil, scipy
from netCDF4 import Dataset
import pandas as pd
#from igm.processes.utils import complete_data
import json

def run(cfg, state):

    path_data = os.path.join(state.original_cwd,cfg.core.folder_data)

    if cfg.inputs.oggm_shop.RGI_ID=="":
        path_RGIs = [os.path.join(path_data,path_RGI) for path_RGI in cfg.inputs.oggm_shop.RGI_IDs]
    else:
        path_RGIs = [os.path.join(path_data,cfg.inputs.oggm_shop.RGI_ID)]

    path_file = os.path.join(path_data,cfg.inputs.oggm_shop.filename)

    if not os.path.exists(path_data):
        os.makedirs(path_data)

    # Fetch the data from OGGM if it does not exist
    if not all(os.path.exists(p) for p in path_RGIs):
        _oggm_util(cfg, path_RGIs)

    # transform the data into IGM readable data if it does not exist
    if not os.path.exists(path_file):
        transform_OGGM_data_into_IGM_readable_data(cfg, state, path_RGIs[0], path_file)

def transform_OGGM_data_into_IGM_readable_data(cfg, state, path_RGI, path_file):
    
    ncpath = os.path.join(path_RGI, "gridded_data.nc")
    if not os.path.exists(ncpath):
        msg = f'OGGM data issue with glacier {cfg.inputs.oggm_shop.RGI_ID}'
        if hasattr(state, "logger"):
            state.logger.info(msg)
        else:
            print(msg)
        return

    # if hasattr(state, "logger"):
    #     state.logger.info("Prepare data using oggm and glathida")

    nc = Dataset(ncpath, "r+")

    x = np.squeeze(nc.variables["x"]).astype("float32")
    y = np.flip(np.squeeze(nc.variables["y"]).astype("float32"))

    if hasattr(nc, 'pyproj_srs'):
        pyproj_srs = nc.pyproj_srs
    else:
        pyproj_srs = None

    #If you know that grids above a certain size are going to make your GPU memory explode,
    #activating this commented block and setting the number in the if statement to your
    #maximum threshold will cause IGM to skip execution and move on to the next one
    #(only relevant if using igm_run_batch, hence why I've not made it a parameter yet)
    # if len(x)*len(y) >= 160000:
    #     print('Skipping this one: '+cfg.inputs.oggm_shop.RGI_ID)
    #     return

    try:
        thk = np.flipud(np.squeeze(nc.variables[cfg.inputs.oggm_shop.thk_source]).astype("float32"))
        thk = np.where(np.isnan(thk), 0, thk)
    except:
        thk = np.flipud(np.squeeze(nc.variables["topo"]).astype("float32"))
        thk = np.where(True,0,0)
        print('Thickness 0 everywhere?: '+cfg.inputs.oggm_shop.RGI_ID)

    usurf = np.flipud(np.squeeze(nc.variables["topo"]).astype("float32"))
    usurfobs = np.flipud(np.squeeze(nc.variables["topo"]).astype("float32"))

    #This will set up some additional masks that are necessary for infer_params in optimize.
    #One of individually numbered RGI7.0G entities within each RGI7.0C and one for which entities
    #(for any RGI version used) are tidewater glaciers as identified in the RGI
    if cfg.inputs.oggm_shop.sub_entity_mask == True:
        if cfg.inputs.oggm_shop.RGI_product == "C":
            icemask = np.flipud(np.squeeze(nc.variables["sub_entities"]).astype("float32"))
            icemaskobs = np.flipud(np.squeeze(nc.variables["sub_entities"]).astype("float32"))
            icemask = np.where(icemask > -1, icemask+1, icemask)
            icemaskobs = np.where(icemaskobs > -1, icemaskobs+1, icemaskobs)
            tidewatermask = icemask.copy()
            slopes = icemask.copy()
        else:
            icemask = np.flipud(np.squeeze(nc.variables["glacier_mask"]).astype("float32"))
            icemaskobs = np.flipud(np.squeeze(nc.variables["glacier_mask"]).astype("float32"))
            tidewatermask = icemask.copy()
            slopes = icemask.copy()
        _get_tidewater_termini_and_slopes(tidewatermask, slopes, [cfg.inputs.oggm_shop.RGI_ID], cfg)
        vars_to_save = ["usurf", "thk", "icemask", "usurfobs", "thkobs", "icemaskobs", "tidewatermask", "slopes"]
    else:
        icemask = np.flipud(np.squeeze(nc.variables["glacier_mask"]).astype("float32"))
        icemaskobs = np.flipud(np.squeeze(nc.variables["glacier_mask"]).astype("float32"))
        vars_to_save = ["usurf", "thk", "icemask", "usurfobs", "thkobs", "icemaskobs"]

    usurf    -= np.where(icemaskobs, 0, thk) # remove thk from usurf outside glacier mask
    usurfobs -= np.where(icemaskobs, 0, thk) # remove thk from usurf outside glacier mask
    thk       = np.where(icemaskobs, thk, 0) # remove thk from usurf outside glacier mask

    if cfg.inputs.oggm_shop.vel_source == "millan_ice_velocity":
        if "millan_vx" in nc.variables:
            uvelsurfobs = np.flipud(
                np.squeeze(nc.variables["millan_vx"]).astype("float32")
            )
            uvelsurfobs = np.where(icemaskobs, uvelsurfobs, 0)
            vars_to_save += ["uvelsurfobs"]
        if "millan_vy" in nc.variables:
            vvelsurfobs = np.flipud(
                np.squeeze(nc.variables["millan_vy"]).astype("float32")
            )
            vvelsurfobs = np.where(icemaskobs, vvelsurfobs, 0)
            vars_to_save += ["vvelsurfobs"]
        else:
            uvelsurfobs = np.where(icemaskobs, 0, 0)
            vvelsurfobs = np.where(icemaskobs, 0, 0)
            vars_to_save += ["uvelsurfobs"]
            vars_to_save += ["vvelsurfobs"]
    else:
        if "itslive_vx" in nc.variables:
            uvelsurfobs = np.flipud(
                np.squeeze(nc.variables["itslive_vx"]).astype("float32")
            )
            uvelsurfobs = np.where(icemaskobs, uvelsurfobs, 0)
            vars_to_save += ["uvelsurfobs"]
        if "itslive_vy" in nc.variables:
            vvelsurfobs = np.flipud(
                np.squeeze(nc.variables["itslive_vy"]).astype("float32")
            )
            vvelsurfobs = np.where(icemaskobs, vvelsurfobs, 0)
            vars_to_save += ["vvelsurfobs"]

    if cfg.inputs.oggm_shop.smooth_obs_vel:
       uvelsurfobs = scipy.signal.medfilt2d(uvelsurfobs, kernel_size=3) # remove outliers
       vvelsurfobs = scipy.signal.medfilt2d(vvelsurfobs, kernel_size=3) # remove outliers

    if cfg.inputs.oggm_shop.thk_source in nc.variables: # either "millan_ice_thickness" or "consensus_ice_thickness"
        thkinit = np.flipud(
            np.squeeze(nc.variables[cfg.inputs.oggm_shop.thk_source]).astype("float32")
        )
        thkinit = np.where(np.isnan(thkinit), 0, thkinit)
        thkinit = np.where(icemaskobs, thkinit, 0)
        vars_to_save += ["thkinit"]

    if "hugonnet_dhdt" in nc.variables:
        dhdt = np.flipud(
            np.squeeze(nc.variables["hugonnet_dhdt"]).astype("float32")
        )
        dhdt = np.where(np.isnan(dhdt), 0, dhdt)
        dhdt = np.where(icemaskobs, dhdt, 0)
        vars_to_save += ["dhdt"]

    thkobs = np.zeros_like(thk) * np.nan

    if cfg.inputs.oggm_shop.incl_glathida:
        if cfg.inputs.oggm_shop.RGI_version==6:
            with open(os.path.join(path_RGI, "glacier_grid.json"), "r") as f:
                data = json.load(f)
            proj = data["proj"]

            try:
                thkobs = _read_glathida(
                    x, y, usurfobs, proj, cfg.inputs.oggm_shop.path_glathida, state
                )
                thkobs = np.where(icemaskobs, thkobs, np.nan)
            except:
                thkobs = np.zeros_like(thk) * np.nan
        elif cfg.inputs.oggm_shop.RGI_version==7:
            path_glathida = os.path.join(path_RGI, "glathida_data.csv")

            try:
                thkobs = _read_glathida_v7(
                    x, y, path_glathida
                )
                thkobs = np.where(icemaskobs, thkobs, np.nan)
            except:
                thkobs = np.zeros_like(thk) * np.nan

    nc.close()

    ########################################################

    # transform from numpy to tensorflow

    # for var in ["x", "y"]:
    #     vars(state)[var] = tf.constant(vars()[var].astype("float32"))

    # if pyproj_srs is not None:
    #     vars(state)["pyproj_srs"] = pyproj_srs

    # for var in vars_to_save:
    #     vars(state)[var] = tf.Variable(vars()[var].astype("float32"), trainable=False)

    # complete_data(state)

    ########################################################

    var_info = {}
    var_info["thk"] = ["Ice Thickness", "m"]
    var_info["usurf"] = ["Surface Topography", "m"]
    var_info["icemaskobs"] = ["Accumulation Mask", "bool"]
    var_info["usurfobs"] = ["Surface Topography", "m"]
    var_info["thkobs"] = ["Ice Thickness", "m"]
    var_info["thkinit"] = ["Ice Thickness", "m"]
    var_info["uvelsurfobs"] = ["x surface velocity of ice", "m/y"]
    var_info["vvelsurfobs"] = ["y surface velocity of ice", "m/y"]
    var_info["icemask"] = ["Ice mask", "no unit"]
    var_info["dhdt"] = ["Ice thickness change", "m/y"]
    if cfg.inputs.oggm_shop.sub_entity_mask == True:
        var_info["tidewatermask"] = ["Tidewater glacier mask", "no unit"]
        var_info["slopes"] = ["Average glacier surface slope", "deg"]

    nc = Dataset(path_file, "w", format="NETCDF4")

    nc.createDimension("y", len(y))
    yn = nc.createVariable("y", np.dtype("float32").char, ("y",))
    yn.units = "m"
    yn.long_name = "y"
    yn.standard_name = "y"
    yn.axis = "Y"
    yn[:] = y

    nc.createDimension("x", len(x))
    xn = nc.createVariable("x", np.dtype("float32").char, ("x",))
    xn.units = "m"
    xn.long_name = "x"
    xn.standard_name = "x"
    xn.axis = "X"
    xn[:] = x

    if pyproj_srs is not None:
        nc.pyproj_srs = pyproj_srs

    for v in vars_to_save:
        E = nc.createVariable(v, np.dtype("float32").char, ("y", "x"))
        E.long_name = var_info[v][0]
        E.units = var_info[v][1]
        E.standard_name = v
        E[:] = vars()[v]

    nc.close()



#########################################################################


def _oggm_util(cfg, path_RGIs):
    """
    Function written by Fabien Maussion
    """

    import oggm.cfg as cfg_oggm # changed the name to avoid namespace conflicts with IGM's config
    from oggm import utils, workflow, tasks, graphics
    import xarray as xr

    if cfg.inputs.oggm_shop.RGI_ID=="":
        RGIs = cfg.inputs.oggm_shop.RGI_IDs
    else:
        RGIs = [cfg.inputs.oggm_shop.RGI_ID]

    if cfg.inputs.oggm_shop.preprocess:
        # This uses OGGM preprocessed directories
        # I think that a minimal environment should be enough for this to run
        # Required packages:
        #   - numpy
        #   - geopandas
        #   - salem
        #   - matplotlib
        #   - configobj
        #   - netcdf4
        #   - xarray
        #   - oggm

        # Initialize OGGM and set up the default run parameters
        cfg_oggm.initialize_minimal()

        cfg_oggm.PARAMS["continue_on_error"] = True
        cfg_oggm.PARAMS["use_multiprocessing"] = False

        WD = "OGGM-prepro"

        # Where to store the data for the run - should be somewhere you have access to
        cfg_oggm.PATHS["working_dir"] = utils.gettempdir(dirname=WD, reset=True)

        # We need the outlines here
        if cfg.inputs.oggm_shop.RGI_version==6:
            rgi_ids = RGIs  # rgi_ids = utils.get_rgi_glacier_entities(RGIs)
            base_url = ( "https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/exps/igm_v2" )
            gdirs = workflow.init_glacier_directories(
                # Start from level 3 if you want some climate data in them
                rgi_ids,
                prepro_border=40,
                from_prepro_level=3,
                prepro_base_url=base_url,
            )
        else:
            rgi_ids = RGIs
            if cfg.inputs.oggm_shop.highres:
                base_url = ( "https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/exps/igm_v4_hr" )
            else:
                base_url = ( "https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/exps/igm_v4" )

            gdirs = workflow.init_glacier_directories(
                # Start from level 3 if you want some climate data in them
                rgi_ids,
                prepro_border=40,
                from_prepro_level=3,
                prepro_rgi_version='70'+cfg.inputs.oggm_shop.RGI_product,
                prepro_base_url=base_url,
            )
            if (cfg.inputs.oggm_shop.sub_entity_mask == True) & (cfg.inputs.oggm_shop.RGI_product == "C"):
                tasks.rgi7g_to_complex(gdirs[0])

    else:
        # Note: if you start from here you'll need most of the packages
        # needed by OGGM, since you start "from scratch" entirely
        # In my view this code should almost never be needed

        WD = "OGGM-dir"

        # Initialize OGGM and set up the default run parameters
        cfg_oggm.initialize()

        cfg_oggm.PARAMS["continue_on_error"] = False
        cfg_oggm.PARAMS["use_multiprocessing"] = False
        cfg_oggm.PARAMS["use_intersects"] = False

        # Map resolution parameters
        cfg_oggm.PARAMS["grid_dx_method"] = "fixed"
        cfg_oggm.PARAMS["fixed_dx"] = cfg.inputs.oggm_shop.dx  # m spacing
        cfg_oggm.PARAMS[
            "border"
        ] = (
            cfg.inputs.oggm_shop.border
        )  # can now be set to any value since we start from scratch
        cfg_oggm.PARAMS["map_proj"] = "utm"

        # Where to store the data for the run - should be somewhere you have access to
        cfg_oggm.PATHS["working_dir"] = utils.gettempdir(dirname=WD, reset=True)

        # We need the outlines here
        rgi_ids = utils.get_rgi_glacier_entities(RGIs)

        # Go - we start from scratch, i.e. we cant download from Bremen
        gdirs = workflow.init_glacier_directories(rgi_ids)

        # # gdirs is a list of glaciers. Let's pick one
        for gdir in gdirs:
            # https://oggm.org/tutorials/stable/notebooks/dem_sources.html
            tasks.define_glacier_region(gdir, source="DEM3")
            # Glacier masks and all
            tasks.simple_glacier_masks(gdir)

        # https://oggm.org/tutorials/master/notebooks/oggm_shop.html
        # If you want data we havent processed yet, you have to use OGGM shop
        from oggm.shop.millan22 import (
            thickness_to_gdir,
            velocity_to_gdir,
            compile_millan_statistics,
            compile_millan_statistics,
        )

        try:
            workflow.execute_entity_task(thickness_to_gdir, gdirs)
            workflow.execute_entity_task(velocity_to_gdir, gdirs)
        except ValueError:
            print("No millan22 velocity & thk data available!")

        # We also have some diagnostics if you want
        df = compile_millan_statistics(gdirs)
        #        print(df.T)

        from oggm.shop.its_live import velocity_to_gdir

        try:
            workflow.execute_entity_task(velocity_to_gdir, gdirs)
        except ValueError:
            print("No its_live velocity data available!")

        from oggm.shop import bedtopo

        workflow.execute_entity_task(bedtopo.add_consensus_thickness, gdirs)

        from oggm.shop import glathida

        workflow.execute_entity_task(glathida.glathida_to_gdir, gdirs)

        from oggm.shop.w5e5 import process_w5e5_data

        workflow.execute_entity_task(process_w5e5_data, gdirs)

        workflow.execute_entity_task(tasks.elevation_band_flowline, gdirs)
        workflow.execute_entity_task(tasks.fixed_dx_elevation_band_flowline, gdirs)
        workflow.execute_entity_task(tasks.mb_calibration_from_geodetic_mb,
                                                gdirs, informed_threestep=True)
    
    for gdir,path_RGI in zip(gdirs,path_RGIs):
        source_folder = gdir.get_filepath("gridded_data").split("gridded_data.nc")[0]
        if os.path.exists(path_RGI):
            shutil.rmtree(path_RGI)
        shutil.copytree(source_folder, path_RGI)

def _read_glathida(x, y, usurf, proj, path_glathida, state):
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

def _read_glathida_v7(x, y, path_glathida):
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

def _get_tidewater_termini_and_slopes(tidewatermask, slopes, RGIs, cfg):
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
        prepro_rgi_version='70'+cfg.inputs.oggm_shop.RGI_product,
        prepro_base_url=base_url,
    )
    if cfg.inputs.oggm_shop.RGI_product == "C":
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
