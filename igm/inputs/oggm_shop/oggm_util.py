#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file
 
import numpy as np
import os, glob, shutil

def oggm_util(cfg, path_RGIs, RGI_version, RGI_product):
    """
    Function written by Fabien Maussion
    """

    import oggm.cfg as cfg_oggm # changed the name to avoid namespace conflicts with IGM's config
    from oggm import utils, workflow, tasks, graphics
    from .masks_subentities import get_tidewater_termini

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
        if RGI_version==6:
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
                prepro_rgi_version='70'+RGI_product,
                prepro_base_url=base_url,
            )
            if (cfg.inputs.oggm_shop.sub_entity_mask == True) & (RGI_product == "C"):
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
        
    if cfg.inputs.oggm_shop.sub_entity_mask == True:
        get_tidewater_termini(
            gdirs[0], RGI_product, path_RGIs)