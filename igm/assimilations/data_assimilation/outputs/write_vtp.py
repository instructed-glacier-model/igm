#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import os, shutil
import numpy as np 
import tensorflow as tf 
 
def update_vtp(cfg, state, it):
    
    import pyvista as pv

    vtp_directory = "vtp"

    if it == 0:
        # Create output directory
        if os.path.exists(vtp_directory):
            shutil.rmtree(vtp_directory)
        os.makedirs(vtp_directory)
      
    # Apply mask to get valid points
    mask = state.X > 0
    x = state.X[mask].numpy()
    y = state.Y[mask].numpy()
    zb = state.topg[mask].numpy()
    zs = state.usurf[mask].numpy()
    
    # Create point cloud and triangulated surface
    points = np.vstack((x, y, zb)).T
    cloud = pv.PolyData(points)
    cloud["topg"] = zb
    surf1 = cloud.delaunay_2d()
     
    # Save topography
    filename = os.path.join(vtp_directory, f"topg-{it:06d}.vtp")
    surf1.save(filename)

    # Create point cloud and triangulated surface
    points = np.vstack((x, y, zs)).T
    cloud = pv.PolyData(points)
    cloud["usurf"] = zs
    surf2 = cloud.delaunay_2d()
     
    # Save topography
    filename = os.path.join(vtp_directory, f"usurf-{it:06d}.vtp")
    surf2.save(filename)
