#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import os
import tensorflow as tf

from pysheds.grid import Grid
import rasterio
from rasterio.transform import from_origin

def initialize(cfg, state):
    
    state.tlast_flow_acc = -10**10

#    state.flow_accumulation = tf.Variable(tf.zeros_like(state.thk), dtype=tf.float32)

#    state.phi = tf.Variable(tf.zeros_like(state.thk), dtype=tf.float32)

def update(cfg, state):

    if (state.t - state.tlast_flow_acc) >= cfg.processes.flow_accumulation.update_freq:

        ############################# Compute hydraulic potential phi #############################

        Pe0         = 0.69
        PCc         = 0.12
        Pdelta      = 0.02 
        PN0         = 1000 # Pa
        P0          = 910 * 9.81 * state.thk

        tilleffpres = np.minimum(P0, 
                                 PN0 * ( Pdelta * P0 / PN0 )**state.tillwat 
                                     * 10**( Pe0 * (1-state.tillwat ) / PCc ) 
        )

        PW  = P0 - tilleffpres # as tilleffpres = P0 - PW
        
        phi = ( 1000 * 9.81 * state.topg + PW ) / ( 1000 * 9.81 )

        state.phi = tf.Variable(phi, dtype=tf.float32)
 
        transform = from_origin(state.x[0], state.y[-1], state.dx, state.dx)

        # Export to GeoTIFF
        with rasterio.open('phi.tif', 'w', driver='GTiff', 
                            height=phi.shape[0], width=phi.shape[1],  count=1, 
                            dtype=np.float32, crs='EPSG:32632', transform=transform,
                            nodata=-9999) as dst:
            dst.write(phi, 1)

        ############################################################

        grid = Grid.from_raster('phi.tif')
        dem = grid.read_raster('phi.tif')

        os.remove('phi.tif')
 
        # Fill pits in DEM
        pit_filled_dem = grid.fill_pits(dem)

        # Fill depressions in DEM
        flooded_dem = grid.fill_depressions(pit_filled_dem)
            
        # Resolve flats in DEM
        inflated_dem = grid.resolve_flats(flooded_dem)
 
        # Specify directional mapping
        dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
            
        # Compute flow directions
        fdir = grid.flowdir(inflated_dem, dirmap=dirmap)

        # Calculate flow accumulation
        state.flow_accumulation = tf.Variable(grid.accumulation(fdir, dirmap=dirmap), dtype=tf.float32)

        state.tlast_flow_acc = state.t.numpy()

def finalize(cfg, state):
    pass
