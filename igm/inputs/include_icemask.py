#!/usr/bin/env python3

# Author: Andreas Henz, andreas.henz@geo.uzh.ch
# Date: 06.09.2023

"""
This IGM module loads an icemask shapefile (ESRI Shapefile) and creates a ice mask out of it.
The icemask can be either the coordinates where there should be glacier (default)
or where there should be glacier (change invert = True). 

Input: shapefile (.shp) exported from any GIS (example: QGIS)
Output: state.icemask

This module can be used together with any model that in the end calculates the new glacier surface via the state.smb variable.
    like in smb_simple.py:
        # if an icemask exists, then force negative smb aside to prevent leaks
        if hasattr(state, "icemask"):
            state.smb = tf.where((state.smb<0)|(state.icemask>0.5),state.smb,-10)

Input can be one or multiple polygon features, it does not matter. Sometimes it easier to select the valley where the glacier should be (invert = True)
or you draw polygons where the glacier should not be (e.g. side valleys with no further interest).

IMPORTANT: Pay attention to the coordinate system used in the nc file and the shapefile
"""

import numpy as np
import tensorflow as tf

import geopandas as gpd
from shapely.geometry import Point

def include_icemask(state, mask_shapefile, mask_invert):

    # read_shapefile
    gdf = read_shapefile(mask_shapefile)

    # Flatten the X and Y coordinates and convert to numpy
    flat_X = state.X.numpy().flatten()
    flat_Y = state.Y.numpy().flatten()

    # Create a list to store the mask values
    mask_values = []

    # Iterate over each grid point
    for x, y in zip(flat_X, flat_Y):
        point = Point(x, y)
        inside_polygon = False

        # Check if the point is inside any polygon in the GeoDataFrame
        for geom in gdf.geometry:
            if point.within(geom):
                inside_polygon = True
                break  # if it is inside one polygon, don't check for others

        # Append the corresponding mask value to the list
        mask_values.append(0 if inside_polygon else 1)

    # reshape
    mask_values = np.array(mask_values, dtype=np.float32)
    mask_values = mask_values.reshape(state.X.shape)

    # Invert the mask values if mask_invert is True
    if mask_invert:
        mask_values = np.logical_not(mask_values).astype(np.float32)

    # define icemask
    state.icemask = tf.constant(mask_values)



def read_shapefile(filepath):
    try:
        # Read the shapefile
        gdf = gpd.read_file(filepath)

        # Print the information about the shapefile
        print("-----------------------")
        print("Icemask Shapefile information:")
        print("Number of features (polygons):", len(gdf))
        print("EPSG code: ", gdf.crs.to_epsg())
        print("Geometry type:", gdf.geometry.type.unique()[0])
        print("-----------------------")

        # Return the GeoDataFrame
        return gdf
    except Exception as e:
        print("Error reading shapefile:", e)
