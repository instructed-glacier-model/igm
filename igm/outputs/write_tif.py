#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import os
import tensorflow as tf


def initialize(cfg, state):
    pass


def run(cfg, state):
    import rasterio

    if state.saveresult:
        for var in cfg.outputs.write_tif.vars_to_save:
            file = var + "-" + str(int(state.t)).zfill(6) + ".tif"

            if hasattr(state, "profile_tif_file"):
                with rasterio.open(file, mode="w", **state.profile_tif_file) as src:
                    src.write(np.flipud(vars(state)[var]), 1)
            else:
                xres = state.dx
                yres = state.dx 
                transform = rasterio.Affine.translation(
                    state.x[0] - xres / 2, state.y[0] - yres / 2
                ) * rasterio.Affine.scale(xres, yres)

                with rasterio.open(
                    file,
                    mode="w",
                    driver="GTiff",
                    height=vars(state)[var].shape[0],
                    width=vars(state)[var].shape[1],
                    count=1,
                    dtype=np.float32,
                    transform=transform,
                ) as src:
                    src.write(vars(state)[var], 1)

            del src
