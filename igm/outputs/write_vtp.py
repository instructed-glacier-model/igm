#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import os,shutil
import numpy as np 
import tensorflow as tf 

from igm.utils.math.getmag import getmag

def initialize(cfg, state):
 
    import pyvista as pv

    directory = "vtp"
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.mkdir(directory)

    ftt = os.path.join(directory, "topg.vtp")

    # Apply mask
    mask = state.X > 0
    x = state.X[mask].numpy()
    y = state.Y[mask].numpy()
    z = state.topg[mask].numpy()

    # Create point cloud
    points = np.vstack((x, y, z)).T  # shape (n, 3)
    cloud = pv.PolyData(points)

    # Optionally add 'topg' as scalar (z)
    cloud["topg"] = z

    surface = cloud.delaunay_2d()

    # Save to VTP format
    surface.save(ftt)

    if "particles" in cfg.processes: 
        directory = "trajectories"
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.mkdir(directory)

def run(cfg, state):

    import pyvista as pv

    directory = "vtp"

    # Apply mask
    mask = state.X > 0
    x = state.X[mask].numpy()
    y = state.Y[mask].numpy()
    z = state.usurf[mask].numpy()

    if state.saveresult:

        if "velbar_mag" in cfg.outputs.write_vtp.vars_to_save:
            state.velbar_mag = getmag(state.ubar, state.vbar)

        if "velsurf_mag" in cfg.outputs.write_vtp.vars_to_save:
            state.velsurf_mag = getmag(state.uvelsurf, state.vvelsurf)

        if "velbase_mag" in cfg.outputs.write_vtp.vars_to_save:
            state.velbase_mag = getmag(state.uvelbase, state.vvelbase)

        for var in cfg.outputs.write_vtp.vars_to_save:

            if hasattr(state, var): 

                varfl = vars(state)[var][mask].numpy()

                ftt = os.path.join(
                    directory,
                    var + "-" + "{:06d}".format(int(state.t.numpy())) + ".vtp",
                )

                # Create point cloud
                points = np.vstack((x, y, z)).T  # shape (n, 3)
                cloud = pv.PolyData(points)

                # Optionally add 'topg' as scalar (z)
                cloud[var] =  varfl

                surface = cloud.delaunay_2d()

                # Save to VTP format
                surface.save(ftt)

        if "particles" in cfg.processes: 

            filename = os.path.join(
                "trajectories",
                "traj-" + "{:08.2f}".format(state.t.numpy()).replace('.', '-') + ".vtp",
            )

            # Compute positions
            x = state.particle["x"].numpy() + state.x[0].numpy()
            y = state.particle["y"].numpy() + state.y[0].numpy()
            z = state.particle["z"].numpy()

            # Create point coordinates array
            points = np.vstack((x, y, z)).T  # shape (n, 3)

            # Create point data
            data = {key: val.numpy() for key, val in state.particle.items()}

            # Create the PyVista point cloud
            cloud = pv.PolyData(points)

            # Add attributes
            for key, array in data.items():
                cloud[key] = array

            # Write to VTP file
            cloud.save(filename)
