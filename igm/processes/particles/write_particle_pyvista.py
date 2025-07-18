#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import os
import shutil
import numpy as np
import tensorflow as tf

def initialize_write_particle_pyvista(cfg, state):

    directory = "trajectories"
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.mkdir(directory)

def update_write_particle_pyvista(cfg, state):

    import pyvista as pv

    directory = "trajectories"

    if state.saveresult:
        filename = os.path.join(
            directory,
            "traj-" + "{:06d}".format(int(state.t.numpy())) + ".vtp",
        )

        # Compute positions
        x = state.particle["x"].numpy() + state.x[0].numpy()
        y = state.particle["y"].numpy() + state.y[0].numpy()
        z = state.particle["z"].numpy()

        # Create point coordinates array
        points = np.vstack((x, y, z)).T  # shape (n, 3)

        # Create point data
        data = {
            "Id": np.arange(x.shape[0], dtype=np.float32),
            "rh": state.particle["r"].numpy(),
            "t": state.particle["t"].numpy(),
            "topg": state.particle["topg"].numpy(),
            "thk": state.particle["thk"].numpy(),
        }

        if "weights" in cfg.processes.particles.fields:
            data["weights"] = state.particle["w"].numpy()

        if "englt" in cfg.processes.particles.fields:
            data["englt"] = state.particle["englt"].numpy()

        if "velmag" in cfg.processes.particles.fields:
            data["velmag"] = state.particle["velmag"].numpy()

        # Create the PyVista point cloud
        cloud = pv.PolyData(points)

        # Add attributes
        for key, array in data.items():
            cloud[key] = array

        # Write to VTP file
        cloud.save(filename)