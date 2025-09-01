#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import os
import matplotlib.pyplot as plt

import xarray as xr
from matplotlib import animation

ds = xr.open_dataset("output.nc", engine="netcdf4")

tas = ds.thk

# Get a handle on the figure and the axes
fig, ax = plt.subplots(figsize=(7, 7))

# Plot the initial frame.
cax = (
    tas[0, :, :]
    .where(tas[0, :, :] > 0)
    .plot(
        add_colorbar=True,
        cmap="jet",
        vmin=0,
        vmax=np.max(tas),
        cbar_kwargs={"extend": "neither"},
    )
)

cax.axes.set_aspect("equal")

ax.axis("off")

# Next we need to create a function that updates the values for the colormesh, as well as the title.
def animate(frame):
    cax.set_array(tas[frame, :, :].where(tas[frame, :, :] > 0).values.flatten())
    ax.set_title("Time = " + str(tas.coords["time"].values[frame])[:13])

# Finally, we use the animation module to create the animation.
ani = animation.FuncAnimation(
    fig,  # figure
    animate,  # name of the function above
    frames=tas.shape[0],  # Could also be iterable or list
    interval=500,  # ms between frames
)

ani.save("animation.mp4")

os.system("echo rm " + "animation.mp4" + " >> clean.sh")
