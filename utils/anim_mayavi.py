#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
import matplotlib.pyplot as plt

from mayavi import mlab
import xarray as xr
import numpy as np
import os

plt.close("all")

ds = xr.open_dataset("output.nc", engine="netcdf4")

X, Y = np.meshgrid(ds.x, ds.y)

TIME = np.array(ds.time)

vmin = np.min(ds.velsurf_mag)
vmax = np.max(ds.velsurf_mag)

# mlab.figure(bgcolor=(0.16, 0.28, 0.46))

XX = np.where(ds.thk[0] == 0, np.nan, X)
YY = np.where(ds.thk[0] == 0, np.nan, Y)
ZZ = np.where(ds.thk[0] == 0, np.nan, ds.usurf[0])
CC = np.array(ds.velsurf_mag[0])

base = mlab.mesh(X, Y, ds.topg[0], colormap="terrain", opacity=0.75)
surf = mlab.mesh(
    XX, YY, ZZ, scalars=CC, colormap="jet", vmin=vmin, vmax=vmax, opacity=0.75
)
mlab.colorbar(surf, orientation="vertical", title="Ice speed (m/y)")
mlab.title(str(TIME[0]) + " y", size=0.5)

VEC = hasattr(ds, "uvelsurf") & hasattr(ds, "vvelsurf") & hasattr(ds, "wvelsurf")

PAR = os.path.isdir("trajectories")

if VEC:
    quiv = mlab.quiver3d(
        tf.expand_dims(X, axis=0),
        tf.expand_dims(Y, axis=0),
        tf.expand_dims(ds.usurf[0], axis=0),
        tf.expand_dims(ds.uvelsurf[0], axis=0),
        tf.expand_dims(ds.vvelsurf[0], axis=0),
        tf.expand_dims(ds.wvelsurf[0], axis=0),
    )

# if PAR:
#     f = os.path.join("trajectories",'traj-'+"{:06.0f}".format(TIME[0])+".csv")
#     XYZ = np.loadtxt(f,skiprows=1,delimiter=',')
#     pt3d = mlab.points3d(XYZ[:,1], XYZ[:,2], XYZ[:,3], colormap="RdBu",mode='point')
#     pt3d.actor.property.point_size = 5
#     pt3d.mlab_source.dataset.point_data.scalars = XYZ[:,4]

@mlab.animate(ui=True)
def anim(VEC, PAR):
    for i in range(0, ds.thk.shape[0]):
        surf.mlab_source.z = np.where(ds.thk[i] == 0, np.nan, ds.usurf[i])
        surf.mlab_source.scalars = np.array(ds.velsurf_mag[i])
        if VEC:
            quiv.mlab_source.z = np.array(tf.expand_dims(ds.usurf[i], axis=0))
            quiv.mlab_source.u = np.array(tf.expand_dims(ds.uvelsurf[i], axis=0))
            quiv.mlab_source.v = np.array(tf.expand_dims(ds.vvelsurf[i], axis=0))
            quiv.mlab_source.w = np.array(tf.expand_dims(ds.wvelsurf[i], axis=0))
        # if PAR:
        #     f = os.path.join("trajectories",'traj-'+"{:06.0f}".format(TIME[i])+".csv")
        #     XYZ = np.loadtxt(f,skiprows=1,delimiter=',')
        #     pt3d.mlab_source.x = XYZ[:,1]
        #     pt3d.mlab_source.y = XYZ[:,2]
        #     pt3d.mlab_source.z = XYZ[:,3]
        #     pt3d.mlab_source.dataset.point_data.scalars = XYZ[:,4]

        mlab.title("Time " + str(int(TIME[i])), size=0.5)
        yield

anim(VEC, PAR)
mlab.show()
