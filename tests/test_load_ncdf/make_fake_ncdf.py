#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset


def write_ncdf(path="input.nc", times=None):

    x = np.arange(0, 100) * 100  # make x-axis, lenght 10 km, resolution 100 m
    y = np.arange(0, 200) * 100  # make y-axis, lenght 20 km, resolution 100 m

    X, Y = np.meshgrid(x, y)

    topg = 1000 + 0.15 * Y + ((X - 5000) ** 2) / 50000  # define the bedrock topography

    print("write_ncdf ----------------------------------")

    nc = Dataset(path, "w", format="NETCDF4")

    nc.createDimension("y", len(y))
    yn = nc.createVariable("y", np.dtype("float64").char, ("y",))
    yn.units = "m"
    yn.long_name = "y"
    yn.standard_name = "y"
    yn.axis = "Y"
    yn[:] = y

    nc.createDimension("x", len(x))
    xn = nc.createVariable("x", np.dtype("float64").char, ("x",))
    xn.units = "m"
    xn.long_name = "x"
    xn.standard_name = "x"
    xn.axis = "X"
    xn[:] = x

    if times is None:
        dimensions = ("y", "x")
        values = topg
    else:
        nc.createDimension("time", len(times))
        time = nc.createVariable("time", np.dtype("float32").char, ("time",))
        time[:] = times
        dimensions = ("time", "y", "x")
        values = np.stack([topg + index * 100.0 for index in range(len(times))])

    E = nc.createVariable("topg", np.dtype("float32").char, dimensions)
    E.long_name = "basal topography"
    E.units = "m"
    E.standard_name = "topg"
    E[:] = values

    nc.close()
