#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Saves are appended in place, keep their layout, and survive a reader."""

import os
import subprocess
import sys

import numpy as np
import pytest
import tensorflow as tf
import xarray as xr

from igm.outputs import local

from .helpers import initialize_output, save


@pytest.mark.parametrize("keep_open", [False, True])
def test_saves_are_appended_in_place(tmp_path, keep_open):
    cfg, state = initialize_output(tmp_path, keep_open=keep_open)

    for t in (0.0, 1.0, 2.0):
        save(cfg, state, t, last=t == 2.0)

    assert all(f["handle"] is None for f in state.local_netcdf_files.values())

    with xr.open_dataset(tmp_path / "output.nc") as ds:
        assert ds.encoding["unlimited_dims"] == {"time"}
        np.testing.assert_allclose(ds.time, [0.0, 1.0, 2.0])
        np.testing.assert_allclose(ds.thk[:, 0, 0], [1.0, 2.0, 3.0])
        assert ds.thk.attrs == {"long_name": "Ice Thickness", "units": "m"}
        assert ds.T.dims == ("time", "z5", "y", "x")
        assert ds.sizes["z"] == 2
        assert ds.x.attrs["units"] == "m" and ds.time.attrs["units"] == "yr"

    with xr.open_dataset(tmp_path / "output_ts.nc") as ts:
        np.testing.assert_allclose(ts.vol, [6e-5, 12e-5, 18e-5], rtol=1e-6)
        np.testing.assert_allclose(ts.area, [0.0, 0.06, 0.06], rtol=1e-6)
        assert ts.vol.attrs["units"] == "km^3"


def test_append_matches_a_single_xarray_write(tmp_path):
    # Appended saves must hold exactly what xarray writes from the whole series at
    # once, including encodings such as packing, NaN as missing value, compression.
    cfg, state = initialize_output(tmp_path, complevel=1, write_ts=False)
    snapshots = []
    for t in (0.0, 1.0, 2.0):
        state.t.assign(t)
        state.thk.assign(tf.fill((2, 3), t + 1.0))
        state.thk[0, 0].assign(np.nan)
        snapshots.append(local.dataset_ex(cfg, state))
        local.write_netcdf(cfg, state, snapshots[-1], str(tmp_path / "output.nc"))
    local.close_files(state)

    expected = xr.concat(snapshots, dim="time")
    with xr.open_dataset(tmp_path / "output.nc") as ds:
        xr.testing.assert_identical(ds, expected)
        assert np.isnan(ds.thk[:, 0, 0]).all()
        assert ds.thk.encoding["zlib"] and ds.thk.encoding["chunksizes"] == (1, 2, 3)


def test_save_while_another_program_holds_the_file_open(tmp_path):
    cfg, state = initialize_output(tmp_path, write_ts=False)
    save(cfg, state, 0.0)

    # HDF5 refuses to open a file for writing while another process reads it
    reader = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys; from netCDF4 import Dataset\n"
            f"nc = Dataset({str(tmp_path / 'output.nc')!r})\n"
            "print('open', flush=True); sys.stdin.read()",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert reader.stdout.readline().strip() == "open"
        inode = os.stat(tmp_path / "output.nc").st_ino
        save(cfg, state, 1.0)
        # appended to a copy put in place of the file the reader holds
        assert os.stat(tmp_path / "output.nc").st_ino != inode
    finally:
        reader.communicate("")

    inode = os.stat(tmp_path / "output.nc").st_ino
    save(cfg, state, 2.0, last=True)
    assert os.stat(tmp_path / "output.nc").st_ino == inode  # back to appending in place

    with xr.open_dataset(tmp_path / "output.nc") as ds:
        np.testing.assert_allclose(ds.time, [0.0, 1.0, 2.0])
        np.testing.assert_allclose(ds.thk[:, 0, 0], [1.0, 2.0, 3.0])
