#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""The optional Zarr store holds the same saves as the NetCDF file."""

import pytest
import xarray as xr

from .helpers import initialize_output, save


def test_zarr_store_matches_netcdf_file(tmp_path):
    pytest.importorskip("zarr")
    cfg, state = initialize_output(
        tmp_path, file_format_list=["netcdf", "zarr"], write_ts=False
    )

    for t in (0.0, 1.0, 2.0):
        save(cfg, state, t, last=t == 2.0)

    assert (tmp_path / "output.zarr" / ".zmetadata").exists()
    with xr.open_dataset(tmp_path / "output.nc") as nc, xr.open_zarr(
        tmp_path / "output.zarr"
    ) as zarr_ds:
        xr.testing.assert_identical(nc.load(), zarr_ds.load())
