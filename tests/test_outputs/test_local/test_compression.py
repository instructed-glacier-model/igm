#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""NetCDF compression options: lossless codecs and lossy quantization."""

from netCDF4 import Dataset
import numpy as np
import pytest
import xarray as xr

from .helpers import initialize_output, save


def _filter_available(codec):
    if codec == "zlib":  # built into every netCDF-C
        return True
    with Dataset("probe.nc", "w", diskless=True) as nc:
        return getattr(nc, f"has_{codec}_filter")()


@pytest.mark.parametrize("codec", ["zlib", "zstd", "bzip2"])
def test_lossless_compression_keeps_every_value(tmp_path, codec):
    if not _filter_available(codec):
        pytest.skip(f"netCDF-C has no {codec} filter")
    cfg, state = initialize_output(
        tmp_path, complevel=4, compression=codec, write_ts=False
    )
    rng = np.random.default_rng(0)
    saved = [rng.random((2, 3), dtype=np.float32) for _ in range(3)]
    for k, thk in enumerate(saved):
        save(cfg, state, float(k), thk=thk, last=k == 2)

    with Dataset(tmp_path / "output.nc") as nc:
        filters = nc.variables["thk"].filters()
        assert filters[codec]
        assert filters["complevel"] == 4
    with xr.open_dataset(tmp_path / "output.nc") as ds:
        np.testing.assert_array_equal(ds.thk.values, np.stack(saved))


def test_significant_digits_quantizes_appended_saves_too(tmp_path):
    cfg, state = initialize_output(
        tmp_path, complevel=1, significant_digits=3, write_ts=False
    )
    rng = np.random.default_rng(0)
    saved = [rng.uniform(0, 500, (2, 3)).astype(np.float32) for _ in range(3)]
    for k, thk in enumerate(saved):
        save(cfg, state, float(k), thk=thk, last=k == 2)

    with xr.open_dataset(tmp_path / "output.nc") as ds:
        thk = ds.thk.values
    np.testing.assert_allclose(thk, np.stack(saved), rtol=1e-3)
    # every save is quantized, not only the first one written by xarray
    assert all(not np.array_equal(thk[k], saved[k]) for k in range(3))


def test_unknown_codec_is_rejected_at_initialize(tmp_path):
    with pytest.raises(ValueError, match="compression"):
        initialize_output(tmp_path, complevel=1, compression="gzip")
