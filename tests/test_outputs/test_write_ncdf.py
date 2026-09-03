#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Tests for NetCDF output file lifetime and appends."""

from types import SimpleNamespace

from netCDF4 import Dataset
import numpy as np
from omegaconf import OmegaConf
import tensorflow as tf

from igm.outputs import write_ncdf


def _initialize_output(tmp_path):
    output_file = tmp_path / "output.nc"
    cfg = OmegaConf.create(
        {
            "processes": {},
            "outputs": {
                "write_ncdf": {
                    "output_file": str(output_file),
                    "keep_open": True,
                    "vars_to_save": ["thk"],
                }
            },
        }
    )
    state = SimpleNamespace(
        x=tf.constant([0.0, 1.0, 2.0]),
        y=tf.constant([0.0, 1.0]),
        t=tf.Variable(0.0),
        thk=tf.Variable(tf.ones((2, 3))),
        saveresult=True,
        continue_run=True,
    )

    write_ncdf.initialize(cfg, state)
    write_ncdf.run(cfg, state)
    return cfg, state, output_file


def test_keep_open_reuses_handle_and_closes_after_final_save(tmp_path):
    cfg, state, output_file = _initialize_output(tmp_path)

    handle = state._write_ncdf_handle
    assert handle is not None
    assert handle.isopen()

    state.t.assign(1.0)
    state.thk.assign(tf.fill((2, 3), 2.0))
    write_ncdf.run(cfg, state)

    assert state._write_ncdf_handle is handle
    assert len(handle.dimensions["time"]) == 2

    state.t.assign(2.0)
    state.thk.assign(tf.fill((2, 3), 3.0))
    state.continue_run = False
    write_ncdf.run(cfg, state)

    assert state._write_ncdf_handle is None
    assert not handle.isopen()

    with Dataset(output_file) as nc:
        np.testing.assert_allclose(nc.variables["time"][:], [0.0, 1.0, 2.0])
        np.testing.assert_allclose(
            nc.variables["thk"][:, 0, 0], [1.0, 2.0, 3.0]
        )


def test_keep_open_closes_when_run_stops_between_saves(tmp_path):
    cfg, state, _ = _initialize_output(tmp_path)
    handle = state._write_ncdf_handle

    state.saveresult = False
    state.continue_run = False
    write_ncdf.run(cfg, state)

    assert state._write_ncdf_handle is None
    assert not handle.isopen()
