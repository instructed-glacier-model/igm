#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Shared setup for the tests of the local output module."""

from types import SimpleNamespace

from omegaconf import OmegaConf
import tensorflow as tf

from igm.outputs import local


def initialize_output(tmp_path, **local_cfg):
    cfg = OmegaConf.create(
        {
            "processes": {"iceflow": {"numerics": {"Nz": 2}}},
            "outputs": {
                "local": {
                    "file_format_list": ["netcdf"],
                    "output_file": str(tmp_path / "output.nc"),
                    "keep_open": False,
                    "complevel": 0,
                    "compression": "zlib",
                    "significant_digits": None,
                    "vars_to_save": ["thk", "T"],
                    "write_ts": True,
                    "output_ts_file": str(tmp_path / "output_ts.nc"),
                    **local_cfg,
                }
            },
        }
    )
    state = SimpleNamespace(
        x=tf.constant([0.0, 100.0, 200.0]),
        y=tf.constant([0.0, 100.0]),
        dx=100.0,
        t=tf.Variable(0.0),
        thk=tf.Variable(tf.ones((2, 3))),
        # 5 layers, unlike the 2 of iceflow: written on its own vertical dimension
        T=tf.Variable(tf.fill((5, 2, 3), 260.0)),
        saveresult=True,
        continue_run=True,
    )
    local.initialize(cfg, state)
    return cfg, state


def save(cfg, state, t, thk=None, last=False):
    """Run one save at time `t`, with `thk` (default: filled with t + 1)."""

    state.t.assign(t)
    state.thk.assign(tf.fill((2, 3), t + 1.0) if thk is None else thk)
    state.continue_run = not last
    local.run(cfg, state)
