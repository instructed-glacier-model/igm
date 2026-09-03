import igm
import tensorflow as tf
import pytest
import numpy as np
from .make_fake_ncdf import write_ncdf
import os

from igm.common import State
from igm.common.runner.configuration.loader import load_yaml_recursive


def test_load_ncdf():

    write_ncdf()

    state = State()

    state.original_cwd = ""

    cfg = load_yaml_recursive(os.path.join(igm.__path__[0], "conf"))

    cfg.core.folder_data = ""
    cfg.inputs.load_ncdf.input_file = "input.nc"

    igm.inputs.load_ncdf.run(cfg, state)

    ny, nx = state.thk.shape

    mid = state.topg[int(ny / 2), int(nx / 2)]

    assert (mid > 2450) & (mid < 2550)

    for f in ["input.nc"]:
        if os.path.exists(f):
            os.remove(f)


def test_load_ncdf_selects_configured_restart_time(tmp_path):
    path = tmp_path / "restart.nc"
    write_ncdf(path, times=[0.0, 20.0])

    state = State()
    state.original_cwd = str(tmp_path)

    cfg = load_yaml_recursive(os.path.join(igm.__path__[0], "conf"))
    cfg.core.folder_data = ""
    cfg.inputs.load_ncdf.input_file = path.name
    cfg.inputs.load_ncdf.time = 20.0

    igm.inputs.load_ncdf.run(cfg, state)

    ny, nx = state.topg.shape
    mid = state.topg[int(ny / 2), int(nx / 2)]
    assert (mid > 2550) & (mid < 2650)
