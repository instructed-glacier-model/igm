#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Configuration and a MISMIP+-like channel for the bmb tests."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest
import tensorflow as tf

import igm
from igm.processes.ocean import ocean
from igm.processes.thk.surfaces import update_surfaces

RHO_I, RHO_W = 918.0, 1028.0


def make_cfg(**bmb):
    """Default thk, ocean and bmb configurations, with the given bmb overrides.

    The channel flows along +x from an ice divide (symmetric left side)
    between two walls (symmetric top and bottom) to an open right side.
    """
    conf = Path(igm.__file__).parent / "conf" / "processes"
    cfg = OmegaConf.create(
        {
            "processes": {
                name: OmegaConf.load(conf / f"{name}.yaml")[name]
                for name in ("thk", "ocean", "bmb")
            }
        }
    )
    cfg.processes.iceflow = {"physics": {"ice_density": RHO_I, "water_density": RHO_W}}
    cfg.processes.thk.ratio_density = RHO_I / RHO_W
    cfg.processes.thk.boundary = {
        "left": "symmetric",
        "right": "zero",
        "top": "symmetric",
        "bottom": "symmetric",
    }
    cfg.processes.bmb.merge_with(bmb)
    return cfg


def make_state(cfg, thk, topg, water_level=0.0, ubar=0.0, **fields):
    """State with surfaces from flotation and ocean fields from ``cfg``."""
    thk = np.asarray(thk, np.float32)
    as_field = lambda v: tf.constant(np.broadcast_to(np.float32(v), thk.shape).copy())
    state = SimpleNamespace(
        thk=tf.constant(thk),
        topg=as_field(topg),
        water_level=as_field(water_level),
        ubar=as_field(ubar),
        vbar=as_field(0.0),
        dx=tf.constant(1000.0),
        t=tf.Variable(0.0),
        it=0,
        **{name: as_field(value) for name, value in fields.items()},
    )
    update_surfaces(cfg, state)
    ocean.initialize(cfg, state)
    ocean.update(cfg, state)
    return state


def channel(ny=12, nx=40, front=None):
    """Thickness and bed of a grounded ice stream feeding a shelf.

    The bed deepens along x and the ice thins, so the grounding line lies
    near x = 14 km. Without ``front`` the shelf ends on the open right side;
    otherwise it ends at column ``front``, followed by ice-free ocean.
    """
    x = np.arange(nx, dtype=np.float32)
    topg = np.broadcast_to(300.0 - 50.0 * x, (ny, nx)).clip(-720.0).astype(np.float32)
    thk = np.broadcast_to(np.linspace(1000.0, 300.0, nx), (ny, nx)).astype(np.float32)
    thk = thk.copy()
    if front is not None:
        thk[:, front:] = 0.0
    return thk, topg


@pytest.fixture
def cfg_factory():
    return make_cfg


@pytest.fixture
def state_factory():
    return make_state


@pytest.fixture
def channel_factory():
    return channel
