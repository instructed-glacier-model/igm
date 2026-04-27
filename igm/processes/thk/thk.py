#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""
thk  (user module, shadows built-in IGM thk)
============================================

Three modes selected by cfg.processes.thk:

  calving_front: false
      Stock mass-conservation thk update (same shape as igm.processes.thk).

  calving_front: true, method: level_set    (Bondzio 2016)
      Delegates to cf_level_set.update.

  calving_front: true, method: sub_grid     (Albrecht 2011 / PISM)
      Delegates to cf_sub_grid.update.

Sibling files in this folder:
  utils.py         shared helpers (neighbour ops, surface update, stock advect)
  cf_level_set.py  TO BE DONE
  cf_sub_grid.py   Albrecht 2011 / PISM sub-grid calving front parameterization.
"""

import os
import sys

import tensorflow as tf

from igm.utils.grad.compute_divflux_slope_limiter import compute_divflux_slope_limiter

# IGM loads this file under the bare name "thk" via SourceFileLoader, so the
# folder it lives in is not on sys.path. Add it so the sibling modules are
# importable.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# import cf_level_set
import cf_sub_grid
from utils import update_surfaces


def initialize(cfg, state):

    if not hasattr(state, "topg"):
        raise ValueError(
            "The 'thk' module requires an initial topography ('state.topg')."
        )

    p = cfg.processes.thk

    if p.calving_front and p.method == "level_set":
#        cf_level_set.initialize(cfg, state)
        raise ValueError(
                f"cfg.processes.thk.method 'level_set' is not currently implemented in IGM. "
                f"Please use 'sub_grid' instead."
         )
    elif p.calving_front and p.method == "sub_grid":
        cf_sub_grid.initialize(cfg, state)
    elif p.calving_front:
        raise ValueError(
            f"cfg.processes.thk.method must be 'level_set' "
            f"got: {p.method!r}"
        )

    update_surfaces(cfg, state)


def update(cfg, state):

    if state.it >= 0:
        if hasattr(state, "logger"):
            state.logger.info(
                "Ice thickness equation at time : " + str(state.t.numpy())
            )

        p = cfg.processes.thk

        if not p.calving_front:
            # Stock IGM thk update, inlined to match igm.processes.thk shape.
            state.divflux = compute_divflux_slope_limiter(
                state.ubar, state.vbar, state.thk,
                state.dx, state.dx, state.dt,
                slope_type=p.slope_type,
            )
            if not hasattr(state, "smb"):
                state.smb = tf.zeros_like(state.thk)
            state.thk = tf.maximum(
                state.thk + state.dt * (state.smb - state.divflux), 0
            )

        elif p.method == "level_set":
#            cf_level_set.update(cfg, state)
            raise ValueError(
                    f"cfg.processes.thk.method 'level_set' is not currently implemented in IGM. "
                    f"Please use 'sub_grid' instead."
                )

        elif p.method == "sub_grid":
            cf_sub_grid.update(cfg, state)

        else:
            raise ValueError(
                f"cfg.processes.thk.method must be 'level_set' "
                f"got: {p.method!r}"
            )

        update_surfaces(cfg, state)


def finalize(cfg, state):
    pass
