#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Shared helpers for thk."""

import tensorflow as tf


def update_surfaces(cfg, state):
    """Lower / upper ice surfaces. Flotation when state.water_level exists,
    else lsurf = topg."""
    p = cfg.processes.thk
    if hasattr(state, "water_level"):
        state.lsurf = tf.maximum(
            state.topg,
            -p.ratio_density * state.thk + state.water_level,
        )
    else:
        state.lsurf = tf.identity(state.topg)
    state.usurf = state.lsurf + state.thk
