#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Ice-surface reconstruction from thickness and bed topography."""

import math

import tensorflow as tf


def validate_density_ratio(cfg):
    """Require thickness flotation and iceflow physics to use one ratio.

    ``thk.ratio_density`` determines the floating surface, while iceflow uses
    its ice and water densities for grounding and ocean-front stresses.  A
    mismatch therefore describes two different grounding lines in one model.
    Validation is deliberately host-side and runs once during initialization.
    """
    configured = float(cfg.processes.thk.ratio_density)
    if not math.isfinite(configured) or configured <= 0.0:
        raise ValueError("cfg.processes.thk.ratio_density must be positive.")

    processes = getattr(cfg, "processes", None)
    iceflow = None if processes is None else getattr(processes, "iceflow", None)
    physics = None if iceflow is None else getattr(iceflow, "physics", None)
    if physics is None:
        return

    ice_density = getattr(physics, "ice_density", None)
    water_density = getattr(physics, "water_density", None)
    if ice_density is None or water_density is None:
        return

    ice_density = float(ice_density)
    water_density = float(water_density)
    if not math.isfinite(ice_density) or ice_density <= 0.0:
        raise ValueError(
            "cfg.processes.iceflow.physics.ice_density must be positive."
        )
    if not math.isfinite(water_density) or water_density <= 0.0:
        raise ValueError(
            "cfg.processes.iceflow.physics.water_density must be positive."
        )

    physical = ice_density / water_density
    if not math.isclose(configured, physical, rel_tol=5.0e-4, abs_tol=5.0e-6):
        raise ValueError(
            "Inconsistent flotation densities: "
            f"cfg.processes.thk.ratio_density={configured:.12g}, but "
            "cfg.processes.iceflow.physics.ice_density / water_density="
            f"{physical:.12g} ({ice_density:.12g}/{water_density:.12g}). "
            "Configure the same physical density ratio in both modules."
        )


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
