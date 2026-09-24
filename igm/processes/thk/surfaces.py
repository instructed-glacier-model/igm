#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Ice-surface reconstruction from thickness and bed topography."""

import math

import tensorflow as tf


def get_density_ratio(cfg) -> float:
    """Return the canonical ice/water density ratio used for flotation.

    ``thk.ratio_density`` determines the floating surface, while iceflow uses
    its ice and water densities for grounding and ocean-front stresses.  A
    mismatch therefore describes two different grounding lines in one model.
    When iceflow densities are configured, their exact ratio is authoritative;
    the thickness option remains a consistency check and the fallback for
    thickness-only configurations.
    """
    configured = float(cfg.processes.thk.ratio_density)
    if not math.isfinite(configured) or configured <= 0.0:
        raise ValueError("cfg.processes.thk.ratio_density must be positive.")

    processes = getattr(cfg, "processes", None)
    iceflow = None if processes is None else getattr(processes, "iceflow", None)
    physics = None if iceflow is None else getattr(iceflow, "physics", None)
    if physics is None:
        return configured

    ice_density = getattr(physics, "ice_density", None)
    water_density = getattr(physics, "water_density", None)
    if ice_density is None or water_density is None:
        return configured

    ice_density = float(ice_density)
    water_density = float(water_density)
    if not math.isfinite(ice_density) or ice_density <= 0.0:
        raise ValueError("cfg.processes.iceflow.physics.ice_density must be positive.")
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
    return physical


def validate_density_ratio(cfg):
    """Validate the configured flotation densities before TensorFlow tracing."""
    get_density_ratio(cfg)


def update_surfaces(cfg, state):
    """Lower / upper ice surfaces from flotation against ``state.water_level``.

    With the "no ocean" water level (see ``masks.py``) the flotation
    base lies far below any bed, so ``lsurf == topg`` exactly.
    """
    ratio_density = get_density_ratio(cfg)
    state.lsurf = tf.maximum(
        state.topg,
        -ratio_density * state.thk + state.water_level,
    )
    state.usurf = state.lsurf + state.thk
