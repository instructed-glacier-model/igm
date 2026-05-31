#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from typing import Any, Dict
from omegaconf import DictConfig
import tensorflow as tf

from ..energy import EnergyComponent


class SlidingComponent(EnergyComponent):
    """Energy component representing frictional energy."""

    pass


def mask_gr(h: tf.Tensor, topg: tf.Tensor, rho_ratio: tf.Tensor) -> tf.Tensor:
    """Compute grounding mask: 1 where grounded, 0 where floating."""
    phi = h + rho_ratio * topg
    return tf.cast(phi > 0.0, dtype=h.dtype)


def get_friction_field(fieldin: Dict[str, tf.Tensor]) -> tf.Tensor:
    """Return the reference basal shear stress field from a fieldin dict.

    Accepts either the canonical name `tau_ref` (used by the new
    `unified` stack) or the legacy `slidingco` (kept by the
    emulated/solved/diagnostic stack because the pretrained CNN
    emulators' channel order metadata is frozen on that name).

    Dual-name handling lives ONLY here. Callers (the four sliding-law
    kernels) read this helper without caring about which channel name
    is active in the user's yaml.
    """
    if "tau_ref" in fieldin:
        return fieldin["tau_ref"]
    if "slidingco" in fieldin:
        return fieldin["slidingco"]
    raise KeyError(
        "get_friction_field: fieldin has neither 'tau_ref' nor "
        "'slidingco'; the sliding law cannot be evaluated without a "
        "reference basal shear stress channel."
    )


_LAW_KEYS: Dict[str, set] = {
    "weertman":     {"regularization", "exponent", "u_ref"},
    "coulomb":      {"regularization", "exponent", "u_ref", "mu"},
    "budd":         {"regularization", "exponent", "u_ref", "N_ref", "q_exponent"},
    "mohr_coulomb": {"regularization", "exponent", "u_ref",
                     "phi", "phi_min", "phi_max", "bed_min", "bed_max",
                     "tauc_min", "tauc_max", "tauc_ice_free"},
}


def get_sliding_params_args(cfg: DictConfig) -> Dict[str, Any]:
    """Extract friction parameters from configuration.

    Sliding params live flat under `cfg.processes.iceflow.physics.sliding`;
    each law picks the keys it needs from `_LAW_KEYS`.
    """
    cfg_physics = cfg.processes.iceflow.physics
    law = cfg_physics.sliding.law
    if law not in _LAW_KEYS:
        raise ValueError(
            f"❌ Unknown sliding law '{law}'. "
            f"Supported: {sorted(_LAW_KEYS)}."
        )

    args = {k: cfg_physics.sliding[k] for k in _LAW_KEYS[law]}
    args["rho_ratio"] = cfg_physics.water_density / cfg_physics.ice_density
    args["use_mask_gr"] = cfg_physics.sliding.use_mask_gr

    return args
