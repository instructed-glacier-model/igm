#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Build ``ErrorEstimator`` arguments from the Hydra configuration and state."""

import copy
import types
from typing import Any, Dict, Optional

import numpy as np
import tensorflow as tf
from omegaconf import DictConfig, OmegaConf

from igm.common import State
from igm.utils.math.precision import normalize_precision

from .reference import load_reference_surface_velocity


def build_cost_fn_in_precision(cfg: DictConfig, precision: str):
    """Return ``(cost_fn, discr_v)`` evaluated in ``precision``.

    The energy is dtype-agnostic except for the horizontal/vertical
    discretization constants, which are created in ``numerics.precision``.
    When the estimator runs in another precision (typically double inside a
    single-precision network run) it needs its own discretizations and a cost
    function bound to them.
    """
    from igm.processes.iceflow.horizontal import HorizontalDiscrs
    from igm.processes.iceflow.vertical import VerticalDiscrs
    from ..utils import get_cost_fn

    cfg_est = copy.deepcopy(cfg)
    cfg_est.processes.iceflow.numerics.precision = (
        "double" if normalize_precision(precision).name == "float64" else "single"
    )
    cfg_numerics = cfg_est.processes.iceflow.numerics
    discr_v = VerticalDiscrs[cfg_numerics.basis_vertical.lower()](cfg_est)
    discr_h = HorizontalDiscrs[cfg_numerics.basis_horizontal.lower()](cfg_est)
    shadow_state = types.SimpleNamespace(
        iceflow=types.SimpleNamespace(discr_h=discr_h, discr_v=discr_v)
    )
    return get_cost_fn(cfg_est, shadow_state), discr_v


def _cfg_get(cfg: DictConfig, key: str, default: Any) -> Any:
    try:
        value = cfg[key]
    except Exception:
        return default
    return default if value is None else value


def _optional_index(names, name: str) -> Optional[int]:
    return names.index(name) if name in names else None


class InterfaceErrorEstimator:

    @staticmethod
    def enabled(cfg: DictConfig) -> bool:
        cfg_est = getattr(cfg.processes.iceflow.unified, "error_estimator", None)
        return cfg_est is not None and bool(_cfg_get(cfg_est, "enabled", False))

    @staticmethod
    def get_error_estimator_args(
        cfg: DictConfig, state: State, cost_fn, mapping
    ) -> Dict[str, Any]:
        cfg_unified = cfg.processes.iceflow.unified
        cfg_numerics = cfg.processes.iceflow.numerics
        cfg_physics = cfg.processes.iceflow.physics
        cfg_est = cfg_unified.error_estimator

        # Single-patch requirement: the estimator sees the full domain.
        ny, nx = int(state.thk.shape[0]), int(state.thk.shape[1])
        framesizemax = int(cfg_unified.data_preparation.framesizemax)
        num_patches = (ny // framesizemax + 1) * (nx // framesizemax + 1)
        if num_patches > 1:
            raise ValueError(
                "❌ error_estimator requires a single full-domain patch; "
                f"framesizemax={framesizemax} splits the {ny}x{nx} grid into "
                f"{num_patches} patches."
            )
        if bool(
            _cfg_get(getattr(cfg_unified, "adaptive_patching", {}), "enabled", False)
        ):
            raise ValueError(
                "❌ error_estimator is incompatible with adaptive_patching.enabled=true."
            )

        inputs = list(cfg_unified.inputs)
        if "thk" not in inputs:
            raise ValueError("❌ error_estimator requires 'thk' among unified.inputs.")

        precision = _cfg_get(cfg_est, "precision", "double") or cfg_numerics.precision
        Nz = int(cfg_numerics.Nz)

        discr_v = state.iceflow.discr_v
        if normalize_precision(precision) != normalize_precision(
            cfg_numerics.precision
        ):
            # Estimator precision differs from the run: rebuild the energy in
            # the estimator precision (own discretization constants).
            cost_fn, discr_v = build_cost_fn_in_precision(cfg, precision)

        topg = None
        if hasattr(state, "topg") and state.topg is not None:
            topg = np.asarray(
                state.topg.numpy() if hasattr(state.topg, "numpy") else state.topg
            )
        water_level = np.asarray(state.water_level)

        basin_mask = None
        if bool(_cfg_get(cfg_est, "use_basin_mask", True)) and hasattr(
            state, "basinmask"
        ):
            basin = state.basinmask
            basin = basin.numpy() if hasattr(basin, "numpy") else np.asarray(basin)
            basin_mask = np.asarray(basin) > 0.5

        reference = None
        reference_file = _cfg_get(cfg_est, "reference_file", None)
        if reference_file:
            reference = load_reference_surface_velocity(str(reference_file), (ny, nx))

        probe_mode = str(_cfg_get(cfg_est, "probe_mode", "fd"))
        if probe_mode == "fd" and normalize_precision(precision) != tf.float64:
            # Finite-difference probing of the Hessian stencil is only reliable
            # in double precision; in single it produced an indefinite stencil
            # and a NaN preconditioner (estimates collapsed to zero).
            print(
                "[error_estimator] probe_mode=fd is unreliable in single precision; "
                "using probe_mode=autodiff instead.",
                flush=True,
            )
            probe_mode = "autodiff"

        cg_iters = _cfg_get(cfg_est, "cg_iters", [10])
        if isinstance(cg_iters, int):
            cg_iters = [cg_iters]
        cg_iters = [int(k) for k in list(cg_iters)]

        multigrid = _cfg_get(cfg_est, "multigrid", None)
        if multigrid is None:
            preconditioner_options = {}
        elif isinstance(multigrid, DictConfig):
            preconditioner_options = dict(
                OmegaConf.to_container(multigrid, resolve=True)
            )
        else:
            preconditioner_options = dict(multigrid)

        return {
            "cost_fn": cost_fn,
            "bcs": list(mapping.apply_bcs),
            "field_shape": (1, Nz, ny, nx),
            "precision": precision,
            "basis_vertical": str(cfg_numerics.get("basis_vertical", "")),
            "V_s": discr_v.V_s,
            "idx_thk": inputs.index("thk"),
            "idx_usurf": _optional_index(inputs, "usurf"),
            "topg": topg,
            "basin_mask": basin_mask,
            "rho_ice": float(cfg_physics.ice_density),
            "rho_water": float(cfg_physics.water_density),
            "water_level": water_level,
            "freq": int(_cfg_get(cfg_est, "freq", 250)),
            "estimate_at_start": bool(_cfg_get(cfg_est, "estimate_at_start", True)),
            "cg_iters": cg_iters,
            "cg_tol": float(_cfg_get(cfg_est, "cg_tol", 1.0e-6)),
            "hvp_mode": str(_cfg_get(cfg_est, "hvp_mode", "banded")),
            "probe_mode": probe_mode,
            "hvp_verify": bool(_cfg_get(cfg_est, "hvp_verify", False)),
            "preconditioner": str(
                _cfg_get(cfg_est, "preconditioner", "barotropic_multigrid")
            ),
            "preconditioner_options": preconditioner_options,
            "damping": float(_cfg_get(cfg_est, "damping", 1.0e-16)),
            "operator_update_freq": int(_cfg_get(cfg_est, "operator_update_freq", 0)),
            "operator_refresh_rel_change": float(
                _cfg_get(cfg_est, "operator_refresh_rel_change", 0.05)
            ),
            "newton_steps": int(_cfg_get(cfg_est, "newton_steps", 2)),
            "disable_xla": bool(_cfg_get(cfg_est, "disable_xla", True)),
            "record_path": _cfg_get(cfg_est, "record_path", "error_estimate.jsonl"),
            "save_fields_dir": _cfg_get(cfg_est, "save_fields_dir", None),
            "publish_field": bool(_cfg_get(cfg_est, "publish_field", True)),
            "reference": reference,
            "state": state,
            "verbose": bool(_cfg_get(cfg_est, "verbose", True)),
        }
