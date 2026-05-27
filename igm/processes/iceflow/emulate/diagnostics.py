"""Online per-retrain diagnostics for the iceflow emulator.

Captures cost metrics (J_pre, J_post, warmstart gap) and velocity metrics
(RMS change, margin vs. bulk breakdown) for each retraining event.

All calls sit on the eager side (around `update_emulator`), so
`@tf.function` tracing is never disturbed. No-op unless the config flag
`processes.iceflow.emulator.diagnostics.enabled` (or the unified variant)
is true.
"""

from __future__ import annotations

import csv
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import tensorflow as tf

from igm.processes.iceflow.energy.energy import iceflow_energy_XY, iceflow_energy_UV
from igm.processes.iceflow.utils.data_preprocessing import Y_to_UV


# ─── Weight save / load ───────────────────────────────────────────────────────

def save_weights(optimizer, path: str) -> None:
    """Save the network's trainable variables to a .npz file."""
    arrays = {f"w{i}": w.numpy() for i, w in enumerate(optimizer.map.get_theta())}
    np.savez(path, **arrays)


def load_weights(optimizer, path: str) -> None:
    """Restore the network's trainable variables from a .npz file."""
    data = np.load(path)
    theta = list(optimizer.map.get_theta())
    for i, var in enumerate(theta):
        var.assign(data[f"w{i}"])


# ─── Velocity save / load ─────────────────────────────────────────────────────

def save_velocities(state, path: str) -> None:
    """Save ubar, vbar after training to a .npz file for warm-start velocity analysis."""
    ubar = state.ubar.numpy() if hasattr(state.ubar, "numpy") else np.array(state.ubar)
    vbar = state.vbar.numpy() if hasattr(state.vbar, "numpy") else np.array(state.vbar)
    np.savez(path, ubar=ubar, vbar=vbar)


def load_velocities(path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load previously saved ubar, vbar for input-change velocity analysis."""
    try:
        data = np.load(path)
        return data["ubar"], data["vbar"]
    except Exception:
        return None, None


# ─── Energy helpers ───────────────────────────────────────────────────────────

def _eval_energy(bag: Dict[str, Any], X: tf.Tensor, parameters) -> float:
    """Evaluate the Molho/BP energy at the current weights for input X."""
    Nx = parameters.Nx
    Ny = parameters.Ny
    iz = parameters.iz

    total = 0.0
    for i in range(X.shape[0]):
        Y = bag["iceflow_model_inference"](
            tf.pad(X[i, :, :, :, :], bag["PAD"], "CONSTANT")
        )[:, :Ny, :Nx, :]

        energy = iceflow_energy_XY(
            Nz=parameters.Nz,
            fieldin_names=parameters.fieldin_names,
            X=X[i, :, iz : Ny - iz, iz : Nx - iz, :],
            Y=Y[:, iz : Ny - iz, iz : Nx - iz, :],
            discr_h=bag["discr_h"],
            discr_v=bag["discr_v"],
            energy_components=bag["energy_components"],
            batch_size=bag["batch_size"],
            Ny=Ny - 2 * iz,
            Nx=Nx - 2 * iz,
        )
        energy_mean = tf.reduce_mean(energy, axis=[1, 2, 3])
        total += float(tf.reduce_sum(energy_mean).numpy())
    return total


def _unified_evaluate_J(optimizer, inputs: tf.Tensor) -> float:
    """Evaluate cost_fn(U, V, inputs) using current theta, no tape."""
    U, V = optimizer.map.get_UV(inputs)
    return float(tf.reduce_sum(optimizer.cost_fn(U, V, inputs)).numpy())


def eval_energy_standalone(
    model: tf.keras.Model,
    X: tf.Tensor,
    discr_h,
    discr_v,
    energy_components,
    Nz: int = 2,
    inputs_names: Optional[Tuple] = None,
) -> float:
    """Evaluate the iceflow energy without a full IGM state object.

    Useful for standalone diagnostic scripts that construct the model and
    discretization objects directly (not through the IGM simulation loop).

    Parameters
    ----------
    model            : Keras model that maps X → Y  ([B, Ny, Nx, nb_inputs] → [B, Ny, Nx, 2*Nz])
    X                : float32 tensor [B, Ny, Nx, nb_inputs]
    discr_h          : HorizontalDiscr (e.g. Q1Discr)
    discr_v          : VerticalDiscr   (e.g. MOLHODiscr)
    energy_components: list of EnergyComponent objects
    Nz               : number of vertical layers (default 2 for Molho)
    inputs_names     : tuple of input field names; defaults to
                       ("thk", "usurf", "arrhenius", "slidingco", "dX")

    Returns
    -------
    float : total energy (sum over components and spatial mean)
    """
    if inputs_names is None:
        inputs_names = ("thk", "usurf", "arrhenius", "slidingco", "dX")

    Y = model(X, training=False)
    U, V = Y_to_UV(Nz, Y)
    e = iceflow_energy_UV(
        inputs_names=inputs_names,
        inputs=X,
        U=U,
        V=V,
        discr_h=discr_h,
        discr_v=discr_v,
        energy_components=energy_components,
    )
    # e: [n_components, B, Ny-1, Nx-1]
    return float(tf.reduce_sum(tf.reduce_mean(e, axis=[1, 2, 3])).numpy())


# ─── Velocity helpers ─────────────────────────────────────────────────────────

def _compute_margin_mask(thk_np: np.ndarray, margin_width: int = 3) -> np.ndarray:
    """Boolean mask of ice pixels within `margin_width` cells of the ice edge."""
    ice = thk_np > 0.0
    try:
        from scipy.ndimage import binary_dilation
        near_no_ice = binary_dilation(~ice, iterations=margin_width)
    except ImportError:
        # Fallback: single-cell boundary
        no_ice = ~ice
        near_no_ice = np.zeros_like(ice)
        near_no_ice[1:, :] |= no_ice[:-1, :]
        near_no_ice[:-1, :] |= no_ice[1:, :]
        near_no_ice[:, 1:] |= no_ice[:, :-1]
        near_no_ice[:, :-1] |= no_ice[:, 1:]
    return ice & near_no_ice


def _eval_velocity_emulator(
    cfg, state
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Forward-pass at current emulator weights; returns (ubar, vbar) as numpy [Ny, Nx]."""
    try:
        from .emulated import update_iceflow_emulated
        update_iceflow_emulated(cfg, state)
        ubar = state.ubar.numpy() if hasattr(state.ubar, "numpy") else np.array(state.ubar)
        vbar = state.vbar.numpy() if hasattr(state.vbar, "numpy") else np.array(state.vbar)
        return ubar, vbar
    except Exception:
        return None, None


def _eval_velocity_unified(
    cfg, state
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Evaluate unified mapping at current weights; returns (ubar, vbar) as numpy [Ny, Nx]."""
    try:
        from igm.processes.iceflow.unified.evaluator import evaluate_iceflow
        evaluate_iceflow(cfg, state)
        ubar = state.ubar.numpy() if hasattr(state.ubar, "numpy") else np.array(state.ubar)
        vbar = state.vbar.numpy() if hasattr(state.vbar, "numpy") else np.array(state.vbar)
        return ubar, vbar
    except Exception:
        return None, None


def _velocity_metrics(
    ubar_b: Optional[np.ndarray],
    vbar_b: Optional[np.ndarray],
    ubar_a: Optional[np.ndarray],
    vbar_a: Optional[np.ndarray],
    ice_mask: np.ndarray,
    margin_mask: np.ndarray,
    nbit: int,
) -> Tuple[float, float, float, float, float]:
    """Compute RMS velocity-change metrics (m/yr) split by glacier region.

    Returns (dU_l2_mask, dU_l2_per_iter, dU_margin_l2, dU_bulk_l2, margin_frac).
    All NaN if either before/after velocity is unavailable.
    """
    nan5 = (float("nan"),) * 5
    if ubar_b is None or ubar_a is None:
        return nan5
    try:
        du2 = (ubar_a - ubar_b) ** 2 + (vbar_a - vbar_b) ** 2  # [Ny, Nx]
        bulk_mask = ice_mask & ~margin_mask

        n_ice    = ice_mask.sum()
        n_margin = margin_mask.sum()
        n_bulk   = bulk_mask.sum()

        if n_ice == 0:
            return nan5

        dU_l2      = float(np.sqrt(du2[ice_mask].sum()    / n_ice))
        dU_margin  = float(np.sqrt(du2[margin_mask].sum() / n_margin)) if n_margin > 0 else float("nan")
        dU_bulk    = float(np.sqrt(du2[bulk_mask].sum()   / n_bulk))   if n_bulk   > 0 else float("nan")
        total_e    = du2[ice_mask].sum()
        margin_frac = float(du2[margin_mask].sum() / total_e) if (total_e > 0 and n_margin > 0) else float("nan")
        dU_per_iter = (dU_l2 / nbit) if nbit > 0 else float("nan")

        return dU_l2, dU_per_iter, dU_margin, dU_bulk, margin_frac
    except Exception:
        return nan5


# ─── Config helpers ───────────────────────────────────────────────────────────

def _diag_cfg(cfg):
    """Return the diagnostics sub-config for whichever iceflow branch is active."""
    method = getattr(cfg.processes.iceflow, "method", "emulated")
    if method == "unified":
        ucfg = cfg.processes.iceflow.unified
        if hasattr(ucfg, "diagnostics"):
            return ucfg.diagnostics
    return cfg.processes.iceflow.emulator.diagnostics


def is_enabled(cfg) -> bool:
    try:
        return bool(_diag_cfg(cfg).enabled)
    except Exception:
        return False


def _resolve_output_dir(cfg) -> str:
    diag_cfg = _diag_cfg(cfg)
    out = str(diag_cfg.output_dir) if diag_cfg.output_dir else ""
    if not out:
        out = os.path.join(os.getcwd(), "diagnostics")
    os.makedirs(out, exist_ok=True)
    return out


# ─── CSV helpers ──────────────────────────────────────────────────────────────

_CSV_FIELDS = [
    # identity / context
    "retrain_index",
    "it",
    "t",
    "initial",
    "warm_up",
    # training hyperparameters
    "nbit",
    "lr",
    # cost metrics
    "J_pre",
    "J_post",
    "delta_J",
    "J_post_prev",
    "dJ_jump",
    # velocity metrics from training (m/yr, glacier-masked): change due to warm-retrain
    "dU_l2_mask",
    "dU_l2_mask_per_iter",
    "dU_margin_l2",
    "dU_bulk_l2",
    "margin_frac",
    # input-change velocity metrics (sequential warm-start only):
    # change in U due to geometry switch alone (with old θ, new X vs old X)
    "dU_input_l2_mask",
    "dU_input_margin_l2",
    "dU_input_bulk_l2",
    "dU_input_margin_frac",
]


def _csv_path(state) -> str:
    return os.path.join(state.diagnostics["output_dir"], "diagnostics.csv")


def _append_row(state, row: Dict[str, Any]) -> None:
    path = _csv_path(state)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in _CSV_FIELDS})


# ─── Initialization ───────────────────────────────────────────────────────────

def maybe_initialize(cfg, state) -> None:
    """Create the diagnostics state container. No-op if flag is off."""
    if not is_enabled(cfg):
        state.diagnostics = {"enabled": False}
        return

    out_dir = _resolve_output_dir(cfg)
    state.diagnostics = {
        "enabled":       True,
        "output_dir":    out_dir,
        "retrain_index": 0,
        "prev_J_post":   float("nan"),
    }


# ─── Unified path hooks ───────────────────────────────────────────────────────

def before_retrain_unified(
    cfg, state, optimizer, inputs: tf.Tensor, status
) -> Optional[Dict[str, Any]]:
    if not state.diagnostics.get("enabled", False):
        return None

    dcfg = _diag_cfg(cfg)

    # Warm-start: load network weights before evaluating J_pre so that
    # J_pre = J(θ_prev, X_new) (cost of previous model on new geometry).
    weights_loaded = False
    try:
        lwp = str(getattr(dcfg, "load_weights_path", "") or "")
        if lwp and os.path.exists(lwp):
            load_weights(optimizer, lwp)
            weights_loaded = True
    except Exception as exc:
        print(f"[diagnostics] failed to load weights from {lwp}: {exc}")

    ctx: Dict[str, Any] = {
        "status": str(getattr(status, "name", status)),
        "it":     int(getattr(state, "it", 0) or 0),
        "t":      float(state.t.numpy()) if hasattr(state, "t") else 0.0,
        "inputs": inputs,
    }

    try:
        ctx["J_pre"] = _unified_evaluate_J(optimizer, inputs)
    except Exception:
        ctx["J_pre"] = float("nan")

    # When warm-starting, evaluate U(θ_prev, X_new) explicitly so that
    # ubar_before reflects the loaded model (not IGM's initial zero state).
    # This ensures dU_l2_mask = ||U(θ_new, X_new) - U(θ_prev, X_new)||.
    if weights_loaded:
        try:
            ubar_prev_xnew, vbar_prev_xnew = _eval_velocity_unified(cfg, state)
        except Exception:
            ubar_prev_xnew, vbar_prev_xnew = None, None
    else:
        ubar_prev_xnew = (
            state.ubar.numpy() if hasattr(state, "ubar") and state.ubar is not None else None
        )
        vbar_prev_xnew = (
            state.vbar.numpy() if hasattr(state, "vbar") and state.vbar is not None else None
        )

    ctx["ubar_before"] = ubar_prev_xnew
    ctx["vbar_before"] = vbar_prev_xnew

    try:
        thk_np = state.thk.numpy() if hasattr(state.thk, "numpy") else np.array(state.thk)
        ctx["ice_mask"]    = thk_np > 0.0
        ctx["margin_mask"] = _compute_margin_mask(thk_np)
    except Exception:
        ctx["ice_mask"]    = None
        ctx["margin_mask"] = None

    # Input-change velocity metrics: dU_input = ||U(θ_prev, X_new) - U(θ_prev, X_prev)||
    # Requires a saved velocity file from the previous run.
    ctx["dU_input_metrics"] = (float("nan"),) * 4
    try:
        lvp = str(getattr(dcfg, "load_velocities_path", "") or "")
        if lvp and os.path.exists(lvp) and ubar_prev_xnew is not None:
            prev_ubar, prev_vbar = load_velocities(lvp)
            dU_in_l2, _, dU_in_mg, dU_in_bk, mg_frac_in = _velocity_metrics(
                prev_ubar, prev_vbar,
                ubar_prev_xnew, vbar_prev_xnew,
                ctx.get("ice_mask"), ctx.get("margin_mask"),
                nbit=1,
            )
            ctx["dU_input_metrics"] = (dU_in_l2, dU_in_mg, dU_in_bk, mg_frac_in)
    except Exception as exc:
        print(f"[diagnostics] failed to compute input-change velocity metrics: {exc}")

    return ctx


def after_retrain_unified(
    cfg,
    state,
    optimizer,
    inputs: tf.Tensor,
    ctx: Optional[Dict[str, Any]],
    cost_tensor,
) -> None:
    if ctx is None or not state.diagnostics.get("enabled", False):
        return

    if cost_tensor is not None and hasattr(cost_tensor, "numpy"):
        cost_np = np.asarray(cost_tensor.numpy()).ravel().astype(np.float64)
    else:
        cost_np = np.asarray([], dtype=np.float64)
    nbit = int(cost_np.size)

    try:
        J_post = _unified_evaluate_J(optimizer, inputs)
    except Exception:
        J_post = float("nan")

    J_pre      = ctx.get("J_pre", float("nan"))
    delta_J    = (J_pre - J_post) if np.isfinite(J_pre) and np.isfinite(J_post) else float("nan")
    J_post_prev = state.diagnostics.get("prev_J_post", float("nan"))
    dJ_jump     = (J_pre - J_post_prev) if np.isfinite(J_pre) and np.isfinite(J_post_prev) else float("nan")

    ubar_a, vbar_a = _eval_velocity_unified(cfg, state)
    dU_l2, dU_pi, dU_mg, dU_bk, mg_frac = _velocity_metrics(
        ctx.get("ubar_before"), ctx.get("vbar_before"),
        ubar_a, vbar_a,
        ctx.get("ice_mask"), ctx.get("margin_mask"),
        nbit,
    )

    dU_in_l2, dU_in_mg, dU_in_bk, dU_in_mgfrac = ctx.get(
        "dU_input_metrics", (float("nan"),) * 4
    )

    idx = int(state.diagnostics["retrain_index"])
    row = {
        "retrain_index":        idx,
        "it":                   ctx["it"],
        "t":                    ctx["t"],
        "initial":              int(ctx["status"] == "INIT"),
        "warm_up":              int(ctx["status"] == "WARM_UP"),
        "nbit":                 nbit,
        "lr":                   float(optimizer.optim_adam.learning_rate.numpy())
                                if hasattr(optimizer, "optim_adam") else float("nan"),
        "J_pre":                J_pre,
        "J_post":               J_post,
        "delta_J":              delta_J,
        "J_post_prev":          J_post_prev,
        "dJ_jump":              dJ_jump,
        "dU_l2_mask":           dU_l2,
        "dU_l2_mask_per_iter":  dU_pi,
        "dU_margin_l2":         dU_mg,
        "dU_bulk_l2":           dU_bk,
        "margin_frac":          mg_frac,
        "dU_input_l2_mask":     dU_in_l2,
        "dU_input_margin_l2":   dU_in_mg,
        "dU_input_bulk_l2":     dU_in_bk,
        "dU_input_margin_frac": dU_in_mgfrac,
    }
    _append_row(state, row)

    state.diagnostics["prev_J_post"]   = J_post
    state.diagnostics["retrain_index"] = idx + 1

    dcfg = _diag_cfg(cfg)

    # Save weights for downstream warm-start runs
    try:
        swp = str(getattr(dcfg, "save_weights_path", "") or "")
        if swp:
            os.makedirs(os.path.dirname(swp) or ".", exist_ok=True)
            save_weights(optimizer, swp)
    except Exception as exc:
        print(f"[diagnostics] failed to save weights to {swp}: {exc}")

    # Save velocities for downstream input-change velocity analysis
    try:
        svp = str(getattr(dcfg, "save_velocities_path", "") or "")
        if svp and ubar_a is not None:
            os.makedirs(os.path.dirname(svp) or ".", exist_ok=True)
            # ubar_a / vbar_a are numpy arrays from _eval_velocity_unified above
            np.savez(svp, ubar=ubar_a,
                     vbar=vbar_a if vbar_a is not None else np.zeros_like(ubar_a))
    except Exception as exc:
        print(f"[diagnostics] failed to save velocities to {svp}: {exc}")

    if idx == 0:
        try:
            _maybe_write_meta_unified(cfg, state, optimizer, inputs)
        except Exception as exc:
            print(f"[diagnostics] failed to write meta.json: {exc}")


def _maybe_write_meta_unified(cfg, state, optimizer, inputs: tf.Tensor) -> None:
    meta_path = os.path.join(state.diagnostics["output_dir"], "meta.json")
    if os.path.exists(meta_path):
        return
    import json
    import pickle
    from omegaconf import OmegaConf

    cfg_obj = OmegaConf.to_container(cfg, resolve=True) if hasattr(cfg, "keys") else cfg
    cfg_bytes = pickle.dumps(cfg_obj)

    shape = list(inputs.shape)
    try:
        arch = str(cfg.processes.iceflow.unified.network.architecture)
    except Exception:
        arch = str(cfg.processes.iceflow.unified.mapping)
    meta = {
        "method":             "unified",
        "architecture":       arch,
        "mapping":            str(cfg.processes.iceflow.unified.mapping),
        "inputs_shape":       shape,
        "fieldin_names":      list(cfg.processes.iceflow.unified.inputs),
        "Nz":                 int(cfg.processes.iceflow.numerics.Nz),
        "n_trainable_arrays": len(list(optimizer.map.get_theta())),
        "cfg_pickle_hex":     cfg_bytes.hex(),
    }
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)


# ─── Emulator path hooks ──────────────────────────────────────────────────────

def before_retrain(
    cfg,
    state,
    bag: Dict[str, Any],
    X: tf.Tensor,
    parameters,
    initial: bool,
    warm_up: bool,
    nbit: int,
    lr: float,
) -> Optional[Dict[str, Any]]:
    """Snapshot pre-retrain quantities. Returns a context dict for `after_retrain`."""
    if not state.diagnostics.get("enabled", False):
        return None

    ctx: Dict[str, Any] = {
        "initial":  bool(initial),
        "warm_up":  bool(warm_up),
        "nbit":     int(nbit),
        "lr":       float(lr),
        "it":       int(getattr(state, "it", 0) or 0),
        "t":        float(state.t.numpy()) if hasattr(state, "t") else 0.0,
    }

    try:
        ctx["J_pre"] = _eval_energy(bag, X, parameters)
    except Exception:
        ctx["J_pre"] = float("nan")

    ctx["ubar_before"] = (
        state.ubar.numpy() if hasattr(state, "ubar") and state.ubar is not None else None
    )
    ctx["vbar_before"] = (
        state.vbar.numpy() if hasattr(state, "vbar") and state.vbar is not None else None
    )
    try:
        thk_np = state.thk.numpy() if hasattr(state.thk, "numpy") else np.array(state.thk)
        ctx["ice_mask"]    = thk_np > 0.0
        ctx["margin_mask"] = _compute_margin_mask(thk_np)
    except Exception:
        ctx["ice_mask"]    = None
        ctx["margin_mask"] = None

    return ctx


def after_retrain(
    cfg,
    state,
    bag: Dict[str, Any],
    X: tf.Tensor,
    parameters,
    ctx: Optional[Dict[str, Any]],
    cost_tensor: tf.Tensor,
) -> None:
    """Compute post-retrain quantities and log a CSV row."""
    if ctx is None or not state.diagnostics.get("enabled", False):
        return

    cost_np = np.asarray(cost_tensor.numpy()).ravel().astype(np.float64)
    nbit    = int(cost_np.size)

    try:
        J_post = _eval_energy(bag, X, parameters)
    except Exception:
        J_post = float("nan")

    J_pre      = ctx.get("J_pre", float("nan"))
    delta_J    = (J_pre - J_post) if np.isfinite(J_pre) and np.isfinite(J_post) else float("nan")
    J_post_prev = state.diagnostics.get("prev_J_post", float("nan"))
    dJ_jump     = (J_pre - J_post_prev) if np.isfinite(J_pre) and np.isfinite(J_post_prev) else float("nan")

    ubar_a, vbar_a = _eval_velocity_emulator(cfg, state)
    dU_l2, dU_pi, dU_mg, dU_bk, mg_frac = _velocity_metrics(
        ctx.get("ubar_before"), ctx.get("vbar_before"),
        ubar_a, vbar_a,
        ctx.get("ice_mask"), ctx.get("margin_mask"),
        nbit,
    )

    idx = int(state.diagnostics["retrain_index"])
    row = {
        "retrain_index":       idx,
        "it":                  ctx["it"],
        "t":                   ctx["t"],
        "initial":             int(ctx["initial"]),
        "warm_up":             int(ctx["warm_up"]),
        "nbit":                ctx["nbit"],
        "lr":                  ctx["lr"],
        "J_pre":               J_pre,
        "J_post":              J_post,
        "delta_J":             delta_J,
        "J_post_prev":         J_post_prev,
        "dJ_jump":             dJ_jump,
        "dU_l2_mask":          dU_l2,
        "dU_l2_mask_per_iter": dU_pi,
        "dU_margin_l2":        dU_mg,
        "dU_bulk_l2":          dU_bk,
        "margin_frac":         mg_frac,
    }
    _append_row(state, row)

    state.diagnostics["prev_J_post"]   = J_post
    state.diagnostics["retrain_index"] = idx + 1

    if idx == 0:
        try:
            _maybe_write_meta(cfg, state, bag, parameters)
        except Exception as exc:
            print(f"[diagnostics] failed to write meta.json: {exc}")


def _maybe_write_meta(cfg, state, bag: Dict[str, Any], parameters) -> None:
    meta_path = os.path.join(state.diagnostics["output_dir"], "meta.json")
    if os.path.exists(meta_path):
        return
    import json
    import pickle
    from omegaconf import OmegaConf

    discr_v = bag.get("discr_v")
    discr_v_list = (
        discr_v.numpy().tolist() if hasattr(discr_v, "numpy") else None
    )
    cfg_obj = OmegaConf.to_container(cfg, resolve=True) if hasattr(cfg, "keys") else cfg
    cfg_bytes = pickle.dumps(cfg_obj)

    arch = cfg.processes.iceflow.emulator.network.architecture
    nb_inputs  = len(cfg.processes.iceflow.emulator.fieldin)
    nb_outputs = 2 * cfg.processes.iceflow.numerics.Nz

    meta = {
        "architecture":  str(arch),
        "nb_inputs":     int(nb_inputs),
        "nb_outputs":    int(nb_outputs),
        "Nx":            int(parameters.Nx),
        "Ny":            int(parameters.Ny),
        "Nz":            int(parameters.Nz),
        "iz":            int(parameters.iz),
        "fieldin_names": list(parameters.fieldin_names),
        "PAD":           [[int(x) for x in row] for row in state.PAD.numpy().tolist()]
                         if hasattr(state.PAD, "numpy") else state.PAD,
        "batch_size":    int(bag["batch_size"]),
        "discr_h":       float(bag["discr_h"].numpy() if hasattr(bag["discr_h"], "numpy") else bag["discr_h"]),
        "discr_v":       discr_v_list,
        "cfg_pickle_hex": cfg_bytes.hex(),
    }
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
