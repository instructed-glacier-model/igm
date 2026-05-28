#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Generic relaxation-based data assimilation.

A relaxation run is a list of independent ``steps``. Each step is an
orthogonal triple

    (residual, update law, control)

with optional modifiers (mask, cadence, start/end time, smoother,
control bounds, geometry policy). Inside one outer time loop:

    1. forward_model.update                     (e.g. iceflow)
    2. aux_processes update                     (configurable list)
    3. ensure derived fields                    (velsurf_mag, divflux)
    4. advance time                             (CFL-limited iff a step writes a geometry control)
    5. for each step that is due:
           r := residual
           r := mask_aware_smooth(r) if requested
           ΔC = update_law(C, r, dt)
           C ← clip(C + ΔC,  control.bounds)
           apply geometry_policy if writing thk / topg / usurf
    6. snapshot (output hooks + misfit CSV)

state.t is deleted at exit so the outer IGM loop short-circuits.

"""

import importlib
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import tensorflow as tf

from igm.utils.grad.compute_divflux import compute_divflux


# ===================================================================== #
#  Residuals                                                            #
# ===================================================================== #
#
#   linear     r = T - M
#   relative   r = (T - M) / max(|T|, eps)
#   log_ratio  r = log(max(M, eps) / max(T, eps))
#

def _resid_linear(T, M, eps):
    return T - M


def _resid_relative(T, M, eps):
    return (T - M) / tf.maximum(tf.abs(T), tf.cast(eps, T.dtype))


def _resid_log_ratio(T, M, eps):
    e = tf.cast(eps, T.dtype)
    return tf.math.log(tf.maximum(M, e) / tf.maximum(T, e))


_RESIDUALS = {
    "linear":    _resid_linear,
    "relative":  _resid_relative,
    "log_ratio": _resid_log_ratio,
}


# ===================================================================== #
#  Update laws                                                          #
# ===================================================================== #
#
# All laws share the signature  (C, r, α, dt) -> C_new  and accept an
# optional r_max that clips r to [-r_max, +r_max] before applying.
#
#   additive              C ← C + α · r · dt
#   multiplicative        C ← C · exp(α · r · dt)            (exact ODE)
#   multiplicative_linear C ← C · (1 + α · r · dt)           (Pollard-style)
#   replace               C ← α · r                          (absolute write; dt ignored)
#
# `dt` is the *effective* time over which the residual integrates. For
# per-step kernels (cadence = 0) it is the outer-loop state.dt; for
# cadenced kernels (cadence > 0) it is 1.0, so α is interpreted as a
# pure per-application gain (matches the legacy "every N years, apply
# this multiplier" convention used by the friction kernels).
#

def _upd_additive(C, r, alpha, dt, r_max=None):
    if r_max is not None:
        r = tf.clip_by_value(r, -r_max, r_max)
    return C + alpha * r * dt


def _upd_multiplicative(C, r, alpha, dt, r_max=None):
    if r_max is not None:
        r = tf.clip_by_value(r, -r_max, r_max)
    return C * tf.exp(alpha * r * dt)


def _upd_multiplicative_linear(C, r, alpha, dt, r_max=None):
    if r_max is not None:
        r = tf.clip_by_value(r, -r_max, r_max)
    return C * (tf.ones_like(C) + alpha * r * dt)


def _upd_replace(C, r, alpha, dt, r_max=None):
    """Absolute write C := α·r (dt is ignored). Pair with residual.kind=linear
    for legacy 'smb := α·(usurf_obs − usurf)' style surface_match."""
    if r_max is not None:
        r = tf.clip_by_value(r, -r_max, r_max)
    return alpha * r


_UPDATE_LAWS = {
    "additive":              _upd_additive,
    "multiplicative":        _upd_multiplicative,
    "multiplicative_linear": _upd_multiplicative_linear,
    "replace":               _upd_replace,
}


# ===================================================================== #
#  Smoother (TF-only, mask-aware Gaussian)                              #
# ===================================================================== #

def _gaussian_kernel_1d(sigma, dtype):
    radius = max(1, int(np.ceil(3.0 * float(sigma))))
    x = tf.range(-radius, radius + 1, dtype=dtype)
    g = tf.exp(-0.5 * (x / tf.cast(sigma, dtype)) ** 2)
    return g / tf.reduce_sum(g), radius


def _gauss_filter_2d(field2d, sigma):
    """Separable Gaussian via two 1D conv2d passes; reflective padding."""
    g, radius = _gaussian_kernel_1d(sigma, field2d.dtype)
    f = field2d[tf.newaxis, ..., tf.newaxis]               # [1, H, W, 1]
    pad = [[0, 0], [radius, radius], [radius, radius], [0, 0]]
    f = tf.pad(f, pad, mode="REFLECT")
    kx = g[tf.newaxis, :, tf.newaxis, tf.newaxis]          # [1, K, 1, 1]
    ky = g[:, tf.newaxis, tf.newaxis, tf.newaxis]          # [K, 1, 1, 1]
    f = tf.nn.conv2d(f, kx, strides=[1, 1, 1, 1], padding="VALID")
    f = tf.nn.conv2d(f, ky, strides=[1, 1, 1, 1], padding="VALID")
    return f[0, ..., 0]


def _smooth(field2d, sigma, mask=None, mask_aware=True):
    if sigma is None or float(sigma) <= 0.0:
        return field2d
    if mask is not None and mask_aware:
        num = _gauss_filter_2d(field2d * mask, sigma)
        den = _gauss_filter_2d(mask, sigma)
        out = num / tf.maximum(den, tf.cast(1e-6, field2d.dtype))
        return tf.where(mask > 0.5, out, field2d)
    return _gauss_filter_2d(field2d, sigma)


# ===================================================================== #
#  Mask resolution                                                      #
# ===================================================================== #

def _resolve_mask(state, spec):
    """Return a float mask (1.0/0.0) or None if no masking is requested.

    spec accepts: None, "none", or a state attribute name. Anything more
    complex (composite masks) should be precomputed by an upstream module
    so the relaxation driver stays simple.
    """
    if spec is None or spec == "none" or spec == "":
        return None
    if not isinstance(spec, str):
        raise ValueError(
            f"time_relaxation: mask spec must be a state attribute name or None, "
            f"got {spec!r}. Precompute composite masks in an upstream module."
        )
    if not hasattr(state, spec):
        raise RuntimeError(
            f"time_relaxation: mask references state.{spec} which is not present."
        )
    m = getattr(state, spec)
    return tf.cast(tf.cast(m, tf.float32) > 0.5, tf.float32)


# ===================================================================== #
#  Step config                                                          #
# ===================================================================== #

@dataclass
class _Step:
    name: str
    # residual
    residual_kind: str
    target: str
    current: str
    eps: float
    # update
    update_kind: str
    alpha: float
    r_max: Optional[float]
    apply_mode: str                # "per_step" | "per_application"
    # control
    control_field: str
    control_bounds: Optional[Tuple[float, float]]
    control_outside_mask: Optional[float]
    control_floor_at: Optional[str]    # state attr; new_C ← max(new_C, state.<attr>)
    control_ceil_at: Optional[str]     # state attr; new_C ← min(new_C, state.<attr>)
    # modifiers
    mask_spec: Any
    cadence: float
    start_time: float
    end_time: float
    smoother_sigma: float
    smoother_mask_aware: bool
    geometry_policy: str
    shares_residual_with: Optional[str]
    # internal clock — initialised at loop start; never None during the loop
    last_applied_time: Optional[tf.Variable] = None


def _build_steps(steps_cfg):
    """Build _Step objects from cfg blocks.

    Notes on config access:
      * ``s["update"]`` (not ``s.update``) — ``update`` collides with the
        ``dict.update`` method that DictConfig inherits, so attribute access
        returns the method rather than the YAML block.
      * Optional keys read via ``.get(name, default)``.
    """
    steps = []
    for s in steps_cfg:
        res = s["residual"]
        upd = s["update"]                              # bracket: see note above
        ctl = s["control"]

        bounds = ctl.get("bounds", None)
        if bounds is not None:
            bounds = (float(bounds[0]), float(bounds[1]))
        outside_mask = ctl.get("outside_mask", None)
        outside_mask = float(outside_mask) if outside_mask is not None else None
        floor_at = ctl.get("floor_at", None)
        ceil_at = ctl.get("ceil_at", None)

        smoother = s.get("smoother", None)
        sigma = float(smoother.get("sigma", 0.0)) if smoother is not None else 0.0
        mask_aware = bool(smoother.get("mask_aware", True)) if smoother is not None else True
        r_max_raw = upd.get("r_max", None)

        cadence = float(s.get("cadence", 0.0))
        # Default apply mode: per_application iff cadenced, per_step otherwise.
        apply_mode = str(upd.get("apply",
                                 "per_application" if cadence > 0.0 else "per_step"))

        steps.append(_Step(
            name=str(s["name"]),
            residual_kind=str(res["kind"]),
            target=str(res["target"]),
            current=str(res.get("current", "")),
            eps=float(res.get("eps", 1.0e-3)),
            update_kind=str(upd.get("kind", "additive")),
            alpha=float(upd.get("alpha", 0.0)),
            r_max=(float(r_max_raw) if r_max_raw is not None else None),
            apply_mode=apply_mode,
            control_field=str(ctl["field"]),
            control_bounds=bounds,
            control_outside_mask=outside_mask,
            control_floor_at=(str(floor_at) if floor_at is not None else None),
            control_ceil_at=(str(ceil_at) if ceil_at is not None else None),
            mask_spec=s.get("mask", None),
            cadence=cadence,
            start_time=float(s.get("start_time", -1.0e30)),
            end_time=float(s.get("end_time", 1.0e30)),
            smoother_sigma=sigma,
            smoother_mask_aware=mask_aware,
            geometry_policy=str(s.get("geometry_policy", "none")),
            shares_residual_with=s.get("shares_residual_with", None),
        ))

    # Cross-reference validation
    names = {s.name for s in steps}
    for s in steps:
        if s.shares_residual_with and s.shares_residual_with not in names:
            raise ValueError(
                f"time_relaxation step {s.name!r}: shares_residual_with refers to "
                f"unknown step {s.shares_residual_with!r}."
            )
        if s.update_kind not in _UPDATE_LAWS:
            raise ValueError(
                f"time_relaxation step {s.name!r}: update.kind must be one of "
                f"{sorted(_UPDATE_LAWS)}, got {s.update_kind!r}."
            )
        if s.residual_kind not in _RESIDUALS:
            raise ValueError(
                f"time_relaxation step {s.name!r}: residual.kind must be one of "
                f"{sorted(_RESIDUALS)}, got {s.residual_kind!r}."
            )
        if s.apply_mode not in ("per_step", "per_application"):
            raise ValueError(
                f"time_relaxation step {s.name!r}: update.apply must be 'per_step' "
                f"or 'per_application', got {s.apply_mode!r}."
            )
    return steps


def _step_due(s, t):
    """A step is due if t is in [start_time, end_time] and either the step
    is per-iteration (cadence ≤ 0) or one full cadence has elapsed since
    the last application (last_applied_time is initialised to t_start)."""
    if t < s.start_time or t > s.end_time:
        return False
    if s.cadence <= 0.0:
        return True
    return (t - float(s.last_applied_time.numpy())) >= s.cadence


# ===================================================================== #
#  Apply one step                                                       #
# ===================================================================== #

def _compute_residual(s, state):
    if not hasattr(state, s.target):
        raise RuntimeError(
            f"time_relaxation step {s.name!r}: residual.target {s.target!r} is not "
            "on state. For target='amb', add an SMB module (e.g. smb_simple) to "
            "pre_processes and ensure the input NetCDF carries a `dhdt` field — "
            "_ensure_derived will then compute state.amb automatically."
        )
    T = getattr(state, s.target)
    if not hasattr(state, s.current):
        raise RuntimeError(
            f"time_relaxation step {s.name!r}: residual.current {s.current!r} is "
            "not on state."
        )
    M = getattr(state, s.current)
    return _RESIDUALS[s.residual_kind](T, M, s.eps)


def _apply_step(s, state, dt, residual_cache):
    # 1. residual (possibly shared)
    if s.shares_residual_with and s.shares_residual_with in residual_cache:
        r = residual_cache[s.shares_residual_with]
    else:
        r = _compute_residual(s, state)
        residual_cache[s.name] = r

    # 2. NaN safety — observation fields (e.g. uvelsurfobs) commonly have
    # no-data regions; propagate them as zero residual so the control is
    # not nudged where there's no observation.
    r = tf.where(tf.math.is_finite(r), r, tf.zeros_like(r))

    # 3. mask
    mask = _resolve_mask(state, s.mask_spec)
    if mask is not None:
        r = r * mask

    # 4. smoother
    if s.smoother_sigma > 0.0:
        r = _smooth(r, s.smoother_sigma, mask=mask,
                    mask_aware=s.smoother_mask_aware)

    # 5. update law
    if not hasattr(state, s.control_field):
        raise RuntimeError(
            f"time_relaxation step {s.name!r}: control.field "
            f"{s.control_field!r} is not on state."
        )
    C = getattr(state, s.control_field)
    fn = _UPDATE_LAWS[s.update_kind]
    alpha = tf.cast(s.alpha, C.dtype)

    # dt_eff:
    #   per_step         → outer-loop dt (current state.dt)
    #   per_application  → 1.0  (α is interpreted as a per-application gain;
    #                            this matches the legacy "every t_fr_update
    #                            years, multiply by (1+r)" friction formula)
    if s.apply_mode == "per_step":
        dt_eff = tf.cast(dt, C.dtype)
    else:
        dt_eff = tf.cast(1.0, C.dtype)

    r_t = tf.cast(r, C.dtype)
    r_max = tf.cast(s.r_max, C.dtype) if s.r_max is not None else None
    new_C = fn(C, r_t, alpha, dt_eff, r_max=r_max)

    # 5. clip — first per-pixel field bounds (floor_at / ceil_at), then
    # the scalar bounds. floor_at/ceil_at let the user enforce e.g. the
    # legacy `usurf >= topg` constraint of Frank/van Pelt.
    if s.control_floor_at is not None:
        floor_field = getattr(state, s.control_floor_at)
        new_C = tf.maximum(new_C, tf.cast(floor_field, new_C.dtype))
    if s.control_ceil_at is not None:
        ceil_field = getattr(state, s.control_ceil_at)
        new_C = tf.minimum(new_C, tf.cast(ceil_field, new_C.dtype))
    if s.control_bounds is not None:
        lo, hi = s.control_bounds
        new_C = tf.clip_by_value(new_C, tf.cast(lo, new_C.dtype),
                                        tf.cast(hi, new_C.dtype))

    # 6. outside-mask fill (e.g. legacy out_of_mask_smb)
    if s.control_outside_mask is not None and mask is not None:
        new_C = tf.where(
            mask > 0.5,
            new_C,
            tf.cast(s.control_outside_mask, new_C.dtype) * tf.ones_like(new_C),
        )

    # 7. write
    setattr(state, s.control_field, new_C)

    # 8. geometry policy
    _apply_geometry_policy(state, s.geometry_policy)

    # 9. clock
    s.last_applied_time.assign(state.t)


def _apply_geometry_policy(state, policy):
    """Restore the (thk, topg, usurf) constraint after a step that wrote one of them.

    none              do nothing (caller is responsible for consistency)
    recompute_usurf   usurf ← topg + thk          (use after writing thk or topg)
    recompute_topg    topg ← usurf − thk          (use after writing usurf)
                      then thk re-clipped to non-negative
    """
    if policy == "none":
        return
    if policy == "recompute_usurf":
        state.usurf = state.topg + state.thk
    elif policy == "recompute_topg":
        state.topg = state.usurf - state.thk
        state.thk = tf.maximum(state.usurf - state.topg, 0.0)
    else:
        raise ValueError(
            f"time_relaxation: unknown geometry_policy {policy!r}. "
            "Use one of: none, recompute_usurf, recompute_topg."
        )


# ===================================================================== #
#  Derived fields & time stepping                                       #
# ===================================================================== #

def _ensure_derived(state):
    """Refresh fields that residuals commonly reference and that depend on
    the forward model's just-updated velocities (velsurf_mag, divflux).
    Always recompute — the previous iteration's values are stale.

    Also exposes the apparent mass balance ``state.amb = state.smb −
    state.dhdt_obs`` (masked by ``state.icemask`` if present), so that
    Frank–van Pelt-style steps with ``residual.target: amb`` work
    out-of-the-box once an upstream SMB module (e.g. ``smb_simple``) and
    a ``dhdt_obs`` field (snapshotted from the input ``dhdt``) are
    available. ``state.amb`` is recomputed every iteration in case the
    SMB module updates ``state.smb`` over time.
    """
    if hasattr(state, "uvelsurf") and hasattr(state, "vvelsurf"):
        state.velsurf_mag = tf.sqrt(state.uvelsurf ** 2 + state.vvelsurf ** 2)
    if hasattr(state, "ubar") and hasattr(state, "vbar") and hasattr(state, "thk"):
        state.divflux = compute_divflux(
            state.ubar, state.vbar, state.thk, state.dx, state.dx
        )
    if hasattr(state, "smb") and hasattr(state, "dhdt_obs"):
        amb = state.smb - state.dhdt_obs
        if hasattr(state, "icemask"):
            amb = amb * state.icemask
        state.amb = amb


def _build_save_times(t_start, t_end, t_save):
    times = []
    k = 0
    while True:
        ts = t_start + k * t_save
        if ts > t_end + 1e-6:
            break
        times.append(round(ts, 6))
        k += 1
    if not times or times[-1] < t_end - 1e-6:
        times.append(round(t_end, 6))
    return times


def _advance_time(state, save_times, step_max, cfl, use_cfl):
    if use_cfl and hasattr(state, "ubar") and hasattr(state, "vbar"):
        velomax = tf.maximum(
            tf.reduce_max(tf.abs(state.ubar)),
            tf.reduce_max(tf.abs(state.vbar)),
        )
        vmax = float(velomax.numpy())
        dt_target = (
            float(tf.minimum(cfl * state.dx / velomax,
                             tf.constant(step_max, dtype=tf.float32)).numpy())
            if vmax > 0.0 else step_max
        )
    else:
        dt_target = step_max

    state.dt.assign(dt_target)

    if state.itsave + 1 < len(save_times):
        next_save = float(save_times[state.itsave + 1])
        if next_save <= float(state.t) + float(state.dt):
            state.dt.assign(next_save - float(state.t))
            state.saveresult = True
            state.itsave += 1
        else:
            state.saveresult = False
    else:
        state.saveresult = False

    if state.it >= 0:
        state.t.assign(state.t + state.dt)


# ===================================================================== #
#  Output hooks + misfit logger                                         #
# ===================================================================== #

def _collect_output_hooks(cfg):
    hooks = []
    outputs_cfg = getattr(cfg, "outputs", None)
    if outputs_cfg is None:
        return hooks
    for name in ("write_ncdf", "write_ts", "write_vtp"):
        if hasattr(outputs_cfg, name):
            mod = importlib.import_module(f"igm.outputs.{name}")
            hooks.append(mod.run)
    return hooks


def _build_misfit_logger(p_outputs, steps):
    if p_outputs is None:
        return None
    misfits = getattr(p_outputs, "misfits", None)
    if misfits is None:
        return None
    path = str(getattr(misfits, "path", "") or "")
    track = list(getattr(misfits, "track", []) or [])
    if not path or not track:
        return None

    step_lookup = {s.name: s for s in steps}
    cols = []
    for entry in track:
        nm = str(entry.step)
        kd = str(entry.kind)               # "rmse" | "mae"
        if nm not in step_lookup:
            raise ValueError(
                f"time_relaxation: outputs.misfits.track references unknown "
                f"step {nm!r}."
            )
        if kd not in ("rmse", "mae"):
            raise ValueError(
                f"time_relaxation: outputs.misfits.track[{nm}].kind must be "
                f"'rmse' or 'mae', got {kd!r}."
            )
        cols.append((nm, kd, f"{nm}_{kd}"))

    with open(path, "w") as f:
        f.write("t," + ",".join(c for _, _, c in cols) + "\n")

    def log(state):
        vals = [float(state.t.numpy())]
        for step_name, kind, _ in cols:
            s = step_lookup[step_name]
            try:
                r = _compute_residual(s, state)
                mask = _resolve_mask(state, s.mask_spec)
                arr = r.numpy()
                if mask is not None:
                    arr = arr[mask.numpy() > 0.5]
                else:
                    arr = arr.reshape(-1)
                arr = arr[np.isfinite(arr)]      # drop NaN/Inf cells
                if arr.size == 0:
                    vals.append(float("nan"))
                elif kind == "rmse":
                    vals.append(float(np.sqrt(np.mean(arr ** 2))))
                else:
                    vals.append(float(np.mean(np.abs(arr))))
            except Exception:
                vals.append(float("nan"))
        with open(path, "a") as f:
            f.write(",".join(f"{v:.6g}" for v in vals) + "\n")
        msg = f"[time_relaxation] t={vals[0]:7.2f}  " + "  ".join(
            f"{c}={v:.4g}" for (_, _, c), v in zip(cols, vals[1:])
        )
        print(msg)

    return log


# ===================================================================== #
#  Forward loop                                                         #
# ===================================================================== #

def _run_loop(cfg, p, state, forward_mod, pre_modules, post_modules, steps):
    """Inner relaxation loop.

    Order each iteration (matches legacy time_relaxation):

      1. pre_processes update    (e.g. effective_pressure — produces fields
                                  consumed by the forward model)
      2. forward_model.update    (iceflow)
      3. advance time            (CFL + save-time alignment of state.dt)
      4. inversion steps         (residual-driven control updates;
                                  e.g. surface_match writes smb, friction
                                  writes slidingco, Frank/van Pelt writes
                                  thk/topg/usurf directly)
      5. post_processes update   (e.g. thk — mass conservation that consumes
                                  the smb just written by step 4 and uses
                                  the post-advance state.dt)
      6. snapshot                (output hooks + misfit CSV if saveresult)
    """
    t_start, t_end = float(p.time.start), float(p.time.end)
    step_max = float(p.time.step)
    cfl = float(getattr(p.time, "cfl", 0.3))

    save_times = _build_save_times(t_start, t_end, float(p.time.save))

    state.t = tf.Variable(t_start, dtype=tf.float32)
    state.dt = tf.Variable(step_max, dtype=tf.float32)
    state.saveresult = False
    state.itsave = -1

    # Initialise step clocks at t_start so the first cadenced firing occurs
    # only after one full cadence has elapsed (matches legacy `tlast_fr` init).
    for s in steps:
        s.last_applied_time = tf.Variable(t_start, dtype=tf.float32)

    # CFL-limit dt iff any step OR post_module evolves geometry.
    use_cfl = any(s.control_field in ("thk", "topg", "usurf") for s in steps) \
              or any(m.__name__.endswith(".thk") for m in post_modules)

    output_hooks = _collect_output_hooks(cfg)
    misfit_log = _build_misfit_logger(getattr(p, "outputs", None), steps)

    i = 0
    while True:
        state.it = i

        # 1. pre-forward modules (need to set fields consumed by forward)
        for m in pre_modules:
            m.update(cfg, state)

        # 2. forward dynamics
        forward_mod.update(cfg, state)

        # 3. derived fields (velsurf_mag, divflux) before residuals reference them
        _ensure_derived(state)

        # 4. advance time (sets save-aligned dt)
        _advance_time(state, save_times, step_max, cfl, use_cfl)

        # 5. apply steps (write controls — smb, slidingco, geometry)
        residual_cache: Dict[str, tf.Tensor] = {}
        t_now = float(state.t.numpy())
        dt_now = float(state.dt.numpy())
        for s in steps:
            if _step_due(s, t_now):
                _apply_step(s, state, dt_now, residual_cache)

        # 6. post-step modules (e.g. thk consumes the smb just written)
        for m in post_modules:
            m.update(cfg, state)

        # 7. snapshot
        if state.saveresult:
            for hook in output_hooks:
                hook(cfg, state)
            if misfit_log is not None:
                misfit_log(state)

        if t_now >= t_end - 1e-6:
            break
        i += 1


# ===================================================================== #
#  Public API                                                           #
# ===================================================================== #

def initialize(cfg, state):
    p = cfg.processes.time_relaxation

    forward_mod = importlib.import_module(f"igm.processes.{p.forward_model}")
    pre_modules = [importlib.import_module(f"igm.processes.{n}")
                   for n in (p.get("pre_processes", []) or [])]
    post_modules = [importlib.import_module(f"igm.processes.{n}")
                    for n in (p.get("post_processes", []) or [])]

    # Idempotent re-init: forward + aux modules are typically also listed in
    # /processes (so Hydra loads their configs and IGM main initialises them
    # once). Re-calling initialize here keeps the legacy contract that the
    # user can list them in any order in /processes — these implementations
    # are expected to be safe to initialise twice.
    forward_mod.initialize(cfg, state)
    for m in pre_modules + post_modules:
        m.initialize(cfg, state)

    steps = _build_steps(p.steps or [])

    _snapshot_observations(state)
    _ensure_derived(state)

    _run_loop(cfg, p, state, forward_mod, pre_modules, post_modules, steps)

    _cleanup_loop_state(state)


def update(cfg, state):
    """The relaxation work happened in initialize(); this signals the outer
    IGM loop to exit. We can't simply ``del state.t`` (that crashes any other
    module in /processes that reads state.t — e.g. smb_simple), so we keep
    state.t and set state.continue_run = False instead.
    """
    state.continue_run = False
    state.saveresult = False


def finalize(cfg, state):
    pass


# ===================================================================== #
#  One-shot helpers                                                     #
# ===================================================================== #

def _snapshot_observations(state):
    """If standard observations are absent, snapshot the current model state.
    Light-touch convenience so that residuals like ``usurf_obs - usurf``,
    ``amb - divflux`` etc. work out-of-the-box; never overwrites existing obs.
    """
    if hasattr(state, "usurf") and not hasattr(state, "usurf_obs"):
        state.usurf_obs = tf.identity(state.usurf)
    if hasattr(state, "thk") and not hasattr(state, "thk_obs"):
        state.thk_obs = tf.identity(state.thk)
    if hasattr(state, "dhdt") and not hasattr(state, "dhdt_obs"):
        # Snapshot before any module overwrites state.dhdt as a diagnostic.
        state.dhdt_obs = tf.identity(state.dhdt)
    if (hasattr(state, "uvelsurfobs") and hasattr(state, "vvelsurfobs")
            and not hasattr(state, "velsurf_magobs")):
        state.velsurf_magobs = tf.sqrt(
            state.uvelsurfobs ** 2 + state.vvelsurfobs ** 2
        )


def _cleanup_loop_state(state):
    """Prepare for the single outer-IGM iteration that follows.

    We do **not** delete state.t — other modules in /processes (e.g.
    smb_simple) read it in their update() and would crash. Instead:
      - state.dt is zeroed so any aux module's update() is a no-op
        (their integrators see dt = 0 and write nothing).
      - state.saveresult = False so output modules don't double-write.
      - state.continue_run = False is set in time_relaxation.update()
        once the outer loop calls it, so the IGM main loop exits after
        one harmless pass over all modules.
    """
    if hasattr(state, "dt"):
        try:
            state.dt.assign(0.0)
        except Exception:
            state.dt = tf.Variable(0.0, dtype=tf.float32)
    state.saveresult = False
