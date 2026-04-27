#!/usr/bin/env python3

"""
Data assimilation by time relaxation.

A forward relaxation loop is run *inside* ``initialize``. At each
step the surface mass balance (SMB) is adapted to fit a chosen
target, while one geometric field (bed ``topg`` or thickness
``thk``) is used as the control variable. An independent friction
inversion can adjust ``slidingco`` to match observed surface
velocities. After the loop, ``state.t`` is cleared so the outer
IGM loop exits without further iterations.

Two orthogonal switches define the geometric inversion:

* ``cost.target``   : ``amb_balance`` | ``surface_match``
* ``control.field`` : ``topg``        | ``thk``

Plus an independent friction inversion (cost = ||v - v_obs||,
control = ``slidingco``) controlled by ``friction.method``
(``additive`` | ``log_mult`` | ``none``).

The combo (``cost.target=amb_balance``, ``control.field=topg``)
corresponds to the method published in:

  Ward van Pelt and Thomas Frank, *The Cryosphere* 19, 1, 2025.
  https://tc.copernicus.org/articles/19/1/2025/

This routine derives from the original implementation by Thomas Frank.

"""

import importlib
import warnings

import numpy as np
import tensorflow as tf
from scipy import ndimage as nd
from scipy.ndimage import gaussian_filter

from igm.utils.grad.compute_divflux import compute_divflux
from igm.processes.iceflow import initialize as iceflow_initialize
from igm.processes.iceflow import update as iceflow_update


# --------------------------------------------------------------------- #
#  Public API                                                           #
# --------------------------------------------------------------------- #

def initialize(cfg, state):
    p = cfg.processes.time_relaxation

    _validate_combo(p)

    _prepare_geometry(state)

    aux_modules = _load_aux_modules(p.aux_processes)
    for m in aux_modules:
        m.initialize(cfg, state)

    iceflow_initialize(cfg, state)

    if p.control.field == "thk":
        from igm.processes import thk as thk_module
        thk_module.initialize(cfg, state)
    else:
        thk_module = None

    _prepare_targets(state)
    _prepare_cost(p, state)

    _run_forward_loop(cfg, p, state, aux_modules, thk_module)

    if p.output.save_result_in_ncdf:
        _write_final_ncdf(p, state)

    _cleanup_loop_state(state)


def update(cfg, state):
    pass


def finalize(cfg, state):
    pass


# --------------------------------------------------------------------- #
#  Initialization helpers                                               #
# --------------------------------------------------------------------- #

_VALID_COSTS = ("amb_balance", "surface_match")
_VALID_CONTROLS = ("topg", "thk")


def _validate_combo(p):
    if p.cost.target not in _VALID_COSTS:
        raise ValueError(
            f"time_relaxation cost.target must be one of {_VALID_COSTS}, "
            f"got {p.cost.target!r}."
        )
    if p.control.field not in _VALID_CONTROLS:
        raise ValueError(
            f"time_relaxation control.field must be one of {_VALID_CONTROLS}, "
            f"got {p.control.field!r}."
        )
    if p.cost.target == "surface_match" and p.control.field == "topg":
        raise ValueError(
            "time_relaxation: (cost.target=surface_match, control.field=topg) "
            "has no canonical kernel. Use control.field='thk' (svalbard-style "
            "surface-nudge mass conservation) or cost.target='amb_balance'."
        )


def _load_aux_modules(names):
    return [importlib.import_module(f"igm.processes.{n}") for n in names]


def _prepare_geometry(state):
    """Make sure (thk, topg, usurf, icemask) are mutually consistent."""
    if not hasattr(state, "icemask"):
        state.icemask = tf.ones_like(state.usurf)

    if not hasattr(state, "thk"):
        if not hasattr(state, "topg"):
            state.thk = tf.zeros_like(state.icemask)
            state.topg = state.usurf
        else:
            state.thk = state.usurf - state.topg
    elif not hasattr(state, "topg"):
        state.topg = state.usurf - state.thk * state.icemask
    else:
        state.thk = state.thk * state.icemask
        state.topg = state.usurf - state.thk

    state.usurf = tf.maximum(state.usurf, 0.0)
    state.icemask = tf.where(state.usurf <= 0.0, 0.0, state.icemask)


def _prepare_targets(state):
    """Snapshot observations BEFORE the loop overwrites the model fields,
    so per-iteration misfits can be computed against the original
    targets."""
    if not hasattr(state, "usurf_obs"):
        state.usurf_obs = tf.identity(state.usurf)
    if not hasattr(state, "thk_obs"):
        state.thk_obs = tf.identity(state.thk)
    if not hasattr(state, "dhdt_obs") and hasattr(state, "dhdt"):
        state.dhdt_obs = tf.identity(state.dhdt)
    if hasattr(state, "uvelsurfobs") and hasattr(state, "vvelsurfobs"):
        if not hasattr(state, "velsurf_magobs"):
            state.velsurf_magobs = tf.sqrt(
                state.uvelsurfobs ** 2 + state.vvelsurfobs ** 2
            )


def _prepare_cost(p, state):
    """Build state.amb (and optionally state.mask_buffer) for amb_balance,
    or initialize the smb / throttle clock for surface_match."""
    target = p.cost.target

    if target == "amb_balance":
        a = p.cost.amb_balance
        if not hasattr(state, "smb"):
            arr = list(a.smb_simple_array)
            if len(arr) <= 1:
                raise ValueError(
                    "time_relaxation cost.target='amb_balance' requires either "
                    "state.smb to be available or a non-empty "
                    "cost.amb_balance.smb_simple_array."
                )
            smbpar = np.array(arr[1:], dtype=np.float32)
            smb = state.usurf - smbpar[:, 3]
            smb *= tf.where(tf.less(smb, 0.0), smbpar[:, 1], smbpar[:, 2])
            state.smb = tf.clip_by_value(smb, -100.0, smbpar[:, 4])
        if not hasattr(state, "dhdt"):
            state.dhdt = tf.zeros_like(state.smb)

        state.amb = (state.smb - state.dhdt) * state.icemask

        net_amb = float(tf.reduce_sum(tf.abs(state.amb)).numpy())
        if net_amb > 0.0:
            warnings.warn(
                f"[time_relaxation] |sum(amb)| = {net_amb:.3e} is non-zero. "
                "Apparent mass-balance imbalance violates steady-state "
                "for non-calving glaciers."
            )

        if int(a.mask_buffer) > 0:
            _create_buffer_with_smb(int(a.mask_buffer), state)
        else:
            state.mask_buffer = tf.zeros_like(state.icemask)

    elif target == "surface_match":
        if not hasattr(state, "smb"):
            state.smb = tf.zeros_like(state.usurf)
        state.amb = tf.zeros_like(state.usurf)
        state.mask_buffer = tf.zeros_like(state.icemask)
        state.tlast_mb = tf.Variable(-1.0e5, dtype=tf.float32)


# --------------------------------------------------------------------- #
#  Forward time loop                                                    #
# --------------------------------------------------------------------- #

def _run_forward_loop(cfg, p, state, aux_modules, thk_module):
    """Forward time-stepping loop.

    With control.field=thk we use CFL-adaptive dt (mirrors
    igm.processes.time.update). With control.field=topg the kernel is
    not a true mass-conservation step and dt is fixed.
    """
    t_start = float(p.time.start)
    t_end = float(p.time.end)
    step_max = float(p.time.step)
    t_save_int = float(p.time.save)
    cfl = float(getattr(p.time, "cfl", 0.3))
    use_cfl = (p.control.field == "thk")

    save_times = []
    k = 0
    while True:
        ts = t_start + k * t_save_int
        if ts > t_end + 1e-6:
            break
        save_times.append(round(ts, 6))
        k += 1
    if not save_times or save_times[-1] < t_end - 1e-6:
        save_times.append(round(t_end, 6))

    state.t = tf.Variable(t_start, dtype=tf.float32)
    state.dt = tf.Variable(step_max, dtype=tf.float32)
    state.saveresult = False
    state.itsave = -1

    fric_method = p.friction.method
    fric_active = fric_method != "none" and float(p.friction.t_fr_update) > 0.0
    if fric_active:
        if not hasattr(state, "velsurf_magobs"):
            raise RuntimeError(
                "time_relaxation friction nudge active but no "
                "velsurf_magobs / (uvelsurfobs, vvelsurfobs) provided."
            )
        state.tlast_fr = tf.Variable(t_start, dtype=tf.float32)

    output_hooks = _collect_output_hooks(cfg)
    misfit_path = str(p.output.misfits_csv) if p.output.misfits_csv else ""
    if misfit_path:
        _misfits_init(misfit_path)

    i = 0
    while True:
        state.it = i

        # Step 1: cost-driven SMB refresh (surface_match only).
        if p.cost.target == "surface_match":
            _refresh_surface_match_smb(p, state)
            state.amb = state.smb  # diagnostic alias

        # Step 2: auxiliary processes (e.g. effective_pressure).
        for m in aux_modules:
            m.update(cfg, state)

        # Step 3: iceflow.
        iceflow_update(cfg, state)

        # Step 4: advance time (mirrors igm.processes.time.update).
        _advance_time(state, save_times, step_max, cfl, use_cfl)

        # Step 5: control-field update.
        if p.control.field == "topg":
            _apply_topg_control(p, state, i, t_end)
        elif p.control.field == "thk":
            if p.cost.target == "amb_balance":
                state.smb = state.amb  # feed apparent MB into mass conservation
            thk_module.update(cfg, state)
            if not hasattr(state, "divflux"):
                state.divflux = compute_divflux(
                    state.ubar, state.vbar, state.thk, state.dx, state.dx
                )
            state.dhdt = state.dt * (state.amb - state.divflux)

        # Step 6: friction nudge.
        if (i > 0 and fric_active
                and float(state.t.numpy()) >= float(p.friction.t_fr_start)):
            if (state.t - state.tlast_fr) >= float(p.friction.t_fr_update):
                if fric_method == "additive":
                    _update_friction_additive(p, state)
                elif fric_method == "log_mult":
                    _update_friction_log_mult(p, state)
                else:
                    raise ValueError(
                        f"time_relaxation friction.method must be "
                        f"'additive', 'log_mult' or 'none', "
                        f"got {fric_method!r}."
                    )
                state.tlast_fr.assign(state.t)

        # Step 7: snapshot.
        if state.saveresult:
            for hook in output_hooks:
                hook(cfg, state)
            if misfit_path:
                _misfits_log(state, misfit_path)

        if float(state.t.numpy()) >= t_end - 1e-6:
            break
        i += 1


def _refresh_surface_match_smb(p, state):
    s = p.cost.surface_match
    update_freq = float(s.update_freq)
    if (state.t - state.tlast_mb) < update_freq:
        return
    raw = float(s.alpha) * (state.usurf_obs - state.usurf)
    smb = tf.clip_by_value(raw, float(s.smb_min), float(s.smb_max))
    if hasattr(state, "icemask"):
        smb = tf.where(state.icemask > 0.5, smb, float(s.out_of_mask_smb))
    state.smb = smb
    state.tlast_mb.assign(state.t)


def _advance_time(state, save_times, step_max, cfl, use_cfl):
    """Replaces igm.processes.time.update inside the relaxation loop."""
    if use_cfl:
        velomax = tf.maximum(
            tf.reduce_max(tf.abs(state.ubar)),
            tf.reduce_max(tf.abs(state.vbar)),
        )
        if float(velomax.numpy()) > 0.0:
            dt_target = tf.minimum(
                cfl * state.dx / velomax, tf.constant(step_max, dtype=tf.float32)
            )
            dt_target = float(dt_target.numpy())
        else:
            dt_target = step_max
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


# --------------------------------------------------------------------- #
#  Control-field kernels                                                #
# --------------------------------------------------------------------- #

def _apply_topg_control(p, state, i, t_end):
    """Aletsch-style bed inversion: thk and topg move; usurf nearly fixed.

    Defined only for cost.target='amb_balance'. The kernel integrates
    F = (amb - divflux) into thk (and into topg via the usurf-fixed
    constraint), so at steady state divflux = amb, equivalently
    dhdt = dhdt_obs.
    """
    g = p.control.topg
    divflux = compute_divflux(
        state.ubar, state.vbar, state.thk, state.dx, state.dx
    )
    state.divflux = divflux
    state.dhdt = state.dt * (state.amb - divflux)

    if i > 0:
        beta = float(g.beta)
        theta = float(g.theta)
        state.thk = tf.minimum(
            tf.maximum(state.thk + state.dhdt * beta * state.icemask, 0.0),
            float(g.max_thk),
        )
        state.topg = tf.where(
            (state.icemask == 1) & (state.usurf > 0.0),
            state.usurf - state.thk,
            state.topg,
        )
        state.usurf = tf.maximum(
            tf.maximum(0.0, state.topg),
            state.usurf
            + state.dhdt * theta * beta * state.icemask
            * (1.0 - state.mask_buffer),
        )
        state.thk = tf.where(
            (state.icemask == 1) & (state.usurf > 0.0),
            state.usurf - state.topg,
            0.0,
        )
        state.usurf = tf.where(
            (state.icemask == 0) & (state.topg < 0.0), 0.0, state.usurf
        )

        if (p.cost.target == "amb_balance"
                and bool(p.cost.amb_balance.crop_to_original)
                and round(float(state.t.numpy()), 6) == round(t_end, 6)):
            original_mask = state.icemask - state.mask_buffer
            state.thk = state.thk * original_mask
            state.topg = tf.where(original_mask == 1, state.topg, state.usurf)
            state.icemask = original_mask


# --------------------------------------------------------------------- #
#  Friction kernels                                                     #
# --------------------------------------------------------------------- #

def _update_friction_additive(p, state):
    f = p.friction
    velsurf = tf.sqrt(state.uvelsurf ** 2 + state.vvelsurf ** 2)
    vel_mismatch = tf.clip_by_value(
        (velsurf - state.velsurf_magobs) / state.velsurf_magobs,
        -float(f.additive.max_vel_ratio),
        float(f.additive.max_vel_ratio),
    )
    vel_mismatch = tf.where(
        tf.math.is_nan(vel_mismatch)
        | (state.velsurf_magobs < 1.0)
        | (state.icemask == 0),
        0.0,
        vel_mismatch,
    )
    state.slidingco = tf.clip_by_value(
        state.slidingco * (1.0 + vel_mismatch),
        float(f.slidingco_min),
        float(f.slidingco_max),
    )


def _update_friction_log_mult(p, state):
    f = p.friction
    lm = f.log_mult
    velsurf = tf.sqrt(state.uvelsurf ** 2 + state.vvelsurf ** 2)
    gamma = float(lm.gamma)
    exponent = float(lm.sliding_exponent)
    max_log_step = float(lm.max_log_step)
    sigma = float(lm.smoothing_sigma)

    safe_vobs = tf.maximum(state.velsurf_magobs, 1.0)
    safe_v = tf.maximum(velsurf, 1.0)
    log_ratio = tf.math.log(safe_v / safe_vobs)

    valid = tf.math.is_finite(log_ratio) & (state.velsurf_magobs >= 1.0)
    if hasattr(state, "icemask"):
        valid = valid & (state.icemask > 0.5)
    log_ratio = tf.where(valid, log_ratio, 0.0)

    if sigma > 0.0:
        lr_np = log_ratio.numpy()
        w_np = valid.numpy().astype(np.float32)
        num = gaussian_filter(lr_np * w_np, sigma=sigma, mode="nearest")
        den = gaussian_filter(w_np, sigma=sigma, mode="nearest")
        lr_sm = np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0)
        lr_sm = np.where(w_np > 0.5, lr_sm, 0.0).astype(np.float32)
        log_ratio = tf.convert_to_tensor(lr_sm)

    log_update = gamma * exponent * log_ratio
    log_update = tf.clip_by_value(log_update, -max_log_step, max_log_step)

    state.slidingco = tf.clip_by_value(
        state.slidingco * tf.exp(log_update),
        float(f.slidingco_min),
        float(f.slidingco_max),
    )


# --------------------------------------------------------------------- #
#  Mask buffer (amb extrapolation outside the original icemask)         #
# --------------------------------------------------------------------- #

def _create_buffer_with_smb(buffer_width, state):
    state.mask_buffer = tf.convert_to_tensor(
        _internal_buffer(buffer_width, 1 - state.icemask.numpy()),
        dtype=tf.float32,
    )
    try:
        usurf_np = state.usurf.numpy()
        amb_np = state.amb.numpy()
        neg = amb_np < 0
        amb_slope, amb_intercept = np.polyfit(usurf_np[neg], amb_np[neg], deg=1)
        amb_fit = amb_intercept + amb_slope * state.usurf
        state.amb = tf.where(
            (amb_fit < 0.0) & (state.mask_buffer == 1), amb_fit, state.amb
        )
    except Exception:
        print(
            "[time_relaxation] amb extrapolation in the buffer failed; "
            "setting negative amb in buffer to 0 instead."
        )
        state.amb = tf.where(
            (state.amb < 0.0) & (state.mask_buffer == 1), 0.0, state.amb
        )
    state.icemask = state.icemask + state.mask_buffer


def _internal_buffer(bw, mask):
    mask_iter = mask == 1
    mask_bw = ~mask_iter
    k = np.ones((3, 3), dtype=int)
    for _ in range(bw):
        boundary = nd.binary_dilation(mask_bw, k) & ~mask_bw
        mask_bw = mask_bw | boundary
    return (mask_bw.astype(np.int32) + mask_iter.astype(np.int32)) - 1


# --------------------------------------------------------------------- #
#  Output hooks + diagnostics                                           #
# --------------------------------------------------------------------- #

def _collect_output_hooks(cfg):
    hooks = []
    if not hasattr(cfg, "outputs"):
        return hooks
    outputs_cfg = cfg.outputs
    if hasattr(outputs_cfg, "write_ncdf"):
        from igm.outputs import write_ncdf
        hooks.append(write_ncdf.run)
    if hasattr(outputs_cfg, "write_ts"):
        from igm.outputs import write_ts
        hooks.append(write_ts.run)
    if hasattr(outputs_cfg, "write_vtp"):
        from igm.outputs import write_vtp
        hooks.append(write_vtp.run)
    return hooks


def _misfits_init(path):
    with open(path, "w") as f:
        f.write("t,rmse_divflux_minus_amb,mean_abs_divflux_minus_amb,"
                "rmse_vel,mean_abs_vel_err,n_vel_obs,"
                "rmse_thk,mean_abs_thk_err,n_thk_obs,"
                "rmse_usurf,mean_abs_usurf_err\n")


def _misfits_log(state, path):
    mask = state.icemask.numpy() > 0.5

    resid = (state.divflux.numpy() - state.amb.numpy())[mask]
    rmse_df = float(np.sqrt(np.mean(resid ** 2))) if resid.size else float("nan")
    mae_df = float(np.mean(np.abs(resid))) if resid.size else float("nan")

    if hasattr(state, "velsurf_magobs"):
        vmod = tf.sqrt(state.uvelsurf ** 2 + state.vvelsurf ** 2).numpy()
        vobs = state.velsurf_magobs.numpy()
        sel_v = mask & np.isfinite(vobs) & (vobs > 0.0)
        n_v = int(sel_v.sum())
        if n_v:
            dv = vmod[sel_v] - vobs[sel_v]
            rmse_v = float(np.sqrt(np.mean(dv ** 2)))
            mae_v = float(np.mean(np.abs(dv)))
        else:
            rmse_v = mae_v = float("nan")
    else:
        rmse_v = mae_v = float("nan")
        n_v = 0

    if hasattr(state, "thkobs"):
        thk = state.thk.numpy()
        tobs = state.thkobs.numpy()
        sel_t = np.isfinite(tobs) & (tobs >= 0.0)
        n_t = int(sel_t.sum())
        if n_t:
            dh = thk[sel_t] - tobs[sel_t]
            rmse_t = float(np.sqrt(np.mean(dh ** 2)))
            mae_t = float(np.mean(np.abs(dh)))
        else:
            rmse_t = mae_t = float("nan")
    else:
        rmse_t = mae_t = float("nan")
        n_t = 0

    if hasattr(state, "usurf_obs"):
        ds = (state.usurf.numpy() - state.usurf_obs.numpy())[mask]
        rmse_s = float(np.sqrt(np.mean(ds ** 2))) if ds.size else float("nan")
        mae_s = float(np.mean(np.abs(ds))) if ds.size else float("nan")
    else:
        rmse_s = mae_s = float("nan")

    t = float(state.t.numpy())
    with open(path, "a") as f:
        f.write(
            f"{t:.3f},{rmse_df:.6g},{mae_df:.6g},"
            f"{rmse_v:.6g},{mae_v:.6g},{n_v},"
            f"{rmse_t:.6g},{mae_t:.6g},{n_t},"
            f"{rmse_s:.6g},{mae_s:.6g}\n"
        )

    print(
        f"[time_relaxation] t={t:7.2f}  "
        f"RMSE(divflux-amb)={rmse_df:8.3f} m/yr  "
        f"RMSE(vel)={rmse_v:8.2f} m/yr (n={n_v})  "
        f"RMSE(thk)={rmse_t:8.2f} m (n={n_t})  "
        f"RMSE(usurf)={rmse_s:7.2f} m"
    )


def _write_final_ncdf(p, state):
    from netCDF4 import Dataset

    nc = Dataset(p.output.save_result_in_ncdf, "w", format="NETCDF4")
    nc.createDimension("x", state.x.shape[0])
    nc.createDimension("y", state.y.shape[0])
    xv = nc.createVariable("x", "f4", ("x",))
    xv[:] = state.x.numpy() if hasattr(state.x, "numpy") else np.asarray(state.x)
    yv = nc.createVariable("y", "f4", ("y",))
    yv[:] = state.y.numpy() if hasattr(state.y, "numpy") else np.asarray(state.y)
    for name in p.output.vars_to_save:
        if not hasattr(state, name):
            continue
        val = getattr(state, name)
        arr = val.numpy() if hasattr(val, "numpy") else np.asarray(val)
        if arr.ndim != 2:
            continue
        v = nc.createVariable(name, "f4", ("y", "x"))
        v[:] = arr
    nc.close()


# --------------------------------------------------------------------- #
#  Outer-loop short-circuit                                             #
# --------------------------------------------------------------------- #

def _cleanup_loop_state(state):
    """Short-circuit the outer IGM update loop.

    update_modules() exits when state.t is missing; we delete it.
    state.dt is kept and zeroed because thk / cf_sub_grid still get
    one update() call from the outer loop on the way out, and rely on
    state.dt for the advection step (dt=0 → no-op).
    """
    if hasattr(state, "t"):
        delattr(state, "t")
    if hasattr(state, "dt"):
        try:
            state.dt.assign(0.0)
        except Exception:
            state.dt = tf.Variable(0.0, dtype=tf.float32)
    state.saveresult = False
