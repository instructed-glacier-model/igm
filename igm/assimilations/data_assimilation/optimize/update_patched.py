#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""
Patch-wise (out-of-core) variant of optimize_update for grids too large to
fit a full-grid gradient tape on the GPU.

Mirrors optimize/update.py step by step, but the tape runs tile by tile:
for each window (core + halo) we slice the state, evaluate the iceflow model
and the cost on the window, take the gradient w.r.t. the window controls and
keep only the core part (see optimize/patching.py for why this reconstructs
the full-grid gradient when the halo exceeds the model receptive field).
The Adam step is then applied once on the full-grid control variables, so
the optimizer trajectory matches the full-grid version up to the per-window
normalization of the mean-based cost terms.

Global (non-tile-separable) quantities are handled by the driver:
  - areaicemask (regu_thk convexity term) and the ELA (regu_thk_v2) are
    computed on the full grid and passed via state attributes;
  - the divfluxfcz linear regression (state.res) is refitted on the full
    grid every 10 iterations from a tape-free patched forward sweep; the
    tiles receive i=-1 so they never refit it on a patch.
Cost terms that are genuinely global remain unsupported here (volume_init,
cook.infer_params) and raise at the first iteration.
"""

import numpy as np
import tensorflow as tf
from scipy import stats

from igm.utils.grad.compute_divflux import compute_divflux
from ..cost_terms.total_cost import total_cost
from ..iceflow_dispatch import iceflow_evaluate
from ..utils import compute_flow_direction_for_anisotropic_smoothing_vel
from ..utils import compute_flow_direction_for_anisotropic_smoothing_usurf
from .patching import plan_windows, slice_state

# velocity fields written by the iceflow evaluators, stitched back core by
# core so that printers/plots/rms diagnostics see full-grid fields
_VEL_KEYS_2D = ("uvelbase", "vvelbase", "uvelsurf", "vvelsurf", "ubar", "vbar")
_VEL_KEYS_3D = ("U", "V")  # buffers kept on CPU (Nz x ny x nx can be large)


def _check_supported(cfg):
    da = cfg.assimilations.data_assimilation
    if "volume_init" in da.cost_list:
        raise ValueError(
            "data_assimilation patching: the 'volume_init' cost is a global "
            "(non tile-separable) term and is not supported with "
            "optimization.patch_size > 0. Set patch_size: 0 or drop it."
        )
    if da.cook.infer_params:
        raise ValueError(
            "data_assimilation patching: cook.infer_params relies on global "
            "per-basin volumes and is not supported with "
            "optimization.patch_size > 0."
        )
    if cfg.processes.iceflow.method.lower() == "emulated" \
            and cfg.processes.iceflow.emulator.network.multiple_window_size > 0:
        raise ValueError(
            "data_assimilation patching: emulator.network.multiple_window_size "
            "> 0 pads to the full-grid shape (state.PAD) and is incompatible "
            "with per-window inference. Set it to 0."
        )


def _get_ctx(state, ny, nx, P, H):
    ctx = getattr(state, "_da_patch", None)
    if ctx is None:
        ctx = {"windows": plan_windows(ny, nx, P, H), "grad_bufs": {}, "vel_bufs": {}}
        state._da_patch = ctx
    return ctx


def _stitch_velocities(state, pstate, ctx, ny, nx, gy, gx, ly, lx):
    bufs = ctx["vel_bufs"]
    for key in _VEL_KEYS_2D + _VEL_KEYS_3D:
        val = getattr(pstate, key, None)
        if val is None:
            continue
        buf = bufs.get(key)
        if buf is None:
            if key in _VEL_KEYS_3D:
                with tf.device("/CPU:0"):
                    buf = tf.Variable(
                        tf.zeros(list(val.shape[:-2]) + [ny, nx], dtype=val.dtype),
                        trainable=False,
                    )
            else:
                buf = tf.Variable(tf.zeros([ny, nx], dtype=val.dtype), trainable=False)
            bufs[key] = buf
        if key in _VEL_KEYS_3D:
            buf[:, gy, gx].assign(val[:, ly, lx])
        else:
            buf[gy, gx].assign(val[ly, lx])
        setattr(state, key, buf)


def _evaluate_patched(cfg, state, ctx, ny, nx):
    """Tape-free full-grid forward pass, window by window (velocities are
    stitched into state; used to refresh global diagnostics)."""
    for (wy0, wy1, cy0, cy1), (wx0, wx1, cx0, cx1) in ctx["windows"]:
        pstate = slice_state(state, ny, nx, wy0, wy1, wx0, wx1)
        if not _tile_is_active(pstate):
            continue
        iceflow_evaluate(cfg, pstate)
        _stitch_velocities(
            state, pstate, ctx, ny, nx,
            slice(cy0, cy1), slice(cx0, cx1),
            slice(cy0 - wy0, cy1 - wy0), slice(cx0 - wx0, cx1 - wx0),
        )


def _tile_is_active(pstate):
    """Tiles with no observed ice mask and no ice carry (numerically) no
    gradient information; skipping them saves most of the compute on
    large domains where the ice occupies a fraction of the grid."""
    return bool(tf.reduce_max(pstate.icemaskobs) > 0.5) \
        or bool(tf.reduce_max(pstate.thk) > 0.0)


def optimize_update_patched(cfg, state, cost, i):

    da = cfg.assimilations.data_assimilation
    opt = da.optimization
    ny, nx = state.thk.shape
    P, H = int(opt.patch_size), int(opt.patch_halo)

    ctx = _get_ctx(state, ny, nx, P, H)

    if i == 0:
        _check_supported(cfg)
        nwin = len(ctx["windows"])
        print(
            f"[data_assimilation] patch-wise inversion: grid {ny}x{nx}, "
            f"{nwin} windows of {min(ny, P + 2 * H)}x{min(nx, P + 2 * H)} "
            f"(core {P}, halo {H})"
        )

    if opt.step_size_decay < 1:
        state.optimizer.lr = opt.step_size * (opt.step_size_decay ** (i / 100))

    # ---- scaled control variables (same as optimize/update.py) ----

    sc = {}
    sc["thk"] = da.scaling.thk
    sc["usurf"] = da.scaling.usurf
    sc[state.da_friction] = da.scaling[state.da_friction]
    sc["arrhenius"] = da.scaling.arrhenius

    log_fric = da.fitting.log_slidingco

    for f in da.control_list:
        if log_fric & (f == state.da_friction):
            new_value = tf.sqrt(getattr(state, f) / sc[f])
        else:
            new_value = getattr(state, f) / sc[f]
        key = f + "_sc"
        existing = getattr(state, key, None)
        if isinstance(existing, tf.Variable) and existing.shape == new_value.shape:
            existing.assign(new_value)
        else:
            setattr(state, key, tf.Variable(new_value))

    # ---- global (non tile-separable) quantities, computed tape-free ----

    if not hasattr(state, "areaicemask"):
        # consumed by regu_thk_v1 (convexity term); the patch-local value would be wrong
        state.areaicemask = tf.reduce_sum(
            tf.where(state.icemask > 0.5, 1.0, 0.0)
        ) * state.dx**2

    if da.regularization.thk_version == 2 \
            and da.regularization.abl_acc_balance != 1:
        # consumed by regu_thk_v2; percentile of the full-grid surface
        state.ELA = np.percentile(
            state.usurf[state.usurf > 0], 66.7, method="linear"
        )

    if "divfluxfcz" in da.cost_list and i % 10 == 0:
        # refresh full-grid velocities and refit the global regression, as
        # the full-grid code does every 10 iterations
        _evaluate_patched(cfg, state, ctx, ny, nx)
        divflux = compute_divflux(
            state.ubar, state.vbar, state.thk, state.dx, state.dx,
            method=da.divflux.method,
            smooth_sigma=da.divflux.smooth_sigma,
        )
        ACT = state.icemaskobs > 0.5
        state.res = stats.linregress(state.usurf[ACT], divflux[ACT])

    # tiles must never refit the regression on a patch: i=-1 disables the
    # (i % 10 == 0) branch in cost_divfluxfcz, which then reuses state.res
    i_tile = -1 if "divfluxfcz" in da.cost_list else i

    # ---- per-tile tapes: assemble the full-grid gradient core by core ----

    grad_bufs = ctx["grad_bufs"]
    for f in da.control_list:
        v = getattr(state, f + "_sc")
        b = grad_bufs.get(f)
        if isinstance(b, tf.Variable) and b.shape == v.shape:
            b.assign(tf.zeros_like(v))
        else:
            grad_bufs[f] = tf.Variable(tf.zeros_like(v), trainable=False)

    anis = da.regularization.smooth_anisotropy_factor != 1

    for (wy0, wy1, cy0, cy1), (wx0, wx1, cx0, cx1) in ctx["windows"]:

        pstate = slice_state(state, ny, nx, wy0, wy1, wx0, wx1)
        if not _tile_is_active(pstate):
            continue

        with tf.GradientTape() as t:

            pvars = []
            for f in da.control_list:
                p = tf.identity(getattr(state, f + "_sc")[wy0:wy1, wx0:wx1])
                t.watch(p)
                pvars.append(p)
                setattr(pstate, f + "_sc", p)
                if log_fric & (f == state.da_friction):
                    setattr(pstate, f, (p**2) * sc[f])
                else:
                    setattr(pstate, f, p * sc[f])

            iceflow_evaluate(cfg, pstate)

            if anis:
                if da.regularization.smooth_anisotropy_var == "vel":
                    compute_flow_direction_for_anisotropic_smoothing_vel(pstate)
                elif da.regularization.smooth_anisotropy_var == "usurf":
                    compute_flow_direction_for_anisotropic_smoothing_usurf(pstate)

            tile_cost = {}
            total_cost(cfg, pstate, tile_cost, i_tile)

            # a mean over an empty selection (e.g. no velocity obs in this
            # window) is nan on a patch while it is well-defined globally:
            # drop such terms, they carry no gradient
            finite = {
                k: v for k, v in tile_cost.items()
                if bool(tf.reduce_all(tf.math.is_finite(v)))
            }
            cost_total = tf.add_n(list(finite.values())) if finite else tf.constant(0.0)

        grads = t.gradient(cost_total, pvars)

        gy, gx = slice(cy0, cy1), slice(cx0, cx1)
        ly, lx = slice(cy0 - wy0, cy1 - wy0), slice(cx0 - wx0, cx1 - wx0)

        for f, g in zip(da.control_list, grads):
            if g is not None:
                grad_bufs[f][gy, gx].assign(g[ly, lx])

        _stitch_velocities(state, pstate, ctx, ny, nx, gy, gx, ly, lx)

    # ---- exact full-grid costs for monitoring (tape-free) ----
    # the cost terms are cheap local operations (no network evaluation), so
    # they can be evaluated on the full grid from the stitched velocities;
    # this makes the reported values identical to the full-grid version
    if anis:
        if da.regularization.smooth_anisotropy_var == "vel":
            compute_flow_direction_for_anisotropic_smoothing_vel(state)
        elif da.regularization.smooth_anisotropy_var == "usurf":
            compute_flow_direction_for_anisotropic_smoothing_usurf(state)
    total_cost(cfg, state, cost, i_tile)

    # ---- masking, descent step, write-back (same as optimize/update.py) ----

    var_to_opti = [getattr(state, f + "_sc") for f in da.control_list]

    grads = []
    for f in da.control_list:
        g = tf.convert_to_tensor(grad_bufs[f])
        if opt.sole_mask:
            if not state.da_friction == f:
                g = tf.where(state.icemaskobs > 0.5, g, 0.0)
            else:
                g = tf.where(state.icemaskobs == 1, g, 0.0)
        else:
            if not state.da_friction == f:
                g = tf.where(state.icemaskobs > 0.5, g, 0.0)
        grads.append(g)

    state.optimizer.apply_gradients(zip(grads, var_to_opti))

    for f in da.control_list:
        if log_fric & (f == state.da_friction):
            setattr(state, f, (getattr(state, f + "_sc") ** 2) * sc[f])
        else:
            setattr(state, f, getattr(state, f + "_sc") * sc[f])

    if "reproject" in opt.obstacle_constraint:

        if "icemask" in da.cost_list:
            state.thk = tf.where(state.icemaskobs > 0.5, state.thk, 0)

        if "thk" in da.control_list:
            state.thk = tf.where(state.thk < 0, 0, state.thk)

        fric = state.da_friction
        if fric in da.control_list:
            setattr(state, fric, tf.where(getattr(state, fric) < 0, 0, getattr(state, fric)))

        if "arrhenius" in da.control_list:
            # Here we assume a minimum value of 1.0 for the arrhenius factor (should not be hard-coded)
            state.arrhenius = tf.where(state.arrhenius < 1.0, 1.0, state.arrhenius)

    state.divflux = compute_divflux(
        state.ubar,
        state.vbar,
        state.thk,
        state.dx,
        state.dx,
        method=da.divflux.method,
        smooth_sigma=da.divflux.smooth_sigma,
    )
