#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""
SMB Inference process for IGM.

Performs gradient-based inversion to recover surface mass balance (SMB)
parameters from glacier thickness observations. Uses a differentiable
glacier dynamics simulation with a CNN or PINN ice flow emulator.

This process is analogous to data_assimilation but targets climate/SMB
parameters rather than ice dynamics parameters (thickness, sliding, etc.).

Like data_assimilation, the full optimization loop runs within initialize().
"""

import os

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .core.glacier import GlacierDynamicsCheckpointed
from .core.inversion import _eval_pair
from .core.lbfgs import lbfgs_minimize
from .core.smb import update_smb_profile
from .data.loader import load_observations_from_nc, load_smb_point_obs
from .data.homogenize import (
    prepare_stake_overlay, draw_anchor_overlay, draw_raw_stake_overlay)
from .visualization.plots import plot_loss_components
from .utils.emulator_tools import compute_divflux


# ─── Helper functions ────────────────────────────────────────────────────────


def _get_dx_dy(cfg, state):
    """Grid spacing from the state coordinates, falling back to the config.

    The fallback only fires for a state without x/y coordinate vectors, which
    no IGM input module produces; it exists so a hand-built state stays usable.
    """
    fallback = cfg.assimilations.smb_inference.physics

    if hasattr(state, "x") and state.x is not None and len(state.x) > 1:
        dx = float(state.x[1] - state.x[0])
    else:
        dx = float(fallback.dx)

    if hasattr(state, "y") and state.y is not None and len(state.y) > 1:
        dy = float(state.y[1] - state.y[0])
    else:
        dy = float(fallback.dy)

    return dx, dy


def _piecewise_linear_extrap(z, alts, vals):
    """Piecewise-linear interpolation with linear extrapolation at both tails.

    `alts` must be sorted ascending with at least 2 points. Within the observed
    range this matches np.interp; outside it extrapolates using the slope of
    the two nearest stakes at each tail.
    """
    out = np.interp(z, alts, vals).astype(np.float32)
    slope_lo = (vals[1] - vals[0]) / (alts[1] - alts[0] + 1e-12)
    slope_hi = (vals[-1] - vals[-2]) / (alts[-1] - alts[-2] + 1e-12)
    lo = z < alts[0]
    hi = z > alts[-1]
    out[lo] = vals[0] + slope_lo * (z[lo] - alts[0])
    out[hi] = vals[-1] + slope_hi * (z[hi] - alts[-1])
    return out


def _load_observation(smb_cfg, state, topg):
    """Load the observation target for inversion."""
    obs_cfg = smb_cfg.observations

    if getattr(obs_cfg, "geology_file", ""):
        obs_years = list(obs_cfg.observation_years)
        obs = load_observations_from_nc(obs_cfg.geology_file, topg, obs_years)
        target_key = obs_years[-1]
        observation = obs[target_key]
        print(
            f"[smb_inference] Loaded {len(obs)} observations from "
            f"{obs_cfg.geology_file}, target: {target_key}"
        )
        return observation, obs

    # Use a state variable as observation target
    source = getattr(obs_cfg, "state_variable", "thk")
    if hasattr(state, source) and getattr(state, source) is not None:
        raw = tf.cast(getattr(state, source), tf.float32)

        # If the observation is a surface elevation, convert to thickness
        if source in ("usurfinfer", "usurfobs", "usurf"):
            observation = raw - topg
            print(
                f"[smb_inference] Using state.{source} as observation target "
                f"(converted to thickness via {source} - topg)"
            )
        else:
            observation = raw
            print(f"[smb_inference] Using state.{source} as observation target")

        return observation, None
    else:
        raise ValueError(
            f"[smb_inference] No observation data found. "
            f"state.{source} is not available. "
            f"Set observations.geology_file or ensure state.{source} exists."
        )


def _run_profile_inversion(cfg, smb_cfg, glacier_model, observation, topg, H_init, ice_mask, outdir=None):
    """
    Optimize a 1D elevation-dependent SMB profile to match observed thickness.

    Returns
    -------
    H_sim : tf.Tensor
        Simulated ice thickness at end of optimization.
    smb_vec : tf.Variable
        Optimized 1D SMB elevation profile.
    z_min : tf.Tensor
        Minimum elevation of the profile.
    dz : tf.Tensor
        Elevation bin spacing.
    loss_history : list
        Total loss per iteration.
    data_history : list
        Data-fidelity term per iteration.
    """
    inv_cfg = smb_cfg.inversion
    opt_cfg = smb_cfg.optimization

    # Elevation range for profile discretization. Restrict to on-glacier
    # pixels — off-glacier terrain (outside ice_mask) can extend well above
    # or below the glacier and would otherwise inflate N_bins with bins
    # that never see any ice.
    Z_surf = topg + H_init
    on_ice = ice_mask > 0.5
    Z_on = tf.boolean_mask(Z_surf, on_ice)
    z_min = tf.reduce_min(Z_on)
    z_max = tf.reduce_max(Z_on)
    dz = tf.constant(float(inv_cfg.dz), dtype=tf.float32)
    N_bins = int(tf.math.ceil((z_max - z_min) / dz).numpy()) + 1

    # Initial SMB profile (m ice eq./yr).
    # If init_from_obs=true and a stake CSV is configured, initialize from the
    # observed profile (piecewise-linear between stakes, linear extrapolation
    # at both tails). Otherwise fall back to linspace(smb_init_low, high).
    z_axis_init = float(z_min) + float(dz) * np.arange(N_bins)
    init_from_obs = bool(getattr(inv_cfg, "init_from_obs", False))
    smb_points_file_init = getattr(smb_cfg.observations, "smb_points_file", "") or ""

    init_values = None
    if init_from_obs and smb_points_file_init and os.path.exists(smb_points_file_init):
        try:
            alts_init, smbs_init = load_smb_point_obs(
                smb_points_file_init, to_ice_eq=True
            )
            if alts_init.size >= 2:
                order = np.argsort(alts_init)
                alts_s = alts_init[order]
                smbs_s = smbs_init[order]
                init_values = _piecewise_linear_extrap(z_axis_init, alts_s, smbs_s)
                print(
                    f"[smb_inference] Initialized smb_vec from {alts_s.size} stakes "
                    f"in {smb_points_file_init} (piecewise + linear extrapolation)"
                )
            else:
                print(
                    f"[smb_inference] init_from_obs requested but only "
                    f"{alts_init.size} stake(s) loaded; falling back to linspace."
                )
        except Exception as e:
            print(f"[smb_inference] init_from_obs failed ({e}); falling back to linspace.")

    if init_values is None:
        init_values = np.linspace(
            float(inv_cfg.smb_init_low), float(inv_cfg.smb_init_high), N_bins,
            dtype=np.float32,
        )

    smb_vec = tf.Variable(
        tf.constant(init_values, dtype=tf.float32),
        trainable=True,
        dtype=tf.float32,
    )

    reg_lambda = float(opt_cfg.regularisation)
    optimizer_name = str(getattr(opt_cfg, "optimizer", "adam")).lower()

    print(
        f"[smb_inference] Profile inversion: {N_bins} elevation bins, "
        f"z_min={float(z_min):.0f}, z_max={float(z_max):.0f}, dz={float(dz):.0f}"
        f", smb min={float(tf.reduce_min(smb_vec).numpy()):.2f}, "
        f"smb max={float(tf.reduce_max(smb_vec).numpy()):.2f}"
        f", optimizer={optimizer_name}"
    )

    log_freq = max(1, opt_cfg.nbitmax // 50) # Log ~50 times during optimization

    smb_vec_dir = None
    if outdir is not None:
        smb_vec_dir = os.path.join(outdir, "smb_vec_iters")
        os.makedirs(smb_vec_dir, exist_ok=True)

    # Stake SMB observations (m w.e./yr, native CSV units) for overlay on the
    # per-iter profile plots. The model's smb_vec is m ice eq./yr (forced by
    # the dynamics in thk.py), so we plot smb_vec * rho_ice/rho_w to express
    # the displayed profile in the same w.e. convention as the stakes and
    # Kneib et al. (2024). Display-only — the trainable variable, the loss,
    # and state.smb stay in m ice eq./yr.
    rho_ice = float(cfg.processes.iceflow.physics.ice_density)
    rho_w = 1000.0
    ice_to_we = rho_ice / rho_w  # ≈ 0.917

    # Group stakes by horizontal position once (Kneib-2024 style): each
    # qualifying cell is one stake observed across many years; its multi-year
    # mean is directly comparable to the inferred multi-year profile.
    stake_prep = None
    smb_points_file = getattr(smb_cfg.observations, "smb_points_file", "") or ""
    if smb_points_file and os.path.exists(smb_points_file):
        try:
            stake_prep = prepare_stake_overlay(smb_points_file)
            if stake_prep is not None:
                print(f"[smb_inference] {stake_prep['clu_alt'].size} stake "
                      f"bins (≥{stake_prep['cluster_min_years']} yr) over "
                      f"{stake_prep['period']} from {smb_points_file}")
        except Exception as e:
            print(f"[smb_inference] Could not load {smb_points_file}: {e}")

    def _save_profile_plot(smb_values, iter_idx):
        if smb_vec_dir is None:
            return
        if (iter_idx - 1) % 5 != 0:
            return
        smb_we = smb_values * ice_to_we
        z_axis = float(z_min) + float(dz) * np.arange(smb_values.shape[0])
        fig, ax = plt.subplots(figsize=(5, 8))
        ax.plot(smb_we, z_axis, "-o", markersize=3, label="profile")
        if stake_prep is not None:
            # Multi-year stakes (Year column) -> repeat-cell anchors; single-
            # snapshot stakes (no Year / no clusters, e.g. Argentiere) -> raw
            # points + profile-vs-stakes RMSE.
            if stake_prep["clu_alt"].size:
                draw_anchor_overlay(ax, stake_prep, z_axis, smb_we)
            else:
                draw_raw_stake_overlay(ax, stake_prep, z_axis, smb_we)
            ax.legend(loc="best", fontsize=8)
        ax.axvline(0.0, color="k", linewidth=0.5)
        ax.set_xlabel("SMB (m w.e./yr)")
        ax.set_ylabel("Elevation (m)")
        ax.set_title(f"SMB profile — iter {iter_idx}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(smb_vec_dir, f"smb_vec_iter_{iter_idx:04d}.png"), dpi=100)
        plt.close(fig)

    def _compute_loss(x):
        """Run the forward model and return (total_loss, data_term, H)."""
        H = glacier_model(
            precip_tensor=None,
            T_m_lowest=None,
            T_s=None,
            melt_factor=None,
            smb_method="profile",
            smb_vec=x,
            z_min=z_min,
            dz=dz,
            retrain_mode=False,
        )
        metrics = _eval_pair(H, observation)
        data_term = metrics["rmse"]
        if reg_lambda > 0 and x.shape[0] > 2:
            second_deriv = x[:-2] - 2.0 * x[1:-1] + x[2:]
            smoothness = tf.reduce_sum(second_deriv ** 2)
            total = data_term + reg_lambda * smoothness
        else:
            total = data_term
        return total, data_term, H

    loss_history = []
    data_history = []
    H_sim = None

    if optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=opt_cfg.learning_rate)
        best_loss = float("inf")
        no_improve_count = 0

        for i in range(opt_cfg.nbitmax):
            with tf.GradientTape() as tape:
                loss, data_term, H_sim = _compute_loss(smb_vec)
            grads = tape.gradient(loss, [smb_vec])
            optimizer.apply_gradients(zip(grads, [smb_vec]))

            loss_val = float(loss.numpy())
            data_val = float(data_term.numpy())
            loss_history.append(loss_val)
            data_history.append(data_val)

            if i % log_freq == 0 or i == opt_cfg.nbitmax - 1:
                _save_profile_plot(smb_vec.numpy(), i + 1)
                print(
                    f"[smb_inference] Iter {i + 1}/{opt_cfg.nbitmax}: "
                    f"loss={loss_val:.5f}, data={data_val:.5f}"
                )

            if i > 10 and opt_cfg.early_stop_patience > 0:
                if best_loss - loss_val > opt_cfg.early_stop_threshold:
                    best_loss = loss_val
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                if no_improve_count >= opt_cfg.early_stop_patience:
                    print(
                        f"[smb_inference] Early stopping at iteration {i + 1} "
                        f"(no improvement for {opt_cfg.early_stop_patience} iters)"
                    )
                    break

                if loss_val <= getattr(opt_cfg, "loss_threshold", 1.0e-4):
                    print(f"[smb_inference] Loss threshold reached at iteration {i + 1}")
                    break

    elif optimizer_name == "lbfgs":
        # lbfgs_minimize runs its own loop and evaluates the objective inside
        # IGM's tf.function line search, so the closure it is given must stay
        # free of Python side effects. History, logs and per-iteration profile
        # plots are captured from the callback instead, which runs eagerly once
        # per accepted step.
        last_H = {"H": None}
        max_iters = int(opt_cfg.nbitmax)
        tolerance = float(getattr(opt_cfg, "lbfgs_tolerance", 1.0e-6))
        x_tolerance = float(getattr(opt_cfg, "lbfgs_x_tolerance", 0.0))
        f_rel_tol_default = (
            float(opt_cfg.early_stop_threshold)
            if float(opt_cfg.early_stop_threshold) > 0 else 0.0
        )
        f_rel_tolerance = float(
            getattr(opt_cfg, "lbfgs_f_relative_tolerance", f_rel_tol_default)
        )
        memory = int(getattr(opt_cfg, "lbfgs_memory", 10))

        # The forward run is the expensive part, so the terms needed only for
        # reporting are cached from the last evaluation rather than recomputed.
        # lbfgs_minimize evaluates the accepted point last, so the cache is
        # already at the position the callback is handed; it falls back to a
        # recomputation if that ever stops holding.
        last_eval = {"x": None, "data": None, "H": None}

        def _value_and_gradient(x):
            with tf.GradientTape() as tape:
                tape.watch(x)
                total, data_term, H = _compute_loss(x)
            last_eval.update(x=x.numpy().copy(), data=data_term, H=H)
            return total, tape.gradient(total, x)

        def _on_iteration(i, x, total):
            """Report an accepted step. Runs eagerly, outside the line search."""
            xa = x.numpy()
            if last_eval["x"] is not None and np.array_equal(last_eval["x"], xa):
                data_term, H = last_eval["data"], last_eval["H"]
                loss_val = float(total.numpy())
            else:
                total_v, data_term, H = _compute_loss(x)
                loss_val = float(total_v.numpy())
            data_val = float(data_term.numpy())
            loss_history.append(loss_val)
            data_history.append(data_val)
            last_H["H"] = H
            if i % log_freq == 0 or i == max_iters - 1:
                _save_profile_plot(x.numpy(), i + 1)
                print(
                    f"[smb_inference] L-BFGS iter {i + 1}: "
                    f"loss={loss_val:.5f}, data={data_val:.5f}"
                )

        initial_position = tf.identity(smb_vec)
        results = lbfgs_minimize(
            _value_and_gradient,
            initial_position=initial_position,
            max_iterations=max_iters,
            num_correction_pairs=memory,
            tolerance=tolerance,
            x_tolerance=x_tolerance,
            f_relative_tolerance=f_rel_tolerance,
            line_search_method=str(
                getattr(opt_cfg, "lbfgs_line_search", "armijo")
            ),
            callback=_on_iteration,
        )

        smb_vec.assign(results.position)
        final_loss, _, H_sim = _compute_loss(smb_vec)
        print(
            f"[smb_inference] L-BFGS converged={bool(results.converged.numpy())}, "
            f"failed={bool(results.failed.numpy())}, "
            f"iterations={int(results.num_iterations.numpy())}, "
            f"evaluations={int(results.num_objective_evaluations.numpy())}, "
            f"final_loss={float(final_loss.numpy()):.5f}"
        )
        if H_sim is None:
            H_sim = last_H["H"]

    else:
        raise ValueError(
            f"[smb_inference] Unknown optimizer '{optimizer_name}'. "
            "Supported: 'adam', 'lbfgs'."
        )

    # Post-optimization: replay forward with retrain_mode=True to capture
    # (H, ubar, vbar) at t_start and at ttot, then plot flux divergence.
    if outdir is not None:
        try:
            glacier_model(
                precip_tensor=None,
                T_m_lowest=None,
                T_s=None,
                melt_factor=None,
                smb_method="profile",
                smb_vec=smb_vec,
                z_min=z_min,
                dz=dz,
                retrain_mode=True,
            )
            _save_divflux_comparison(glacier_model, outdir)
        except Exception as e:
            print(f"[smb_inference] divflux snapshot plot failed: {e}")

    return H_sim, smb_vec, z_min, dz, loss_history, data_history


def _save_divflux_comparison(glacier_model, outdir):
    """Plot flux divergence at the first (t_start) and last (ttot) time steps
    of the post-optimization replay, plus their residual.

    The first two panels share the same colour scale/colorbar.
    The residual panel has its own colourbar.
    """
    snap_start = getattr(glacier_model, "snapshot_start", None)
    snap_end = getattr(glacier_model, "snapshot_end", None)
    if snap_start is None or snap_end is None:
        print("[smb_inference] No snapshots captured; skipping divflux plot.")
        return

    dx = float(glacier_model.dx)
    dy = float(glacier_model.dy)

    div_start = compute_divflux(
        snap_start["ubar"], snap_start["vbar"], snap_start["H"], dx, dy
    ).numpy()

    div_end = compute_divflux(
        snap_end["ubar"], snap_end["vbar"], snap_end["H"], dx, dy
    ).numpy()

    # Residual
    residual = div_end - div_start

    H_start = snap_start["H"].numpy()
    H_end = snap_end["H"].numpy()

    div_start_m = np.where(H_start > 0, div_start, np.nan)
    div_end_m = np.where(H_end > 0, div_end, np.nan)

    # Mask residual where either snapshot has no ice
    mask = (H_start > 0) & (H_end > 0)
    residual_m = np.where(mask, residual, np.nan)

    # Shared color scale for first two panels
    vmax = float(np.nanmax(np.abs(div_start_m)))
    if not np.isfinite(vmax) or vmax == 0.0:
        vmax = 1.0

    # Independent color scale for residual
    vmax_res = float(np.nanmax(np.abs(residual_m)))
    if not np.isfinite(vmax_res) or vmax_res == 0.0:
        vmax_res = 1.0

    # RMSE of residual
    residual_rmse = np.sqrt(np.nanmean(residual_m**2))

    ny, nx = H_start.shape
    extent = [0, nx * dx / 1000, 0, ny * dy / 1000]

    t_start_label = int(round(snap_start["time"]))
    t_end_label = int(round(snap_end["time"]))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=120)

    # --- Start ---
    im0 = axes[0].imshow(
        div_start_m,
        cmap="RdBu_r",
        origin="lower",
        extent=extent,
        vmin=-vmax,
        vmax=vmax,
    )
    axes[0].set_title(f"Flux divergence — {t_start_label}")
    axes[0].set_xlabel("x (km)")
    axes[0].set_ylabel("y (km)")

    # --- End ---
    axes[1].imshow(
        div_end_m,
        cmap="RdBu_r",
        origin="lower",
        extent=extent,
        vmin=-vmax,
        vmax=vmax,
    )
    axes[1].set_title(f"Flux divergence — {t_end_label}")
    axes[1].set_xlabel("x (km)")
    axes[1].set_ylabel("y (km)")

    # Shared colorbar for first two panels
    cbar0 = fig.colorbar(im0, ax=axes[:2], fraction=0.046, pad=0.04)
    cbar0.set_label("div(u·H)  (m/yr)")

    # --- Residual ---
    im_res = axes[2].imshow(
        residual_m,
        cmap="RdBu_r",
        origin="lower",
        extent=extent,
        vmin=-vmax_res,
        vmax=vmax_res,
    )
    axes[2].set_title(f"Residual RMSE = {residual_rmse:.2f}")
    axes[2].set_xlabel("x (km)")
    axes[2].set_ylabel("y (km)")

    # Independent residual colorbar
    cbar_res = fig.colorbar(im_res, ax=axes[2], fraction=0.046, pad=0.04)
    cbar_res.set_label("Residual  (m/yr)")

    out_path = os.path.join(
        outdir,
        f"divflux_{t_start_label}_vs_{t_end_label}.png"
    )

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[smb_inference] Saved divflux comparison to {out_path}")

# ─── IGM process interface ───────────────────────────────────────────────────

from igm.processes.iceflow import initialize as iceflow_initialize

def initialize(cfg, state):
    """
    Initialize and run SMB inference (inversion).

    Like data_assimilation, the full optimization loop runs within
    initialize(). The optimized SMB parameters and resulting glacier
    state are stored back into the IGM state object.
    """
    iceflow_initialize(cfg, state) # initialize the iceflow model

    smb_cfg = cfg.assimilations.smb_inference

    # ── 1. Read data from IGM state ──────────────────────────────────────
    topg = tf.cast(state.topg, tf.float32)

    if hasattr(state, "icemask") and state.icemask is not None:
        ice_mask = tf.cast(state.icemask, tf.float32)
    else:
        ice_mask = tf.ones_like(topg)

    if hasattr(state, "thk") and state.thk is not None:
        H_init = tf.cast(state.thk, tf.float32)
    else:
        H_init = tf.zeros_like(topg)

    # Optional: re-anchor the forward run to a surface other than the one the
    # DA was performed on. Used by the "DA at t2" setup: the DA inverts the
    # thickness against the END-of-period surface — the epoch the observed
    # velocities actually sample — which fixes the bed as topg = usurf(t2) -
    # thk_DA. The forward run must still start at t_start, so H_init is rebuilt
    # from the t1 DEM on that same bed. The target is unchanged (usurfinfer -
    # topg, i.e. the DA thickness at t2), so the run measures exactly the SMB
    # needed to carry the t1 geometry onto the observed t2 surface.
    usurf_var = str(getattr(
        getattr(smb_cfg, "initial_state", None), "usurf_variable", "") or "")
    if usurf_var:
        if not hasattr(state, usurf_var) or getattr(state, usurf_var) is None:
            raise ValueError(
                f"[smb_inference] initial_state.usurf_variable='{usurf_var}' "
                f"but state.{usurf_var} is not available — add it to the input "
                f"NetCDF (it must be NaN-free; it feeds the emulator)."
            )
        on_ice = tf.cast(ice_mask > 0.5, tf.float32)
        n_on_ice = tf.maximum(tf.reduce_sum(on_ice), 1.0)
        thk_da = H_init  # the DA geometry, for the log line below
        # The DA drives thk to 0 outside the icemask; keep the re-anchored
        # geometry consistent instead of growing ice on off-glacier terrain.
        H_init = tf.maximum(
            tf.cast(getattr(state, usurf_var), tf.float32) - topg, 0.0) * on_ice
        print(
            f"[smb_inference] Forward run re-anchored to state.{usurf_var}: "
            f"H_init = max({usurf_var} - topg, 0) * icemask. On-ice mean "
            f"thickness {float(tf.reduce_sum(H_init) / n_on_ice):.1f} m "
            f"(DA state was {float(tf.reduce_sum(thk_da * on_ice) / n_on_ice):.1f} m)"
        )

    dx, dy = _get_dx_dy(cfg, state)
    print(f"[smb_inference] Grid: {topg.shape}, dx={dx}, dy={dy}")

    # ── 2. Load ice flow emulator (fallback if not on state) ────────────
    has_state_model = hasattr(state, 'iceflow_model') and state.iceflow_model is not None
    if has_state_model:
        print("[smb_inference] Emulator inherited from state.iceflow_model")
        # Detect input_fields from the iceflow config (pretrained CNN manifest)
        iceflow_cfg = cfg.processes.iceflow
        input_fields = list(getattr(iceflow_cfg.unified, 'inputs', []))
        if input_fields:
            print(f"[smb_inference] Using pretrained CNN with inputs: {input_fields}")
        else:
            input_fields = None
        model, V_bar, Nz = None, None, None
    else:
        print("[smb_inference] Emulator not found on state")
        model, V_bar, Nz, input_fields = None, None, None, None

    # ── 3. Build glacier dynamics model ──────────────────────────────────
    glacier_model = GlacierDynamicsCheckpointed(
        Z_topo=topg,
        H_init=H_init,
        ice_mask=ice_mask,
        cfg=cfg,
        dx=dx,
        dy=dy,
        model=model,
        V_bar=V_bar,
        Nz=Nz,
        input_fields=input_fields,
        state=state,
    )

    # ── 4. Load observations ─────────────────────────────────────────────
    observation, all_obs = _load_observation(smb_cfg, state, topg)

    # ── 5. Run inversion ─────────────────────────────────────────────────
    method = smb_cfg.inversion.method

    if method == "profile":
        outdir = smb_cfg.output.outdir
        os.makedirs(outdir, exist_ok=True)
        H_sim, smb_vec, z_min, dz, loss_history, data_history = (
            _run_profile_inversion(
                cfg, smb_cfg, glacier_model, observation, topg, H_init, ice_mask,
                outdir=outdir,
            )
        )

        # Store profile-specific results
        state.smb_vec = smb_vec
        state.smb_z_min = z_min
        state.smb_dz = dz

        # Compute 2D SMB field from optimized profile
        Z_surf_final = topg + H_sim
        state.smb = update_smb_profile(Z_surf_final, smb_vec, z_min, dz) * ice_mask

    else:
        raise ValueError(
            f"[smb_inference] Unknown inversion method: '{method}'. "
            "Supported: 'profile'"
        )

    # ── 6. Update state ──────────────────────────────────────────────────
    state.thk = H_sim
    state.usurf = topg + H_sim
    state.smb_inference_loss_history = loss_history
    state.smb_inference_data_history = data_history

    # ── 7. Output ────────────────────────────────────────────────────────
    outdir = smb_cfg.output.outdir
    os.makedirs(outdir, exist_ok=True)

    if smb_cfg.output.plot_loss:
        try:
            plot_loss_components(loss_history, data_history, outdir)
        except Exception as e:
            print(f"[smb_inference] plot_loss_components failed: {e}")

    if smb_cfg.output.save_results:

        np.save(os.path.join(outdir, "thk_optimized.npy"), H_sim.numpy())
        np.save(os.path.join(outdir, "thk_observed.npy"), observation.numpy())
        np.save(os.path.join(outdir, "loss_history.npy"), np.array(loss_history))
        # data-fidelity term (surface RMSE) per iter — the SMB L-curve reads
        # this final value as the data-misfit axis (vs profile roughness).
        np.save(os.path.join(outdir, "data_history.npy"), np.array(data_history))
        if hasattr(state, "smb_vec"):
            np.save(os.path.join(outdir, "smb_vec.npy"), state.smb_vec.numpy())
        if hasattr(state, "smb_z_min"):
            np.save(os.path.join(outdir, "z_min.npy"), np.array(float(state.smb_z_min)))
        if hasattr(state, "smb_dz"):
            np.save(os.path.join(outdir, "dz.npy"), np.array(float(state.smb_dz)))
        if hasattr(state, "smb") and state.smb is not None:
            np.save(os.path.join(outdir, "smb_field.npy"), state.smb.numpy())
            np.save(os.path.join(outdir, "smb_profile.npy"), smb_vec.numpy())

        # Mass-balance diagnostic fields (dh/dt + div(flux) = smb), on the
        # cropped model grid so plot_smb_results.py can bin them by elevation.
        # The flux divergence is NOT recomputed here — the continuity estimate
        # uses div(flux) straight from the DA output (geology-optimized.nc).
        #   • modeled surface — elevation axis for the binning
        #   • Hugonnet dh/dt as loaded onto the state from the input .nc
        np.save(os.path.join(outdir, "usurf_modeled.npy"), (topg + H_sim).numpy())

        if hasattr(state, "dhdt") and state.dhdt is not None:
            np.save(
                os.path.join(outdir, "dhdt_obs.npy"),
                tf.cast(state.dhdt, tf.float32).numpy(),
            )
        else:
            print("[smb_inference] state.dhdt not available; skipping dhdt_obs.npy")

        print(f"[smb_inference] Results saved to {outdir}")

    print(f"[smb_inference] Optimization complete. Final loss: {loss_history[-1]:.5f}")


def update(cfg, state):
    pass


def finalize(cfg, state):
    pass
