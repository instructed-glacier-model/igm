#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""
Adaptive patch selection for the unified iceflow solver.

PIPELINE (one call to `select_patches`):

  Once early — compute the fixed-shape budget:
      bs      = max(1, floor(framesizemax^2 / (ly · lx)))
      N_train = floor(grid_cover_count / bs) · bs          (multiple of bs)

  STEP 1 — WINDOW GENERATION
      list_windows = generator(state, cfg_ap)
      Available generators:
        - "regular_grid"     non-overlapping tiles, exact coverage
        - "peak_augmented"   regular_grid + n_extra_peaks windows at score peaks (DEFAULT)

  STEP 2 — SCORING (with the thk eligibility filter)
      score[i] = per-cell squared change of the input field between the previous
                 and current scoring snapshot, aggregated to window i (the
                 "temporal_variance" proxy). Captures "this region is locally
                 evolving". Bootstrap on the first call falls back to |dh/dt| max.

      eligibility[i] = True iff max(thk in window i) >= min_thk_in_window.
                       (Fallback: if zero windows pass, all are eligible — used
                        to avoid pathological "sample window 0 repeatedly" at
                        ice-free t=0 of bootstrap simulations.)

  STEP 3 — TRAINING WINDOWS
      training_windows = length-N_train int array of indices into list_windows.
        - selection == "all"        → deterministic cyclic tiling of the
                                        eligible window indices, length N_train.
        - selection == "scored"     → N_train iid draws with replacement,
                                        p[i] ∝ (score[i] + eps)^score_alpha
                                        restricted to eligible windows.
      If shuffle_training_windows (default true), the resulting indices are shuffled.
      If record_training_windows, one JSONL line is appended to record_path.

  STEP 4 — SOLVER LOOP   (lives in solver.py, NOT here)
      The solver receives the [N_train, ly, lx, C] training tensor together with bs
      and runs `ceil(N_train / bs) = N_train // bs` calls to optimizer.minimize(),
      each on a fixed-shape [bs, ly, lx, C] slice. Constant batch shape → no XLA
      recompile, no tail batch.

KEY CONFIG KNOBS
  (see igm/conf/processes/iceflow.yaml for the full schema and defaults)

      data_preparation.framesizemax     GPU capacity per batch (= one tile fits at bs=1)
      adaptive_patching.patch_size      Actual tile side. 0 → fall back to framesizemax.
      adaptive_patching.windows         "regular_grid" | "peak_augmented"
      adaptive_patching.selection       "all" | "scored"
      adaptive_patching.min_thk_in_window  (m) Drop windows below this thk → ineligible.
      adaptive_patching.score_alpha     Continuous-weighting exponent (default 1.0).
      adaptive_patching.temporal_downsample  Downsampling factor for the per-cell
                                          temporal_variance signal (default 1 = full).
      adaptive_patching.n_extra_peaks   peak_augmented-only; -1 = auto (= #grid tiles).
      adaptive_patching.shuffle_training_windows  bool (default true)
      adaptive_patching.record_training_windows   bool (default false)
      adaptive_patching.record_path     JSONL output path (default training_windows.jsonl)
      adaptive_patching.rng_seed        int|null — null means fresh randomness per call
      adaptive_patching.rescore_freq    Steps 1+2 (windows + scoring) are recomputed
                                          every N retrain calls; in between, the cache
                                          is reused. Step 3 (sampling) runs every call.

HISTORY: tier-1 cleanup on 2026-05-20 removed the legacy `weighting: integer`
mode (and its `min_freq` / `freq_ratio` knobs), the `scoring: max / mean` legacy
proxies (kept only as cold-start bootstrap), and the `sliding_overlap` window
generator (with its `stride_factor` knob). Older configs may still set these
keys — they are silently ignored.

ARCHIVE: the prior multi-pass round-robin `scheduled` selector, the `top_k` and
`nms` selectors, and `_build_schedule` live in `patch_selection_archive.py`.
"""

import collections
import json
import os

import numpy as np
import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State


# ===========================================================================
# Helpers
# ===========================================================================

def _patch_grid_dims(ny: int, nx: int, framesizemax: int):
    """Return (sy, sx, ly, lx) — number of strips and patch size in each dim."""
    sy = ny // framesizemax + 1
    sx = nx // framesizemax + 1
    ly = ny // sy
    lx = nx // sx
    return sy, sx, ly, lx


def _get_dhdt(state: State) -> np.ndarray:
    """Return |dh/dt| as a numpy array [H, W], with NaN/inf sanitised to 0.

    NaN guard matters early in a run: if the network is untrained the
    predicted velocities can be wild and the resulting thickness can drift
    to NaN. Treat NaN/inf cells as "no proxy signal here" — downstream
    falls back to uniform-freq behaviour rather than crashing.
    """
    if hasattr(state, "dhdt") and state.dhdt is not None:
        a = np.abs(state.dhdt.numpy() if hasattr(state.dhdt, "numpy") else np.array(state.dhdt))
    elif hasattr(state, "_thk_prev") and state._thk_prev is not None:
        dt = float(state.dt) if hasattr(state, "dt") and state.dt > 0 else 1.0
        a = np.abs((state.thk.numpy() - state._thk_prev.numpy()) / dt)
    else:
        a = np.zeros_like(state.thk.numpy())
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


def _cache_full_field(state: State, cfg: DictConfig) -> tf.Tensor:
    """Build and cache the full [H, W, C] input field on state."""
    from igm.processes.iceflow.utils.data_preprocessing import fieldin_state_to_X
    from igm.utils.math.precision import normalize_precision
    X = fieldin_state_to_X(cfg, state)
    dtype = normalize_precision(cfg.processes.iceflow.numerics.precision)
    state._adaptive_patching_X = tf.cast(X, dtype)
    return state._adaptive_patching_X


def _current_step(state: State) -> int:
    if hasattr(state, "it"):
        it = state.it
        return int(it.numpy()) if hasattr(it, "numpy") else int(it)
    return 0


# ===========================================================================
# Layer 1 — Window generators (all return list of (y0, x0, ly, lx))
# ===========================================================================

def _windows_regular_grid(state, cfg_ap):
    """Non-overlapping regular grid. Exact coverage, zero overlap."""
    dhdt = _get_dhdt(state)
    ny, nx = dhdt.shape
    sy, sx, ly, lx = _patch_grid_dims(ny, nx, cfg_ap.framesizemax)
    windows = []
    for j in range(sy):
        for i in range(sx):
            y0 = j * ly if j < sy - 1 else (ny - ly)
            x0 = i * lx if i < sx - 1 else (nx - lx)
            windows.append((int(y0), int(x0), int(ly), int(lx)))
    return windows


def _windows_peak_augmented(state, cfg_ap):
    """Regular grid (covering base) + n_extra_peaks windows centered at peaks.

    Special value `n_extra_peaks: -1` → match the number of grid windows
    (so the output has 2N windows: N grid for coverage, N peaks for focus).
    """
    grid = _windows_regular_grid(state, cfg_ap)
    n_extra = int(cfg_ap.n_extra_peaks)
    if n_extra < 0:
        n_extra = len(grid)
    if n_extra == 0:
        return grid

    dhdt = _get_dhdt(state).copy()
    if dhdt.max() <= 0:
        return grid  # no proxy signal yet → grid only

    ny, nx = dhdt.shape
    _, _, ly, lx = _patch_grid_dims(ny, nx, cfg_ap.framesizemax)
    half_y, half_x = ly // 2, lx // 2

    extras = []
    for _ in range(n_extra):
        if dhdt.max() <= 0:
            break
        cy, cx = np.unravel_index(np.argmax(dhdt), dhdt.shape)
        y0 = int(max(0, min(ny - ly, cy - half_y)))
        x0 = int(max(0, min(nx - lx, cx - half_x)))
        extras.append((y0, x0, int(ly), int(lx)))
        dhdt[y0:y0+ly, x0:x0+lx] = 0.0

    return grid + extras


_WINDOW_GENS = {
    "regular_grid": _windows_regular_grid,
    "peak_augmented": _windows_peak_augmented,
}


# ===========================================================================
# Step 2 — Scoring + freq
# ===========================================================================

def _score_windows(state, windows,
                   X_full=None, temporal_downsample: int = 1) -> np.ndarray:
    """Score each window via the cell-level squared change of the input field
    between the previous and the current scoring call, aggregated into each
    window.

    Robust to windows being regenerated each rescore (e.g. peak_augmented
    placing extras at fresh peak locations) because the temporal signal lives
    on the (stable) grid, not on windows.

    `X_full` (the cached [H, W, C] input tensor) is required; the previous-call
    snapshot is parked on `state._prev_full_field` (a single downsampled
    [H_lr, W_lr, C] array). Bootstrap on the first call falls back to
    per-window |dh/dt| max so the cold-start case is sensible.
    """
    return _score_temporal_variance(
        state, windows, X_full, downsample=temporal_downsample,
    )


def _score_temporal_variance(state, windows, X_full, downsample=1):
    """Per-window temporal change of the input field, computed at the CELL
    level so the proxy is robust to windows being regenerated each rescore
    (e.g. peak_augmented placing the extras at fresh peak locations).

    The proxy is the per-cell **squared difference between the current and
    the previous scoring-call snapshot** of the input field, summed across
    channels and aggregated to each window. Equivalent to a 2-sample
    temporal variance (= |Δ|² / 4 up to a constant scale) — the variance
    over a longer rolling history is no longer worth the memory / compute
    overhead (run_06 showed K∈{2,5,10} indistinguishable in practice).

    Bootstrap (no previous snapshot yet) falls back to per-window |dh/dt|
    max so the cold-start case is well-defined.

    Memory: 1 × H/downsample × W/downsample × C × 4 bytes. For our typical
    grid (3010 × 4510 × 5) at downsample=1: ~270 MB. At downsample=4: ~17 MB.
    """
    if X_full is None:
        raise ValueError("temporal_variance scoring requires X_full (cached input tensor).")

    X = X_full.numpy() if hasattr(X_full, "numpy") else np.asarray(X_full)
    Xd = X[::downsample, ::downsample, :].astype(np.float32)  # [H_lr, W_lr, C]

    prev = getattr(state, "_prev_full_field", None)
    # Invalidate the cached previous snapshot if its shape changed (e.g. the
    # downsample factor was reconfigured mid-run).
    if prev is not None and prev.shape != Xd.shape:
        prev = None

    scores = np.zeros(len(windows), dtype=np.float64)

    if prev is None:
        # Bootstrap from |dh/dt| max per window so the proxy is sensible
        # before the first squared-diff is available.
        dhdt = _get_dhdt(state)
        for i, (y0, x0, ly, lx) in enumerate(windows):
            scores[i] = float(np.max(dhdt[y0:y0+ly, x0:x0+lx]))
        state._prev_full_field = Xd
        return scores

    # Per-cell squared change, summed across channels — a [H_lr, W_lr] field.
    diff_sq = ((Xd - prev) ** 2).sum(axis=-1)
    state._prev_full_field = Xd

    # Aggregate to each (possibly newly-regenerated) window by averaging the
    # downsampled cells that fall inside its (y0, x0, ly, lx) extent.
    for i, (y0, x0, ly, lx) in enumerate(windows):
        y0d = int(y0) // downsample
        y1d = int(y0 + ly) // downsample
        x0d = int(x0) // downsample
        x1d = int(x0 + lx) // downsample
        if y1d > y0d and x1d > x0d:
            sub = diff_sq[y0d:y1d, x0d:x1d]
            scores[i] = float(sub.mean())
        else:
            # Window smaller than one downsampled cell — use the nearest cell.
            scores[i] = float(diff_sq[y0d, x0d])

    return scores


def _eligibility_mask(state, windows, min_thk_in_window: float) -> np.ndarray:
    """Boolean mask: which windows have enough ice to be considered for training.

    Windows with max(thk) < min_thk_in_window are ineligible → freq=0 → never picked,
    regardless of selector. Returns all-True when min_thk_in_window <= 0, or as a
    fallback when zero windows pass the filter (e.g. INIT step of an ice-free
    bootstrap simulation) — there is nothing to filter by, so a no-op is the right
    behaviour; the alternative is sampling 'window 0' bs times and overfitting
    the init pass to a single corner of the grid.
    """
    if min_thk_in_window <= 0.0:
        return np.ones(len(windows), dtype=bool)
    thk = state.thk.numpy() if hasattr(state.thk, "numpy") else np.array(state.thk)
    mask = np.array(
        [float(np.max(thk[y0:y0+ly, x0:x0+lx])) >= min_thk_in_window
         for (y0, x0, ly, lx) in windows],
        dtype=bool,
    )
    if not mask.any():
        return np.ones(len(windows), dtype=bool)
    return mask


# ===========================================================================
# Step 3 — Build training_windows (length N_train, indices into list_windows)
# ===========================================================================

def _continuous_probs(scores: np.ndarray, eligible: np.ndarray,
                      score_alpha: float, eps: float = 1e-9) -> np.ndarray:
    """Probability vector p[i] ∝ (score[i] + eps)^score_alpha for eligible windows.

    Scale-invariant: only relative scores matter, so the same `score_alpha`
    works irrespectively of the patch size (whose mean magnitude is otherwise
    a confound for both `|dh/dt|`-based and temporal_variance scoring).

    Falls back to uniform-over-eligibles if all eligible scores are zero
    (e.g. the field is exactly stationary). All-ineligible → all-zero vector
    (the caller must handle that edge).
    """
    p = np.zeros_like(scores, dtype=np.float64)
    if not eligible.any():
        return p
    s = np.asarray(scores[eligible], dtype=np.float64)
    p[eligible] = (np.maximum(s, 0.0) + eps) ** float(score_alpha)
    total = p.sum()
    if not np.isfinite(total) or total <= 0.0:
        p[eligible] = 1.0
        total = p.sum()
    return p / total


def _build_training_windows(scores: np.ndarray, eligible: np.ndarray,
                            selection: str, score_alpha: float,
                            n_train: int, rng: np.random.Generator) -> np.ndarray:
    """Return a length-N_train int array of indices into list_windows.

    `selection == "all"`:
        Deterministic. Tile the eligible-window indices cyclically to fill
        N_train. Ineligible windows are skipped. `score_alpha` has no effect.

    `selection == "scored"`:
        N_train iid draws with replacement, with p[i] ∝ (score[i] + eps)^score_alpha
        restricted to eligible windows (continuous weighting, scale-invariant).
    """
    n = len(scores)

    if selection == "all":
        eligible_idx = np.where(eligible)[0]
        if eligible_idx.size == 0:
            return np.zeros(n_train, dtype=np.int32)
        reps = int(np.ceil(n_train / eligible_idx.size))
        out = np.tile(eligible_idx, reps)[:n_train]
        return out.astype(np.int32)

    # selection == "scored"
    p = _continuous_probs(scores, eligible, score_alpha)
    if p.sum() <= 0:
        return np.zeros(n_train, dtype=np.int32)
    return rng.choice(n, size=n_train, replace=True, p=p).astype(np.int32)


# ===========================================================================
# Patch extraction
# ===========================================================================

def _gather_patches(state, windows, indices) -> tf.Tensor:
    """Slice patches at windows[indices] from the cached full field.

    Returns a tensor of shape [N, ly, lx, C] where N = len(indices).
    """
    X = state._adaptive_patching_X
    patches = [X[w[0]:w[0]+w[2], w[1]:w[1]+w[3], :]
               for w in (windows[int(i)] for i in indices)]
    return tf.stack(patches, axis=0) if patches else tf.zeros((0,) + tuple(X.shape[1:]), dtype=X.dtype)


# ===========================================================================
# Dispatcher
# ===========================================================================

_SELECTORS = ("all", "scored")


def select_patches(cfg: DictConfig, state: State, inputs: tf.Tensor):
    """4-step adaptive patch selection.

    Returns a tuple `(training_inputs, bs)`:
      - training_inputs : tf.Tensor of shape [N_train, ly, lx, C]
      - bs              : int — batch size for the solver loop. N_train is
                          guaranteed to be a multiple of bs.

    See module docstring for the full pipeline.

    The `inputs` argument (the pre-split regular-grid tensor) is accepted but
    unused; we slice patches directly from the cached full field.
    """
    cfg_ap = cfg.processes.iceflow.unified.adaptive_patching

    def _get(key, default):
        try:
            v = cfg_ap[key]
            return default if v is None else v
        except Exception:
            return default

    class _Cfg:
        pass
    ap = _Cfg()
    ap.windows                    = str(_get("windows", "peak_augmented"))
    ap.selection                  = str(_get("selection", "scored"))
    ap.min_thk_in_window          = float(_get("min_thk_in_window", 0.0))
    ap.n_extra_peaks              = int(_get("n_extra_peaks", -1))
    ap.shuffle_training_windows   = bool(_get("shuffle_training_windows", True))
    ap.record_training_windows    = bool(_get("record_training_windows", False))
    ap.record_path                = _get("record_path", None) or "training_windows.jsonl"
    ap.rng_seed                   = _get("rng_seed", None)
    ap.rescore_freq               = int(_get("rescore_freq", 10))
    ap.score_alpha                = float(_get("score_alpha", 1.0))
    ap.temporal_downsample        = int(_get("temporal_downsample", 1))
    # NOTE — legacy knobs `weighting`, `min_freq`, `freq_ratio`, `scoring`,
    # `stride_factor`, `temporal_history_K` were removed in the 2026-05-20
    # tier-1 cleanup. They are silently ignored if still present in old configs.

    # Capacity = data_preparation.framesizemax (one tile of this size fits at bs=1).
    framemax = int(cfg.processes.iceflow.unified.data_preparation.framesizemax)
    # Actual tile size used by the window generators. When > 0 overrides framemax;
    # when 0 falls back to framemax.
    _patch_size = int(_get("patch_size", 0) or 0)
    ap.framesizemax = _patch_size if _patch_size > 0 else framemax

    if ap.windows not in _WINDOW_GENS:
        raise ValueError(
            f"unknown windows generator: {ap.windows!r}; "
            f"available: {list(_WINDOW_GENS)}"
        )
    if ap.selection not in _SELECTORS:
        raise ValueError(
            f"unknown selection: {ap.selection!r}; "
            f"available: {list(_SELECTORS)} (top_k/nms archived — see patch_selection_archive.py)"
        )

    # --- Once early: bs and N_train, computed from grid dims and framesizemax ---
    dhdt = _get_dhdt(state)
    ny, nx = dhdt.shape
    _, _, ly, lx = _patch_grid_dims(ny, nx, ap.framesizemax)
    bs = max(1, framemax ** 2 // max(1, ly * lx))
    grid_cover_count = (ny // ap.framesizemax + 1) * (nx // ap.framesizemax + 1)
    n_train = max(bs, (grid_cover_count // bs) * bs)

    rng = (np.random.default_rng(int(ap.rng_seed))
           if ap.rng_seed is not None else np.random.default_rng())

    # STAGE 2 of adaptive_unique_strategy — credit-based scoring branch.
    # When adaptive_training.enabled is true, sample patches from the OLD
    # windows (those the observer accumulated δ_w against) weighted by
    # `state._ct_cumulative_delta_w`, then REBUILD windows for the next
    # inter-retrain interval (§13c Option 1). The legacy temporal-variance
    # scoring path below is bypassed entirely.
    cfg_at = getattr(cfg.processes.iceflow.unified, "adaptive_training", None)
    use_credit_scores = (cfg_at is not None) and bool(getattr(cfg_at, "enabled", False))
    if use_credit_scores:
        cache = getattr(state, "_ap_cache", None)
        if cache is None or "list_windows" not in cache or not cache["list_windows"]:
            # Bootstrap: no prior windows → build them now, sample uniformly.
            list_windows = _WINDOW_GENS[ap.windows](state, ap)
            scores = np.zeros(len(list_windows), dtype=np.float64)
            cur_call_n = 0
        else:
            list_windows = cache["list_windows"]
            cum_tf = getattr(state, "_ct_cumulative_delta_w", None)
            # cumulative_delta_w is now a TF tensor on GPU; convert to numpy once.
            if isinstance(cum_tf, tf.Tensor):
                cum = cum_tf.numpy().astype(np.float64)
            else:
                cum = np.asarray(cum_tf if cum_tf is not None else [], dtype=np.float64)
            if cum.size == len(list_windows) and cum.sum() > 0:
                scores = cum
            else:
                # Defensive fallback (mis-aligned size, or no Δ accumulated yet).
                scores = np.zeros(len(list_windows), dtype=np.float64)
            cur_call_n = int(cache.get("call_n", 0)) + 1

        eligible = _eligibility_mask(state, list_windows, ap.min_thk_in_window)
        _cache_full_field(state, cfg)   # needed by _gather_patches below

        training_idx = _build_training_windows(
            scores, eligible, "scored", 1.0, n_train, rng,
        )
        if ap.shuffle_training_windows:
            rng.shuffle(training_idx)
        if ap.record_training_windows:
            _append_training_record(state, ap, list_windows, scores,
                                    training_idx, bs, n_train)
        training_inputs = _gather_patches(state, list_windows, training_idx)

        # Rebuild windows for the NEXT inter-retrain period (§13c Option 1).
        list_windows_new = _WINDOW_GENS[ap.windows](state, ap)
        state._ap_cache = {
            "key": (ap.windows, ap.selection, ap.min_thk_in_window,
                    ap.framesizemax, ap.n_extra_peaks, ap.score_alpha,
                    ap.temporal_downsample),
            "list_windows": list_windows_new,
            "scores": None,
            "eligible": None,
            "call_n": cur_call_n,
        }
        # The observer's cumulative_delta_w will be re-shaped at the next
        # update_credit_observer call (it allocates a fresh zeros_like array
        # when its shape mismatches the new len(list_windows)).
        state._thk_prev = tf.identity(state.thk)
        return training_inputs, bs

    # --- Steps 1+2: cached across calls, rebuilt every `rescore_freq` calls ---
    # Cache lives on state._ap_cache. It carries list_windows, scores, the
    # eligible mask, and the call-counter at last rebuild. The cache is
    # invalidated when:
    #   - it doesn't exist yet (first call)
    #   - the call counter has advanced by >= rescore_freq since last rebuild
    #   - the configured selection/thk-filter/window knobs changed
    #     (so a per-trial sweep starting fresh always rebuilds)
    #   - the bs/ly/lx changed (e.g. grid size changed mid-run)
    cache = getattr(state, "_ap_cache", None)
    cache_key = (ap.windows, ap.selection, ap.min_thk_in_window,
                 ap.framesizemax, ap.n_extra_peaks,
                 ap.score_alpha, ap.temporal_downsample)
    call_n = (cache["call_n"] + 1) if cache is not None else 0
    needs_rebuild = (
        cache is None
        or cache.get("key") != cache_key
        or (ap.rescore_freq > 0 and call_n % ap.rescore_freq == 0)
    )
    if needs_rebuild:
        # Step 1: generate list_windows
        list_windows = _WINDOW_GENS[ap.windows](state, ap)
        # Step 2: scoring. temporal_variance scoring needs the cached input
        #         field, so build it BEFORE _score_windows; it is anyway needed
        #         in Step 4.
        X_full = _cache_full_field(state, cfg)
        scores = _score_windows(state, list_windows,
                                X_full=X_full,
                                temporal_downsample=ap.temporal_downsample)
        eligible = _eligibility_mask(state, list_windows, ap.min_thk_in_window)
        cache = {
            "key": cache_key,
            "list_windows": list_windows,
            "scores": scores,
            "eligible": eligible,
            "call_n": call_n,
        }
        state._ap_cache = cache
    else:
        list_windows = cache["list_windows"]
        scores = cache["scores"]
        eligible = cache["eligible"]
        cache["call_n"] = call_n
        # Step 4 still needs X_full each call to gather the chosen patches.
        _cache_full_field(state, cfg)

    # --- Step 3: build training_windows (length n_train, indices into list_windows) ---
    training_idx = _build_training_windows(
        scores, eligible, ap.selection,
        ap.score_alpha, n_train, rng,
    )
    if ap.shuffle_training_windows:
        rng.shuffle(training_idx)

    if ap.record_training_windows:
        _append_training_record(state, ap, list_windows, scores,
                                training_idx, bs, n_train)

    # --- Step 4 prep: gather the training patches (full field already cached above) ---
    training_inputs = _gather_patches(state, list_windows, training_idx)

    # Bookkeep previous thickness so |dh/dt| is computable next call.
    state._thk_prev = tf.identity(state.thk)

    return training_inputs, bs


def _append_training_record(state, ap, list_windows, scores,
                            training_idx, bs, n_train):
    """One JSONL line per call describing the training_windows."""
    rec = {
        "step": _current_step(state),
        "t": (float(state.t.numpy()) if hasattr(state, "t") and hasattr(state.t, "numpy")
              else None),
        "selection": ap.selection,
        "bs": int(bs),
        "n_train": int(n_train),
        "n_list_windows": int(len(list_windows)),
        "windows": [list(w) for w in list_windows],
        "scores": [float(s) for s in scores],
        "training_idx": [int(i) for i in training_idx],
    }
    with open(ap.record_path, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


# ===========================================================================
# Credit-tracker observer (STAGE 1 of the adaptive_unique_strategy migration)
#
# This block is OBSERVATION-ONLY. It maintains the per-step credit accumulator
# and the per-window cumulative Δ_w on `state`, and optionally writes one JSONL
# line per simulation step describing those quantities. It does NOT modify
# `get_status` or any retrain decisions — those still come from `retrain_freq`
# until stage 2 lands. See `adaptive_unique_strategy.md` (clever-patch).
# ===========================================================================

SIGMA_REFRESH_FREQ = 10    # refresh _ct_sigma_c every N steps. Cheap (one
                           # full-grid reduce_variance per refresh). The
                           # short cadence matters early in a run when a
                           # channel transitions from variance≈0 (ice-free
                           # bootstrap, thk=0) to a non-trivial scale: the
                           # accumulator must follow that change closely or
                           # it produces hugely-inflated z-units in the
                           # interim.
VAR_FLOOR = 1e-8           # var_c < VAR_FLOOR → channel is "constant" → it
                           # is fully excluded from both σ_c estimation AND
                           # the Δ reduction. Replaces an earlier σ_c floor
                           # that mistakenly included sqrt(epsilon)=1e-3
                           # channels in the mean.


def _ct_get(cfg_unified, key, default):
    """Safe accessor for processes.iceflow.unified.adaptive_training.<key>."""
    try:
        block = getattr(cfg_unified, "adaptive_training", None)
        if block is None:
            return default
        v = getattr(block, key, default)
        return default if v is None else v
    except Exception:
        return default


def _ct_full_field(cfg, state) -> tf.Tensor:
    """Build the full [H, W, C] input field as a TF float32 tensor on GPU.

    No GPU→CPU transfer here — everything downstream stays on GPU until the
    final small-scalar log step.
    """
    from igm.processes.iceflow.utils.data_preprocessing import fieldin_state_to_X
    X = fieldin_state_to_X(cfg, state)
    if not isinstance(X, tf.Tensor):
        X = tf.convert_to_tensor(X)
    return tf.cast(X, tf.float32)


def _ct_refresh_sigma_c(state, X_full: tf.Tensor) -> None:
    """Lazy refresh of per-channel σ_c, cached on state._ct_sigma_c (TF tensor).

    Computed every SIGMA_REFRESH_FREQ steps from the variance of the current
    full field across the spatial dims. Also caches the raw per-channel
    variance (state._ct_var_c) — `_ct_compute_delta` masks channels with
    `var_c < VAR_FLOOR` so genuinely constant channels are cleanly excluded.
    All ops stay on GPU; the resulting σ_c is a [C] tf.float32 tensor.
    Independent of the network's input_normalizer — see §5 of
    adaptive_unique_strategy.md.
    """
    n = int(getattr(state, "_ct_n_steps_observed", 0))
    var = getattr(state, "_ct_var_c", None)
    if var is None or (n % SIGMA_REFRESH_FREQ == 0):
        # Variance over spatial dims [H, W] → shape [C]. Stay on GPU.
        var = tf.math.reduce_variance(X_full, axis=[0, 1])
        state._ct_var_c = var
        # σ from raw variance; channels with var < VAR_FLOOR are excluded by
        # the mask in _ct_compute_delta. A tiny 1e-12 epsilon avoids
        # exactly-zero σ for cleaner downstream arithmetic.
        state._ct_sigma_c = tf.sqrt(tf.maximum(var, 1e-12))


def _ct_compute_delta(state, X_full: tf.Tensor):
    """Return (Δ_scalar_tf, δ_c_tf [C], per_cell_tf [H, W]) — all TF tensors on GPU.

    δ_c = mean_xy(|X_c - X_prev_c|), per-channel.
    Δ̂_c = δ_c / σ_c, channels with var_c < VAR_FLOOR are excluded (set to 0).
    Δ   = mean_c Δ̂_c restricted to eligible channels (hardcoded mean reduction).
    per_cell [H, W] = max_c |ΔX_c| / σ_c (over eligible channels) — used for
        per-window aggregation.
    Bootstrap (no previous snapshot yet): returns zeros.
    """
    prev = getattr(state, "_ct_prev_X", None)
    sigma = state._ct_sigma_c                 # tf [C]
    var = getattr(state, "_ct_var_c", None)   # tf [C]
    C = int(X_full.shape[-1])

    if prev is None or tuple(prev.shape) != tuple(X_full.shape):
        zeros_c = tf.zeros([C], dtype=tf.float32)
        zeros_hw = tf.zeros(X_full.shape[:2], dtype=tf.float32)
        return tf.constant(0.0, dtype=tf.float32), zeros_c, zeros_hw

    # All ops below stay on GPU.
    diff = tf.abs(X_full - prev)                       # [H, W, C]
    delta_c = tf.reduce_mean(diff, axis=[0, 1])         # [C]

    # Channel eligibility: var_c >= VAR_FLOOR (else channel is "constant",
    # divide by σ-floor would give huge numbers). Float mask: 1 if eligible,
    # 0 otherwise.
    mask = tf.cast(var >= VAR_FLOOR, tf.float32)        # [C]
    n_elig = tf.reduce_sum(mask)

    # z_c = δ_c / σ_c, but zero where the channel is ineligible.
    z_c = mask * (delta_c / sigma)                       # [C]
    # Δ = mean over ELIGIBLE channels only.
    Delta = tf.cond(
        n_elig > 0,
        lambda: tf.reduce_sum(z_c) / n_elig,
        lambda: tf.constant(0.0, dtype=tf.float32),
    )

    # per_cell = max over eligible channels of |diff_c| / σ_c.
    # Trick: set ineligible channels' contribution to 0 by multiplying by mask.
    per_cell_z = mask * (diff / sigma)                   # [H, W, C] via broadcast
    per_cell = tf.reduce_max(per_cell_z, axis=-1)        # [H, W]

    return Delta, delta_c, per_cell


def _ct_aggregate_delta_w(per_cell: tf.Tensor, list_windows) -> tf.Tensor:
    """Aggregate per-cell |ΔX|/σ field to per-window means.

    Stays on GPU: slice + tf.reduce_mean per window, then tf.stack the result
    into a [n_windows] TF tensor. For ~300 windows on a 3010×4510 grid this
    is much cheaper than transferring the full per_cell field (~50 MB) to CPU.
    """
    if list_windows is None or len(list_windows) == 0:
        return tf.zeros([0], dtype=tf.float32)
    means = [tf.reduce_mean(per_cell[y0:y0+ly, x0:x0+lx])
             for (y0, x0, ly, lx) in list_windows]
    return tf.stack(means)


def update_credit_observer(cfg, state) -> None:
    """Per-step credit accumulator — fully on GPU (TF tensors throughout).

    Updates these attributes on `state`:
      _ct_credit                : Python float — Σ Δ since last retrain
      _ct_steps_since_retrain   : int — steps since last retrain
      _ct_cumulative_delta_w    : tf.Tensor [n_windows] — Σ δ_w since last retrain
      _ct_sigma_c, _ct_var_c    : tf.Tensor [C] — per-channel σ, raw variance
      _ct_prev_X                : tf.Tensor [H, W, C] — previous full field snapshot
      _ct_n_steps_observed      : int — total steps observed (for refresh cadence)
      _ct_last_delta            : Python float — last Δ value (for the JSONL log)
      _ct_last_delta_c          : tf.Tensor [C] — last per-channel δ_c (for log)
      _ct_last_delta_w_stats    : dict (min/mean/max/n floats) — last per-window stats

    Only scalars/small vectors are transferred GPU→CPU; the [H, W, C] field
    (1.3 GB on exp3) stays on GPU between steps.

    No-op unless adaptive_training.enabled_observation OR
    adaptive_training.enabled is true.
    """
    cfg_unified = cfg.processes.iceflow.unified
    if not (bool(_ct_get(cfg_unified, "enabled_observation", False))
            or bool(_ct_get(cfg_unified, "enabled", False))):
        return

    X_full = _ct_full_field(cfg, state)
    _ct_refresh_sigma_c(state, X_full)
    Delta_tf, delta_c_tf, per_cell_tf = _ct_compute_delta(state, X_full)

    # Map per-cell ΔX/σ onto the current windows (from the patch-selection
    # cache when available; before the first retrain the cache may not exist).
    list_windows = None
    cache = getattr(state, "_ap_cache", None)
    if cache is not None:
        list_windows = cache.get("list_windows", None)

    delta_w_tf = (_ct_aggregate_delta_w(per_cell_tf, list_windows)
                  if list_windows is not None
                  else tf.zeros([0], dtype=tf.float32))

    # Re-size cumulative_delta_w when the window list grew/shrank between retrains.
    cum = getattr(state, "_ct_cumulative_delta_w", None)
    if (cum is None) or (tuple(cum.shape) != tuple(delta_w_tf.shape)):
        cum = tf.zeros_like(delta_w_tf)
    state._ct_cumulative_delta_w = cum + delta_w_tf

    # Per-step scalars/small vectors transferred GPU→CPU here (negligible).
    Delta = float(Delta_tf.numpy())
    state._ct_credit = float(getattr(state, "_ct_credit", 0.0)) + Delta
    state._ct_steps_since_retrain = int(getattr(state, "_ct_steps_since_retrain", 0)) + 1
    state._ct_n_steps_observed = int(getattr(state, "_ct_n_steps_observed", 0)) + 1
    state._ct_prev_X = X_full                # KEPT ON GPU — no .numpy()
    state._ct_last_delta = Delta
    state._ct_last_delta_c = delta_c_tf      # tf [C], converted in the log step

    # δ_w stats: compute on GPU, transfer 3 floats to CPU.
    if int(tf.size(delta_w_tf)) > 0:
        state._ct_last_delta_w_stats = {
            "min":  float(tf.reduce_min(delta_w_tf).numpy()),
            "mean": float(tf.reduce_mean(delta_w_tf).numpy()),
            "max":  float(tf.reduce_max(delta_w_tf).numpy()),
            "n":    int(delta_w_tf.shape[0]),
        }
    else:
        state._ct_last_delta_w_stats = {"min": 0.0, "mean": 0.0, "max": 0.0, "n": 0}


def log_and_maybe_reset_credit_observer(cfg, state, status, do_solve: bool) -> None:
    """Append one JSONL line and reset accumulators if a retrain just fired.

    Called from `solve_iceflow` AFTER the optimizer step has run, so the
    log reflects the state at retrain time (with credit/cumulative_δ_w
    BEFORE the reset).

    No-op unless adaptive_training.enabled_observation OR
    adaptive_training.enabled is true.
    """
    cfg_unified = cfg.processes.iceflow.unified
    if not (bool(_ct_get(cfg_unified, "enabled_observation", False))
            or bool(_ct_get(cfg_unified, "enabled", False))):
        return

    record_path = str(_ct_get(cfg_unified, "record_path", "credit_log.jsonl"))

    status_name = getattr(status, "name", str(status))
    is_retrain = bool(do_solve) and status_name in ("INIT", "WARM_UP", "DEFAULT")

    # σ_c and δ_c are TF tensors on GPU now; bring them to CPU once for the log.
    sigma_tf = getattr(state, "_ct_sigma_c", None)
    delta_c_tf = getattr(state, "_ct_last_delta_c", None)
    sigma_list = ([float(s) for s in sigma_tf.numpy()]
                  if isinstance(sigma_tf, tf.Tensor) else [])
    delta_c_list = ([float(d) for d in delta_c_tf.numpy()]
                    if isinstance(delta_c_tf, tf.Tensor) else [])

    rec = {
        "step": _current_step(state),
        "t": (float(state.t.numpy()) if hasattr(state, "t") and hasattr(state.t, "numpy")
              else None),
        "delta":  float(getattr(state, "_ct_last_delta", 0.0)),
        "credit": float(getattr(state, "_ct_credit", 0.0)),
        "steps_since_retrain": int(getattr(state, "_ct_steps_since_retrain", 0)),
        "sigma_c": sigma_list,
        "delta_c": delta_c_list,
        "delta_w_stats": getattr(state, "_ct_last_delta_w_stats", {}),
        "status": status_name,
        "is_retrain": is_retrain,
    }
    try:
        with open(record_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    except OSError:
        pass  # never let logging crash a run

    if is_retrain:
        # Reset accumulators so the NEXT inter-retrain interval starts fresh.
        # Note: _ct_cumulative_delta_w is resized lazily next step if the
        # window list has been rebuilt by select_patches.
        state._ct_credit = 0.0
        state._ct_steps_since_retrain = 0
        if hasattr(state, "_ct_cumulative_delta_w"):
            state._ct_cumulative_delta_w = tf.zeros_like(state._ct_cumulative_delta_w)
