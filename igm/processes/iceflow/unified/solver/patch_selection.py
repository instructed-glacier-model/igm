#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""
Adaptive patch selection for the unified iceflow solver.

PIPELINE (one call to `select_patches`):

  Once early — compute the fixed-shape budget:
      bs      = max(1, floor(framesizemax^2 / (ly · lx)))
      N_train = floor(grid_cover_count / bs) · bs          (multiple of bs)

  STEP 1 — WINDOW GENERATION (Layer 1)
      list_windows = generator(state, cfg_ap)
      Available generators:
        - "regular_grid"     non-overlapping tiles, exact coverage
        - "sliding_overlap"  windows at stride = ly · stride_factor
        - "peak_augmented"   regular_grid + n_extra_peaks windows at score peaks (DEFAULT)

  STEP 2 — SCORING (with freq computation, including the thk filter)
      score[i] is computed by one of three modes (cfg.scoring):
        - "max" / "mean"          → reduction of |dh/dt| inside window i
        - "temporal_variance"     → per-cell squared change of the input
                                     field between the previous and current
                                     scoring snapshot, aggregated to window i.
                                     Captures "this region is locally evolving"
                                     without using |dh/dt| directly. Bootstrap
                                     on the first call falls back to |dh/dt| max.

      freq[i] / weight[i] is computed by one of two modes (cfg.weighting):
        - "integer" (default — legacy)
              freq[i] = 0                   if max(thk in window i) < min_thk_in_window
                        1                   if selection == "all"
                        clip(round(min_freq + (max_freq − min_freq) · score[i] / s_max),
                             1, max_freq)   if selection == "scored"
                        where max_freq = min_freq · freq_ratio.
        - "continuous"
              p[i]    ∝ (score[i] + eps)^score_alpha           if eligible
                       0                                       if not eligible
              No integer clamping; only relative ratios matter, so the result
              is scale-invariant and works irrespectively of the window size.

  STEP 3 — TRAINING WINDOWS (Layer 2)
      training_windows = length-N_train int array of indices into list_windows.
        - selection == "all"        → deterministic: list_windows truncated or
                                        padded (cyclically) to N_train.
        - selection == "scored", weighting == "integer"
                                    → N_train draws with replacement weighted by freq.
        - selection == "scored", weighting == "continuous"
                                    → N_train draws with replacement weighted by
                                       (score + eps)^score_alpha.
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
      adaptive_patching.windows         Window generator (Step 1)
      adaptive_patching.selection       "all" | "scored"  (Step 2/3 weighting)
      adaptive_patching.scoring         "max" | "mean" | "temporal_variance"
      adaptive_patching.min_thk_in_window  (m) Drop windows below this thk → freq=0.
      adaptive_patching.weighting       "integer" (default) | "continuous"
      adaptive_patching.score_alpha     Continuous-weighting exponent (default 2.0).
                                          Only used when weighting=continuous.
      adaptive_patching.temporal_downsample  Downsampling factor for the
                                          per-cell temporal_variance signal
                                          (default 1 = full resolution).
      adaptive_patching.min_freq        Integer-weighting freq floor (default 1)
      adaptive_patching.freq_ratio      Integer-weighting max_freq / min_freq (default 10)
      adaptive_patching.shuffle_training_windows  bool (default true)
      adaptive_patching.record_training_windows   bool (default false)
      adaptive_patching.record_path     JSONL output path (default training_windows.jsonl)
      adaptive_patching.rng_seed        int|null — null means fresh randomness per call
      adaptive_patching.rescore_freq    Steps 1+2 (windows + scoring + freq) are
                                          recomputed every N retrain calls; in between,
                                          the cached list_windows/scores/freqs are
                                          reused. Step 3 (sampling) runs every call.
                                          Default 10. Use 1 to recompute every call.

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


def _windows_sliding_overlap(state, cfg_ap):
    """Sliding windows at stride = ly · stride_factor.

    stride_factor=1.0 → no overlap (equivalent to regular_grid).
    stride_factor=0.5 → 4× redundancy in 2D.
    """
    dhdt = _get_dhdt(state)
    ny, nx = dhdt.shape
    _, _, ly, lx = _patch_grid_dims(ny, nx, cfg_ap.framesizemax)
    sf = float(cfg_ap.stride_factor)
    stride_y = max(1, int(round(ly * sf)))
    stride_x = max(1, int(round(lx * sf)))

    ys = list(range(0, max(1, ny - ly + 1), stride_y))
    xs = list(range(0, max(1, nx - lx + 1), stride_x))
    if not ys or ys[-1] != ny - ly:
        ys.append(ny - ly)
    if not xs or xs[-1] != nx - lx:
        xs.append(nx - lx)

    return [(int(y0), int(x0), int(ly), int(lx)) for y0 in ys for x0 in xs]


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
    "sliding_overlap": _windows_sliding_overlap,
    "peak_augmented": _windows_peak_augmented,
}


# ===========================================================================
# Step 2 — Scoring + freq
# ===========================================================================

def _score_windows(state, windows, scoring: str,
                   X_full=None, temporal_downsample: int = 1) -> np.ndarray:
    """Score each window. Two scoring families are supported:

      "max" / "mean"          → reduction of |dh/dt| inside each window
                                 (the legacy proxy; no extra state needed).
      "temporal_variance"     → cell-level squared change of the input field
                                 between the previous and the current scoring
                                 call, aggregated into each window. Robust to
                                 windows being regenerated each rescore (e.g.
                                 peak_augmented placing extras at fresh peak
                                 locations) because the temporal signal lives
                                 on the (stable) grid, not on windows.

    For temporal_variance, `X_full` (the cached [H, W, C] input tensor) is
    required; the previous-call snapshot is parked on
    `state._prev_full_field` (a single downsampled [H_lr, W_lr, C] array —
    no rolling history, only the latest pair). Bootstrap on the first call
    falls back to per-window |dh/dt| max so the cold-start case is sensible.
    """
    if scoring == "temporal_variance":
        return _score_temporal_variance(
            state, windows, X_full,
            downsample=temporal_downsample,
        )
    # default: dh/dt max / mean
    dhdt = _get_dhdt(state)
    scores = np.zeros(len(windows), dtype=np.float64)
    for i, (y0, x0, ly, lx) in enumerate(windows):
        p = dhdt[y0:y0+ly, x0:x0+lx]
        scores[i] = float(np.mean(p)) if scoring == "mean" else float(np.max(p))
    return scores


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
    regardless of selector. Returns all-True when min_thk_in_window <= 0.
    """
    if min_thk_in_window <= 0.0:
        return np.ones(len(windows), dtype=bool)
    thk = state.thk.numpy() if hasattr(state.thk, "numpy") else np.array(state.thk)
    return np.array(
        [float(np.max(thk[y0:y0+ly, x0:x0+lx])) >= min_thk_in_window
         for (y0, x0, ly, lx) in windows],
        dtype=bool,
    )


def _scores_to_freqs(scores: np.ndarray, eligible: np.ndarray,
                     selection: str, min_freq: int, freq_ratio: int) -> np.ndarray:
    """Map per-window scores to integer frequencies.

    - Ineligible (thk-filtered) windows → freq = 0 (never sampled).
    - selection == "all"       → freq = 1 for every eligible window (uniform).
    - selection == "scored" → freq = clip(round(min_freq + (max_freq - min_freq)
                                              · score / s_max), 1, max_freq),
                                  where max_freq = min_freq · freq_ratio.
    """
    n = len(scores)
    freqs = np.zeros(n, dtype=np.int32)

    if selection == "all":
        freqs[eligible] = 1
        return freqs

    # selection == "scored"
    max_freq = max(min_freq, min_freq * int(freq_ratio))
    if max_freq <= 0:
        return freqs

    s_clean = np.where(np.isfinite(scores), scores, 0.0)
    pos_mask = eligible & (s_clean > 0)
    if not pos_mask.any():
        # No positive |dh/dt| anywhere yet → fall back to uniform among eligibles.
        freqs[eligible] = max(1, min_freq)
        return freqs

    s_max = float(s_clean[pos_mask].max())
    if not np.isfinite(s_max) or s_max <= 0.0:
        freqs[eligible] = max(1, min_freq)
        return freqs

    span = max(1, max_freq - min_freq)
    for i in range(n):
        if not eligible[i]:
            continue
        f = min_freq + span * (s_clean[i] / s_max)
        freqs[i] = int(np.clip(round(f), max(1, min_freq), max_freq))
    return freqs


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


def _build_training_windows(freqs: np.ndarray, scores: np.ndarray,
                            eligible: np.ndarray, selection: str,
                            weighting: str, score_alpha: float,
                            n_train: int, rng: np.random.Generator) -> np.ndarray:
    """Return a length-N_train int array of indices into list_windows.

    `selection == "all"`:
        Deterministic. Tile the eligible-window indices cyclically to fill
        N_train. Ineligible (freq=0) windows are skipped. `weighting` has
        no effect in this mode.

    `selection == "scored"`:
        N_train independent draws with replacement, with weights derived
        from `weighting`:
          - "integer"     → p ∝ freqs (the legacy clamped-integer values)
          - "continuous"  → p ∝ (scores+eps)^score_alpha (scale-invariant)
    """
    n = len(freqs)

    if selection == "all":
        eligible_idx = np.where(freqs > 0)[0]
        if eligible_idx.size == 0:
            return np.zeros(n_train, dtype=np.int32)
        reps = int(np.ceil(n_train / eligible_idx.size))
        out = np.tile(eligible_idx, reps)[:n_train]
        return out.astype(np.int32)

    # selection == "scored"
    if weighting == "continuous":
        p = _continuous_probs(scores, eligible, score_alpha)
        if p.sum() <= 0:
            return np.zeros(n_train, dtype=np.int32)
        return rng.choice(n, size=n_train, replace=True, p=p).astype(np.int32)

    # weighting == "integer" (legacy): probability ∝ freqs
    if freqs.sum() <= 0:
        return np.zeros(n_train, dtype=np.int32)
    p = freqs.astype(np.float64) / float(freqs.sum())
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
    ap.scoring                    = str(_get("scoring", "max"))
    ap.min_thk_in_window          = float(_get("min_thk_in_window", 0.0))
    ap.min_freq                   = int(_get("min_freq", 1))
    ap.freq_ratio                 = int(_get("freq_ratio", 10))
    ap.stride_factor              = float(_get("stride_factor", 1.0))
    ap.n_extra_peaks              = int(_get("n_extra_peaks", -1))
    ap.shuffle_training_windows   = bool(_get("shuffle_training_windows", True))
    ap.record_training_windows    = bool(_get("record_training_windows", False))
    ap.record_path                = _get("record_path", None) or "training_windows.jsonl"
    ap.rng_seed                   = _get("rng_seed", None)
    ap.rescore_freq               = int(_get("rescore_freq", 10))
    # New (back-compat defaults preserve legacy behaviour):
    ap.weighting                  = str(_get("weighting", "integer"))
    ap.score_alpha                = float(_get("score_alpha", 2.0))
    ap.temporal_downsample        = int(_get("temporal_downsample", 1))
    # Note: temporal_history_K was removed in 2026-05-20 after run_06 showed
    # K∈{2,5,10} gave indistinguishable results. The scoring now uses a fixed
    # 2-sample squared diff (current vs previous snapshot). If the knob still
    # appears in legacy configs it is silently ignored.

    # Capacity = data_preparation.framesizemax (one tile of this size fits at bs=1).
    framemax = int(cfg.processes.iceflow.unified.data_preparation.framesizemax)
    # Actual tile size used by the window generators. When > 0 overrides framemax;
    # when 0 falls back to framemax (legacy single-knob behaviour).
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
    if ap.scoring not in ("max", "mean", "temporal_variance"):
        raise ValueError(
            f"unknown scoring: {ap.scoring!r}; "
            f"available: max, mean, temporal_variance"
        )
    if ap.weighting not in ("integer", "continuous"):
        raise ValueError(
            f"unknown weighting: {ap.weighting!r}; available: integer, continuous"
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

    # --- Steps 1+2: cached across calls, rebuilt every `rescore_freq` calls ---
    # Cache lives on state._ap_cache. It carries list_windows, scores,
    # eligible mask, freqs, and the call-counter at last rebuild. The cache
    # is invalidated when:
    #   - it doesn't exist yet (first call)
    #   - the call counter has advanced by >= rescore_freq since last rebuild
    #   - the configured selection/scoring/thk-filter/freq knobs changed
    #     (so a per-trial sweep starting fresh always rebuilds)
    #   - the bs/ly/lx changed (e.g. grid size changed mid-run)
    cache = getattr(state, "_ap_cache", None)
    cache_key = (ap.windows, ap.selection, ap.scoring, ap.min_thk_in_window,
                 ap.min_freq, ap.freq_ratio, ap.framesizemax, ap.n_extra_peaks,
                 ap.weighting, ap.score_alpha, ap.temporal_downsample)
    call_n = (cache["call_n"] + 1) if cache is not None else 0
    needs_rebuild = (
        cache is None
        or cache.get("key") != cache_key
        or (ap.rescore_freq > 0 and call_n % ap.rescore_freq == 0)
    )
    if needs_rebuild:
        # Step 1: generate list_windows
        list_windows = _WINDOW_GENS[ap.windows](state, ap)
        # Step 2: scoring → freqs (thk filter folded in here).
        #         temporal_variance scoring needs the cached input field, so
        #         build it BEFORE _score_windows; it is anyway needed in Step 4.
        X_full = _cache_full_field(state, cfg)
        scores = _score_windows(state, list_windows, ap.scoring,
                                X_full=X_full,
                                temporal_downsample=ap.temporal_downsample)
        eligible = _eligibility_mask(state, list_windows, ap.min_thk_in_window)
        freqs = _scores_to_freqs(scores, eligible, ap.selection,
                                 ap.min_freq, ap.freq_ratio)
        cache = {
            "key": cache_key,
            "list_windows": list_windows,
            "scores": scores,
            "eligible": eligible,
            "freqs": freqs,
            "call_n": call_n,
        }
        state._ap_cache = cache
    else:
        list_windows = cache["list_windows"]
        scores = cache["scores"]
        eligible = cache["eligible"]
        freqs = cache["freqs"]
        cache["call_n"] = call_n
        # Step 4 still needs X_full each call to gather the chosen patches.
        _cache_full_field(state, cfg)

    # --- Step 3: build training_windows (length n_train, indices into list_windows) ---
    training_idx = _build_training_windows(
        freqs, scores, eligible, ap.selection,
        ap.weighting, ap.score_alpha, n_train, rng,
    )
    if ap.shuffle_training_windows:
        rng.shuffle(training_idx)

    if ap.record_training_windows:
        _append_training_record(state, ap, list_windows, scores, freqs,
                                training_idx, bs, n_train)

    # --- Step 4 prep: gather the training patches (full field already cached above) ---
    training_inputs = _gather_patches(state, list_windows, training_idx)

    # Bookkeep previous thickness so |dh/dt| is computable next call.
    state._thk_prev = tf.identity(state.thk)

    return training_inputs, bs


def _append_training_record(state, ap, list_windows, scores, freqs,
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
        "freqs": [int(f) for f in freqs],
        "training_idx": [int(i) for i in training_idx],
    }
    with open(ap.record_path, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
