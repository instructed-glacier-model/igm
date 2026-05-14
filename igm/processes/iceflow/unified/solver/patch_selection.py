#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""
Adaptive patch selection for the unified iceflow solver.

Two-layer design:

  Layer 1 — WINDOW GENERATORS produce a list of (y0, x0, ly, lx) corners
  whose union covers the full domain. Configured by
  cfg.processes.iceflow.unified.adaptive_patching.windows:

    - "regular_grid"     non-overlapping tiles, exact coverage, zero overlap
    - "sliding_overlap"  windows at stride = ly * stride_factor, exact
                          coverage; stride_factor=1.0 → no overlap (= grid),
                          stride_factor=0.5 → 4x redundancy in 2D
    - "peak_augmented"   regular_grid (covering base) + n_extra_peaks
                          windows centered at score peaks (DEFAULT)

  Layer 2 — SELECTORS pick which window indices to retrain on at each call.
  Configured by cfg.processes.iceflow.unified.adaptive_patching.selection:

    - "all"        return every window
    - "top_k"      max_retrain_patches highest-scoring windows
    - "nms"        top-K with greedy IoU overlap suppression
    - "scheduled"  frequency-weighted, memory-bounded round-robin schedule
                    (no duplicates per batch) — DEFAULT

Scoring proxy is |dh/dt|; reduced inside each window by max or mean
(cfg.scoring). Windows below cfg.min_dhdt get score=0 (and freq=0 in the
scheduled selector if min_freq=0).

For "scheduled" selection, all schedule state is parked on
state._patch_schedule across calls; if cfg.record_schedule is true,
per-call metadata is appended to state._patch_schedule_record for
downstream visualization.
"""

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
    to NaN. The score/freq computation downstream then explodes. Treat
    NaN/inf cells as "no proxy signal here" — the scheduler falls back to
    uniform-freq behaviour and keeps running rather than crashing.
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
            # Snap last row/col inward so the union covers [0:ny] x [0:nx] exactly
            y0 = j * ly if j < sy - 1 else (ny - ly)
            x0 = i * lx if i < sx - 1 else (nx - lx)
            windows.append((int(y0), int(x0), int(ly), int(lx)))
    return windows


def _windows_sliding_overlap(state, cfg_ap):
    """Sliding windows at stride = ly * stride_factor; the last row/col is
    snapped to cover the trailing edge.

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

    Coverage is guaranteed by the grid base. Extras add training samples
    where the proxy field |dh/dt| has localised peaks. Each picked peak
    is suppressed in the score map before the next peak is chosen, so
    extras don't collapse onto the same hotspot.

    Special value `n_extra_peaks: -1` → match the number of grid windows
    (so the output has 2N windows: N grid for coverage, N peaks for focus).
    """
    grid = _windows_regular_grid(state, cfg_ap)
    n_extra = int(cfg_ap.n_extra_peaks)
    if n_extra < 0:
        n_extra = len(grid)        # auto = match grid count
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
# Scoring (shared)
# ===========================================================================

def _score_windows(state, windows, scoring, min_thk_in_window=0.0):
    """Score each window by max or mean of |dh/dt| within its extent.

    If `min_thk_in_window > 0`, windows whose max thickness is below the
    threshold are marked ineligible (sentinel score = -1.0). Downstream
    selectors treat negative scores as "skip entirely" — top_k/nms drop
    them via their threshold check, scheduled gives them freq=0.
    """
    dhdt = _get_dhdt(state)
    scores = np.zeros(len(windows), dtype=np.float64)
    if min_thk_in_window > 0.0:
        thk = state.thk.numpy() if hasattr(state.thk, "numpy") else np.array(state.thk)
    else:
        thk = None
    for i, (y0, x0, ly, lx) in enumerate(windows):
        if thk is not None and float(np.max(thk[y0:y0+ly, x0:x0+lx])) < min_thk_in_window:
            scores[i] = -1.0
            continue
        p = dhdt[y0:y0+ly, x0:x0+lx]
        scores[i] = float(np.mean(p)) if scoring == "mean" else float(np.max(p))
    return scores


# ===========================================================================
# Layer 2 — Selectors
# ===========================================================================
# Signature: (scores, cfg_ap, state, windows) → np.ndarray[int]
# Return the indices INTO `windows` that should be retrained this call.
# ===========================================================================

def _select_all(scores, cfg_ap, state, windows):
    return np.arange(len(scores), dtype=np.int32)


def _select_top_k(scores, cfg_ap, state, windows):
    """Top-K windows by score above min_dhdt; fallback to argmax if none.

    forgetting_prevention=True appends one random ice-covered window not
    already selected.
    """
    K = int(cfg_ap.max_retrain_patches)
    threshold = float(cfg_ap.min_dhdt)
    above = np.where(scores >= threshold)[0]
    if len(above) == 0:
        selected = np.array([int(np.argmax(scores))], dtype=np.int32)
    else:
        ranked = above[np.argsort(-scores[above])]
        selected = ranked[:K].astype(np.int32)

    if cfg_ap.forgetting_prevention:
        rng = np.random.default_rng()
        thk = state.thk.numpy() if hasattr(state.thk, "numpy") else np.array(state.thk)
        ice_candidates = []
        for i, (y0, x0, ly, lx) in enumerate(windows):
            if i in selected:
                continue
            if np.max(thk[y0:y0+ly, x0:x0+lx]) > 1.0:
                ice_candidates.append(i)
        if ice_candidates:
            selected = np.append(selected, rng.choice(ice_candidates)).astype(np.int32)
    return selected


def _window_iou(a, b):
    """IoU between two axis-aligned (y0,x0,ly,lx) windows."""
    ay0, ax0, aly, alx = a
    by0, bx0, bly, blx = b
    iy0 = max(ay0, by0); ix0 = max(ax0, bx0)
    iy1 = min(ay0+aly, by0+bly); ix1 = min(ax0+alx, bx0+blx)
    if iy1 <= iy0 or ix1 <= ix0:
        return 0.0
    inter = (iy1 - iy0) * (ix1 - ix0)
    union = aly * alx + bly * blx - inter
    return inter / union if union > 0 else 0.0


def _select_nms(scores, cfg_ap, state, windows):
    """Top-K with greedy IoU overlap suppression."""
    K = int(cfg_ap.max_retrain_patches)
    threshold = float(cfg_ap.min_dhdt)
    order = np.argsort(-scores)
    selected = []
    for idx in order:
        if scores[idx] < threshold:
            break
        if any(_window_iou(windows[idx], windows[s]) > 0.0 for s in selected):
            continue
        selected.append(int(idx))
        if len(selected) >= K:
            break
    if not selected:
        selected.append(int(np.argmax(scores)))
    return np.array(selected, dtype=np.int32)


# --- scheduled selector ---

def _scores_to_freqs(scores, min_freq, max_freq, min_dhdt):
    n = len(scores)
    # NaN/inf scrub.
    scores = np.where(np.isfinite(scores), scores, 0.0)
    # Sentinel: score < 0 → ineligible (e.g. ice-free window).
    eligible = scores >= 0.0
    pos = scores[eligible & (scores > 0)]
    s_max = float(pos.max()) if pos.size else 0.0
    freqs = np.zeros(n, dtype=np.int32)
    if not np.isfinite(s_max) or s_max <= 0.0:
        # No positive scores anywhere — give every eligible window min_freq.
        if min_freq > 0:
            freqs[eligible] = min_freq
        return freqs
    span = max(1, max_freq - min_freq)
    for i in range(n):
        if not eligible[i]:
            freqs[i] = 0
        elif scores[i] < min_dhdt:
            freqs[i] = 0 if min_freq == 0 else min_freq
        else:
            f = min_freq + span * scores[i] / s_max
            freqs[i] = int(np.clip(round(f), max(1, min_freq), max_freq))
    return freqs


def _build_schedule(scores, cfg_ap, n_windows, bs, rng):
    """Round-robin pass scheduler: pass k contains every window with freq>=k.

    Each pass is split into batches of size `bs` with distinct window
    indices, guaranteeing no duplicate index inside any returned batch.
    """
    min_freq = int(cfg_ap.min_freq)
    max_freq = max(min_freq, min_freq * int(cfg_ap.freq_ratio))
    freqs = _scores_to_freqs(scores, min_freq, max_freq, float(cfg_ap.min_dhdt))

    batches = []
    batch_pass_k = []
    for k in range(1, max_freq + 1):
        eligible = np.where(freqs >= k)[0]
        if eligible.size == 0:
            continue
        if cfg_ap.shuffle_within_pass:
            rng.shuffle(eligible)
        for c0 in range(0, eligible.size, bs):
            batches.append(eligible[c0:c0+bs].astype(np.int32))
            batch_pass_k.append(k)

    if cfg_ap.shuffle_pass_order and len(batches) > 1:
        order = np.arange(len(batches))
        rng.shuffle(order)
        batches = [batches[i] for i in order]
        batch_pass_k = [batch_pass_k[i] for i in order]

    return {
        "batches": batches,
        "batch_pass_k": batch_pass_k,
        "cursor": 0,
        "freqs": freqs,
        "scores": scores.copy(),
        "n_windows": n_windows,
    }


def _select_scheduled(scores, cfg_ap, state, windows):
    """Frequency-weighted, memory-bounded round-robin schedule.

    Schedule state is parked on state._patch_schedule across calls. Rebuilt
    when absent, exhausted, on grid changes, or every schedule_rebuild_freq
    retrain-step calls.
    """
    n = len(scores)
    if n == 0:
        return np.array([], dtype=np.int32)
    ly = windows[0][2]
    lx = windows[0][3]

    # bs derived from GPU-capacity (Scenario A): we know one tile of
    # (framemax_capacity, framemax_capacity) fits at bs=1. For tiles of
    # (ly, lx) ≤ (framemax_capacity, framemax_capacity) we can stack:
    #     bs = floor(framemax_capacity^2 / (ly * lx))
    # framemax_capacity = data_preparation.framesizemax. The actual tile
    # size (ly, lx) is derived from `adaptive_patching.patch_size`
    # (overrides framesizemax in the window generators) when set.
    framemax_cap = int(getattr(cfg_ap, "framemax_capacity", 0) or 0)
    bs = max(1, framemax_cap ** 2 // max(1, ly * lx)) if framemax_cap > 0 else n
    bs = max(1, min(bs, n))

    rng = (np.random.default_rng(int(cfg_ap.rng_seed))
           if cfg_ap.rng_seed is not None
           else np.random.default_rng())

    sched = getattr(state, "_patch_schedule", None)
    step_now = _current_step(state)
    rebuild_freq = int(cfg_ap.schedule_rebuild_freq)
    age = step_now - (sched["last_build_step"] if sched else 0)
    needs_rebuild = (
        sched is None
        or sched["cursor"] >= len(sched["batches"])
        or (rebuild_freq > 0 and age >= rebuild_freq)
        or sched.get("bs") != bs
        or sched.get("n_windows") != n
    )

    if needs_rebuild:
        sched = _build_schedule(scores, cfg_ap, n_windows=n, bs=bs, rng=rng)
        sched["last_build_step"] = step_now
        sched["bs"] = bs
        state._patch_schedule = sched

    cursor = sched["cursor"]
    batch_idx = sched["batches"][cursor]
    pass_k = sched["batch_pass_k"][cursor]
    sched["cursor"] = cursor + 1

    if cfg_ap.record_schedule:
        record = {
            "step": step_now,
            "t": float(state.t.numpy()) if hasattr(state, "t") and hasattr(state.t, "numpy") else None,
            "pass_k": int(pass_k),
            "cursor": int(cursor),
            "patch_idx": batch_idx.tolist(),
            "windows": [list(w) for w in (windows[int(i)] for i in batch_idx.tolist())],
            "freqs": sched["freqs"].tolist() if cursor == 0 else None,
        }
        # In-memory list for downstream introspection in the same process.
        rec = getattr(state, "_patch_schedule_record", None)
        if rec is None:
            rec = []
            state._patch_schedule_record = rec
        rec.append(record)
        # Persistent JSON-lines file in the run cwd so an external plot
        # script can read the schedule after the simulation finishes.
        import json, os
        path = getattr(cfg_ap, "record_path", None) or "schedule_record.jsonl"
        with open(path, "a") as fh:
            fh.write(json.dumps(record) + "\n")

    return batch_idx


_SELECTORS = {
    "all": _select_all,
    "top_k": _select_top_k,
    "nms": _select_nms,
    "scheduled": _select_scheduled,
}


# ===========================================================================
# Patch extraction
# ===========================================================================

def _gather_patches(state, windows, indices):
    """Slice patches at windows[indices] from the cached full field."""
    X = state._adaptive_patching_X  # set by select_patches dispatcher below
    patches = [X[w[0]:w[0]+w[2], w[1]:w[1]+w[3], :]
               for w in (windows[int(i)] for i in indices)]
    return tf.stack(patches, axis=0) if patches else tf.zeros((0,) + tuple(X.shape[1:]), dtype=X.dtype)


# ===========================================================================
# Dispatcher
# ===========================================================================

def select_patches(cfg: DictConfig, state: State, inputs: tf.Tensor) -> tf.Tensor:
    """Two-layer adaptive patch selection.

    Pipeline per call:
      1. Generate covering windows (cfg.windows).
      2. Score each window by |dh/dt| reduction (cfg.scoring).
      3. Pick window indices (cfg.selection).
      4. Slice patches at those windows from the full field and return.

    Called from solve_iceflow() between input preparation and
    optimizer.minimize(). The `inputs` argument (the pre-split regular
    grid tensor) is accepted but not used — we slice patches directly
    from the full field for uniform handling of off-grid windows.
    """
    cfg_ap = cfg.processes.iceflow.unified.adaptive_patching

    def _get(key, default):
        try:
            v = cfg_ap[key]
            return default if v is None else v
        except Exception:
            return default

    # Pack all knobs (with defaults) into a plain namespace.
    class _Cfg:
        pass
    ap = _Cfg()
    ap.windows               = str(_get("windows", "peak_augmented"))
    ap.selection             = str(_get("selection", "scheduled"))
    ap.scoring               = str(_get("scoring", "max"))
    ap.min_dhdt              = float(_get("min_dhdt", 0.0))
    # Skip windows whose max(thk) is below this (m). 0 = no ice-mask filter.
    ap.min_thk_in_window     = float(_get("min_thk_in_window", 0.0))
    # Capacity = data_preparation.framesizemax (= "framemax" in the bs formula:
    # the max single-tile size that fits on the GPU at bs=1). Used only by
    # the scheduled selector to derive batch size.
    ap.framemax_capacity     = int(cfg.processes.iceflow.unified.data_preparation.framesizemax)
    # Actual tile size used by the window generators (= "ps" in the bs formula).
    # When patch_size = 0 (default) we fall back to framesizemax, preserving the
    # original single-knob behaviour. The generators read this as ap.framesizemax.
    _patch_size              = int(_get("patch_size", 0) or 0)
    ap.framesizemax          = _patch_size if _patch_size > 0 else ap.framemax_capacity
    # Generator-specific
    ap.stride_factor         = float(_get("stride_factor", 1.0))
    ap.n_extra_peaks         = int(_get("n_extra_peaks", -1))
    # Selector-specific (top_k / nms)
    ap.max_retrain_patches   = int(_get("max_retrain_patches", 2))
    ap.forgetting_prevention = bool(_get("forgetting_prevention", False))
    # Selector-specific (scheduled)
    ap.min_freq              = int(_get("min_freq", 1))
    ap.freq_ratio            = int(_get("freq_ratio", 10))
    ap.schedule_rebuild_freq = int(_get("schedule_rebuild_freq", 10))
    ap.shuffle_within_pass   = bool(_get("shuffle_within_pass", True))
    ap.shuffle_pass_order    = bool(_get("shuffle_pass_order", False))
    ap.record_schedule       = bool(_get("record_schedule", False))
    ap.record_path           = _get("record_path", None)  # default: schedule_record.jsonl in cwd
    ap.rng_seed              = _get("rng_seed", None)

    if ap.windows not in _WINDOW_GENS:
        raise ValueError(
            f"unknown windows generator: {ap.windows!r}; "
            f"available: {list(_WINDOW_GENS)}"
        )
    if ap.selection not in _SELECTORS:
        raise ValueError(
            f"unknown selection: {ap.selection!r}; "
            f"available: {list(_SELECTORS)}"
        )

    # 1. Generate covering windows
    windows = _WINDOW_GENS[ap.windows](state, ap)

    # 2. Score each window
    scores = _score_windows(state, windows, ap.scoring, ap.min_thk_in_window)

    # 3. Pick window indices
    indices = _SELECTORS[ap.selection](scores, ap, state, windows)

    # 4. Cache the full field and slice the chosen windows out of it
    _cache_full_field(state, cfg)
    selected = _gather_patches(state, windows, indices)

    # Bookkeep previous thickness so |dh/dt| is computable next call
    state._thk_prev = tf.identity(state.thk)

    n_total = len(windows)
    n_selected = int(indices.shape[0]) if hasattr(indices, "shape") else len(indices)
#    print(f"  Adaptive patching ({ap.windows} × {ap.selection}): "
#          f"{n_selected}/{n_total} windows")

    return selected
