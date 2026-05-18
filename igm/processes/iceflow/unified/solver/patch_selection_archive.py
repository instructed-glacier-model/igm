#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""
ARCHIVED selectors and helpers for the adaptive patch-selection layer.

Three things live here, all removed from `patch_selection.py` during the
2026-05 cleanup of the unified iceflow solver:

  1. The `top_k` selector — Top-K windows by |dh/dt| score, optionally with
     a "forgetting_prevention" random ice-covered window appended.

  2. The `nms` selector — Top-K with greedy IoU overlap suppression
     (relevant only when windows overlap, e.g. `sliding_overlap` or
     `peak_augmented`).

  3. The previous multi-pass round-robin `scheduled` selector
     (`_select_scheduled_v1` + `_build_schedule_v1` + `_scores_to_freqs_v1`).
     Replaced by a simpler "weighted sampling with replacement" scheme
     where each retrain call samples N_train indices from list_windows
     with probability ∝ freq, and the solver sequentially loops over
     `N_train/bs` fixed-shape batches.

The new design is documented in `patch_selection.py` and
`clever-patch/patch_selection.md`. These archives are kept so any of
the old algorithms can be revived without re-deriving them.

────────────────────────────────────────────────────────────────────
Reviving `top_k` and `nms`:
  1. In `patch_selection.py`, import them:
        from .patch_selection_archive import _select_top_k, _select_nms
  2. Restore them in the dispatch table:
        _SELECTORS = ("all", "scheduled", "top_k", "nms")  # update validation
        # …then plumb the call back into select_patches; note the new
        # patch_selection.py returns (tensor, bs), so a top_k/nms call
        # must also return a fixed-shape [N_train, ly, lx, C] tensor.
  3. Restore knobs in select_patches() (or its equivalent):
        ap.max_retrain_patches   = int(_get("max_retrain_patches", 2))
        ap.forgetting_prevention = bool(_get("forgetting_prevention", False))
        ap.min_dhdt              = float(_get("min_dhdt", 0.0))
  4. Restore schema defaults in `igm/conf/processes/iceflow.yaml`:
        adaptive_patching:
          max_retrain_patches: 10
          forgetting_prevention: true
          min_dhdt: 0.0

Reviving the multi-pass round-robin `scheduled` selector:
  1. Replace the new `_build_training_windows` call site with
     `_select_scheduled_v1`, which expects schedule state parked on
     `state._patch_schedule` across calls.
  2. Restore the legacy `_scores_to_freqs_v1` (handles a sentinel score=-1.0
     for ineligible windows; the new design uses a separate boolean mask).
  3. Restore knobs: `schedule_rebuild_freq, shuffle_within_pass,
     shuffle_pass_order, min_dhdt`.
────────────────────────────────────────────────────────────────────
"""

import json

import numpy as np


# ===========================================================================
# Archived: top_k / nms selectors
# ===========================================================================

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


# ===========================================================================
# Archived: previous multi-pass round-robin `scheduled` selector
# ===========================================================================
# Replaced by independent weighted draws in patch_selection.py. The old
# design walked a cross-call schedule (one batch per IGM step over many
# steps) and rebuilt periodically; the new design samples N_train indices
# every call, and the solver loops over N_train/bs fixed batches in one
# IGM step.
# ===========================================================================


def _scores_to_freqs_v1(scores, min_freq, max_freq, min_dhdt):
    """Old freq mapping — uses a score=-1 sentinel for ineligibility.

    The new version (in patch_selection.py) takes an explicit `eligible`
    boolean mask instead, which is cleaner.
    """
    n = len(scores)
    scores = np.where(np.isfinite(scores), scores, 0.0)
    eligible = scores >= 0.0
    pos = scores[eligible & (scores > 0)]
    s_max = float(pos.max()) if pos.size else 0.0
    freqs = np.zeros(n, dtype=np.int32)
    if not np.isfinite(s_max) or s_max <= 0.0:
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


def _build_schedule_v1(scores, cfg_ap, n_windows, bs, rng):
    """Old round-robin scheduler.

    Pass k contains every window with freq >= k. Each pass is split into
    chunks of `bs` distinct indices. The full schedule is the concatenation
    of all passes; the cursor advances 1 batch per retrain call.
    """
    min_freq = int(cfg_ap.min_freq)
    max_freq = max(min_freq, min_freq * int(cfg_ap.freq_ratio))
    freqs = _scores_to_freqs_v1(scores, min_freq, max_freq, float(cfg_ap.min_dhdt))

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


def _current_step_v1(state):
    if hasattr(state, "it"):
        it = state.it
        return int(it.numpy()) if hasattr(it, "numpy") else int(it)
    return 0


def _select_scheduled_v1(scores, cfg_ap, state, windows):
    """Old multi-pass round-robin schedule.

    Schedule state parked on state._patch_schedule. Rebuilt when absent,
    exhausted, on grid changes, or every schedule_rebuild_freq calls.
    """
    n = len(scores)
    if n == 0:
        return np.array([], dtype=np.int32)
    ly = windows[0][2]
    lx = windows[0][3]

    framemax_cap = int(getattr(cfg_ap, "framemax_capacity", 0) or 0)
    bs = max(1, framemax_cap ** 2 // max(1, ly * lx)) if framemax_cap > 0 else n
    bs = max(1, min(bs, n))

    rng = (np.random.default_rng(int(cfg_ap.rng_seed))
           if cfg_ap.rng_seed is not None
           else np.random.default_rng())

    sched = getattr(state, "_patch_schedule", None)
    step_now = _current_step_v1(state)
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
        sched = _build_schedule_v1(scores, cfg_ap, n_windows=n, bs=bs, rng=rng)
        sched["last_build_step"] = step_now
        sched["bs"] = bs
        state._patch_schedule = sched

    if len(sched["batches"]) == 0:
        return np.array([int(np.argmax(scores))], dtype=np.int32)

    cursor = sched["cursor"]
    batch_idx = sched["batches"][cursor]
    pass_k = sched["batch_pass_k"][cursor]
    sched["cursor"] = cursor + 1

    if cfg_ap.record_schedule:
        record = {
            "step": step_now,
            "t": (float(state.t.numpy()) if hasattr(state, "t") and hasattr(state.t, "numpy")
                  else None),
            "pass_k": int(pass_k),
            "cursor": int(cursor),
            "patch_idx": batch_idx.tolist(),
            "windows": [list(w) for w in (windows[int(i)] for i in batch_idx.tolist())],
            "freqs": sched["freqs"].tolist() if cursor == 0 else None,
        }
        rec = getattr(state, "_patch_schedule_record", None)
        if rec is None:
            rec = []
            state._patch_schedule_record = rec
        rec.append(record)
        path = getattr(cfg_ap, "record_path", None) or "schedule_record.jsonl"
        with open(path, "a") as fh:
            fh.write(json.dumps(record) + "\n")

    return batch_idx
