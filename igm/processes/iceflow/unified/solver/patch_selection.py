#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""
Adaptive patch selection for the unified iceflow solver.

Instead of retraining on all patches, score each patch by |dh/dt| and
select only those where the geometry is changing fastest. Four strategies
are available, selectable via cfg.processes.iceflow.unified.adaptive_patching.strategy:

  - "none"    : retrain all patches (current default behaviour)
  - "topk"    : regular grid + top-K scoring
  - "nms"     : sliding window + greedy non-maximum suppression
  - "maxpool" : max-pool peak detection + patch extraction
  - "tfnms"   : TF-native sliding window + tf.image.non_max_suppression
"""

import numpy as np
import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State


# ---------------------------------------------------------------------------
# Helper: compute patch grid dimensions (must match split_field_into_patches)
# ---------------------------------------------------------------------------

def _patch_grid_dims(ny: int, nx: int, framesizemax: int):
    """Return (sy, sx, ly, lx) — number of strips and patch size in each dim."""
    sy = ny // framesizemax + 1
    sx = nx // framesizemax + 1
    ly = ny // sy
    lx = nx // sx
    return sy, sx, ly, lx


# ---------------------------------------------------------------------------
# Helper: get |dh/dt| field from state
# ---------------------------------------------------------------------------

def _get_dhdt(state: State) -> np.ndarray:
    """Return |dh/dt| as a numpy array [H, W]."""
    if hasattr(state, "dhdt") and state.dhdt is not None:
        return np.abs(state.dhdt.numpy() if hasattr(state.dhdt, "numpy") else np.array(state.dhdt))

    # Fallback: compute from stored previous thickness
    if hasattr(state, "_thk_prev") and state._thk_prev is not None:
        dt = float(state.dt) if hasattr(state, "dt") and state.dt > 0 else 1.0
        dhdt = np.abs((state.thk.numpy() - state._thk_prev.numpy()) / dt)
        return dhdt

    # First timestep: no history yet, return zeros (will retrain all)
    return np.zeros_like(state.thk.numpy())


# ---------------------------------------------------------------------------
# Helper: add a random ice-covered patch for forgetting prevention
# ---------------------------------------------------------------------------

def _add_random_patch(selected_indices, thk_patches, rng=None):
    """Append one random ice-covered patch index not already selected."""
    if rng is None:
        rng = np.random.default_rng()

    n_patches = thk_patches.shape[0]
    ice_mask = np.array([
        np.max(thk_patches[i]) > 1.0 for i in range(n_patches)
    ])
    candidates = np.where(ice_mask)[0]
    candidates = np.setdiff1d(candidates, selected_indices)

    if len(candidates) > 0:
        return np.append(selected_indices, rng.choice(candidates))
    return selected_indices


# ===========================================================================
# Strategy: none — retrain all patches (default behaviour)
# ===========================================================================

def select_none(inputs: tf.Tensor, state: State, cfg_ap: DictConfig) -> tf.Tensor:
    """No filtering — return all patches unchanged."""
    return inputs


# ===========================================================================
# Strategy A: topk — regular grid + top-K
# ===========================================================================

def select_topk(inputs: tf.Tensor, state: State, cfg_ap: DictConfig) -> tf.Tensor:
    """Score each patch on the existing regular grid by max|dh/dt|, pick top-K."""
    dhdt = _get_dhdt(state)
    ny, nx = dhdt.shape
    framesizemax = cfg_ap.framesizemax
    sy, sx, ly, lx = _patch_grid_dims(ny, nx, framesizemax)

    K = cfg_ap.max_retrain_patches
    threshold = cfg_ap.min_dhdt
    scoring = cfg_ap.scoring  # "max" or "mean"

    # Score each patch
    scores = []
    for j in range(sy):
        for i in range(sx):
            patch_dhdt = dhdt[j * ly : (j + 1) * ly, i * lx : (i + 1) * lx]
            if scoring == "mean":
                scores.append(np.mean(patch_dhdt))
            else:
                scores.append(np.max(patch_dhdt))

    scores = np.array(scores)

    # Select: above threshold, then top-K
    above = np.where(scores >= threshold)[0]
    if len(above) == 0:
        # Nothing above threshold — retrain the single highest patch
        selected = np.array([np.argmax(scores)])
    else:
        ranked = above[np.argsort(-scores[above])]
        selected = ranked[:K]

    # Forgetting prevention
    if cfg_ap.forgetting_prevention:
        # Build thickness patches for ice detection
        thk_idx = _get_thk_channel_index(state)
        if thk_idx is not None:
            thk_patches = inputs.numpy()[:, :, :, thk_idx]
        else:
            thk_patches = np.max(np.abs(inputs.numpy()), axis=-1)
        selected = _add_random_patch(selected, thk_patches)

    return tf.gather(inputs, selected.astype(np.int32))


# ===========================================================================
# Strategy B: nms — sliding window + greedy non-maximum suppression
# ===========================================================================

def select_nms(inputs: tf.Tensor, state: State, cfg_ap: DictConfig) -> tf.Tensor:
    """Half-stride sliding window over |dh/dt|, greedy NMS, extract patches from X."""
    from scipy.ndimage import uniform_filter

    dhdt = _get_dhdt(state)
    ny, nx = dhdt.shape
    framesizemax = cfg_ap.framesizemax
    _, _, ly, lx = _patch_grid_dims(ny, nx, framesizemax)

    K = cfg_ap.max_retrain_patches
    threshold = cfg_ap.min_dhdt
    scoring = cfg_ap.scoring

    # Score map: mean or max |dh/dt| in every ly×lx window
    if scoring == "mean":
        # uniform_filter gives the mean in a window of size (ly, lx)
        score_map = uniform_filter(dhdt, size=(ly, lx), mode="constant")
    else:
        from scipy.ndimage import maximum_filter
        score_map = maximum_filter(dhdt, size=(ly, lx), mode="constant")

    # Candidate positions on a half-stride grid
    stride_y = max(ly // 2, 1)
    stride_x = max(lx // 2, 1)

    candidates = []
    for iy in range(ly // 2, ny - ly // 2, stride_y):
        for ix in range(lx // 2, nx - lx // 2, stride_x):
            candidates.append((score_map[iy, ix], iy, ix))

    candidates.sort(reverse=True, key=lambda c: c[0])

    # Greedy NMS — suppress overlapping candidates
    selected_positions = []
    suppressed = set()
    for score, cy, cx in candidates:
        if (cy, cx) in suppressed or score < threshold:
            continue
        y0 = max(cy - ly // 2, 0)
        x0 = max(cx - lx // 2, 0)
        # Clamp to valid range
        y0 = min(y0, ny - ly)
        x0 = min(x0, nx - lx)
        selected_positions.append((y0, x0))
        # Suppress nearby candidates
        for _, sy, sx in candidates:
            if abs(sy - cy) < ly and abs(sx - cx) < lx:
                suppressed.add((sy, sx))
        if len(selected_positions) >= K:
            break

    if len(selected_positions) == 0:
        # Fallback: pick center of domain
        selected_positions.append((ny // 2 - ly // 2, nx // 2 - lx // 2))

    # Forgetting prevention: add a random position over ice
    if cfg_ap.forgetting_prevention and len(selected_positions) < K:
        rng = np.random.default_rng()
        for _ in range(10):  # try up to 10 times
            ry = rng.integers(0, max(ny - ly, 1))
            rx = rng.integers(0, max(nx - lx, 1))
            thk = state.thk.numpy()[ry : ry + ly, rx : rx + lx]
            if np.max(thk) > 1.0:
                # Check no overlap with existing
                overlap = any(abs(ry - y) < ly and abs(rx - x) < lx for y, x in selected_positions)
                if not overlap:
                    selected_positions.append((ry, rx))
                    break

    # Extract patches from the full field
    X = _get_full_field(state, inputs)
    patches = [X[y0 : y0 + ly, x0 : x0 + lx, :] for y0, x0 in selected_positions]

    dtype = inputs.dtype
    return tf.cast(tf.stack(patches, axis=0), dtype)


# ===========================================================================
# Strategy C: maxpool — max-pool peak detection
# ===========================================================================

def select_maxpool(inputs: tf.Tensor, state: State, cfg_ap: DictConfig) -> tf.Tensor:
    """Find local maxima in |dh/dt| at patch scale, extract patches at peak centers."""
    from scipy.ndimage import maximum_filter, uniform_filter

    dhdt = _get_dhdt(state)
    ny, nx = dhdt.shape
    framesizemax = cfg_ap.framesizemax
    _, _, ly, lx = _patch_grid_dims(ny, nx, framesizemax)

    K = cfg_ap.max_retrain_patches
    threshold = cfg_ap.min_dhdt

    # Smooth dhdt at patch scale to avoid pixel-noise peaks
    score_map = uniform_filter(dhdt, size=(ly, lx), mode="constant")

    # Detect peaks: points equal to the local max at patch scale AND above threshold
    local_max = maximum_filter(score_map, size=(ly, lx), mode="constant")
    peaks = (score_map == local_max) & (score_map > threshold)

    # Extract peak coordinates (avoid flat zero regions by the threshold check)
    peak_coords = np.argwhere(peaks)  # (N, 2) array of [y, x]

    if len(peak_coords) == 0:
        # No peaks above threshold — return all patches
        return inputs

    # Score each peak and sort descending
    peak_scores = [(score_map[cy, cx], cy, cx) for cy, cx in peak_coords]
    peak_scores.sort(reverse=True, key=lambda c: c[0])

    # Greedy selection with minimum-distance suppression (no overlapping patches)
    selected_positions = []
    for score, cy, cx in peak_scores:
        y0 = max(min(cy - ly // 2, ny - ly), 0)
        x0 = max(min(cx - lx // 2, nx - lx), 0)
        # Check no overlap with already selected
        overlap = any(abs(y0 - y) < ly and abs(x0 - x) < lx for y, x in selected_positions)
        if not overlap:
            selected_positions.append((y0, x0))
        if len(selected_positions) >= K:
            break

    if len(selected_positions) == 0:
        return inputs

    # Forgetting prevention
    if cfg_ap.forgetting_prevention and len(selected_positions) < K:
        rng = np.random.default_rng()
        for _ in range(10):
            ry = rng.integers(0, max(ny - ly, 1))
            rx = rng.integers(0, max(nx - lx, 1))
            thk = state.thk.numpy()[ry : ry + ly, rx : rx + lx]
            if np.max(thk) > 1.0:
                overlap = any(abs(ry - y) < ly and abs(rx - x) < lx for y, x in selected_positions)
                if not overlap:
                    selected_positions.append((ry, rx))
                    break

    X = _get_full_field(state, inputs)
    patches = [X[y0 : y0 + ly, x0 : x0 + lx, :] for y0, x0 in selected_positions]

    dtype = inputs.dtype
    return tf.cast(tf.stack(patches, axis=0), dtype)


# ===========================================================================
# Strategy D: tfnms — TensorFlow-native with built-in NMS (fully GPU)
# ===========================================================================

def select_tfnms(inputs: tf.Tensor, state: State, cfg_ap: DictConfig) -> tf.Tensor:
    """TF-native: extract_patches + tf.image.non_max_suppression, fully on GPU."""
    dhdt = _get_dhdt(state)
    ny, nx = dhdt.shape
    framesizemax = cfg_ap.framesizemax
    _, _, ly, lx = _patch_grid_dims(ny, nx, framesizemax)

    K = cfg_ap.max_retrain_patches
    threshold = cfg_ap.min_dhdt

    # Build 4D tensor for tf.image.extract_patches
    C = tf.abs(tf.constant(dhdt, dtype=tf.float32))[tf.newaxis, :, :, tf.newaxis]

    stride_y = max(ly // 2, 1)
    stride_x = max(lx // 2, 1)

    # Extract overlapping patches of |dh/dt|
    dhdt_patches = tf.image.extract_patches(
        C,
        sizes=[1, ly, lx, 1],
        strides=[1, stride_y, stride_x, 1],
        rates=[1, 1, 1, 1],
        padding="VALID",
    )  # (1, ny_p, nx_p, ly*lx)

    # Score each candidate patch
    scores = tf.reduce_max(dhdt_patches, axis=-1)  # (1, ny_p, nx_p)
    ny_p = scores.shape[1]
    nx_p = scores.shape[2]

    # Build bounding boxes for NMS
    iy, ix = tf.meshgrid(tf.range(ny_p), tf.range(nx_p), indexing="ij")
    y0 = tf.cast(tf.reshape(iy, [-1]) * stride_y, tf.float32)
    x0 = tf.cast(tf.reshape(ix, [-1]) * stride_x, tf.float32)
    y1 = y0 + float(ly)
    x1 = x0 + float(lx)
    boxes = tf.stack([y0, x0, y1, x1], axis=1)
    flat_scores = tf.reshape(scores, [-1])

    # NMS with iou_threshold=0.0 → no overlap allowed
    selected_idx = tf.image.non_max_suppression(
        boxes, flat_scores, max_output_size=K,
        iou_threshold=0.0, score_threshold=threshold,
    )

    if tf.size(selected_idx) == 0:
        # Nothing above threshold — return all patches
        return inputs

    selected_boxes = tf.gather(boxes, selected_idx)

    # Extract patches from the full input field
    X = _get_full_field(state, inputs)
    patches = []
    for i in range(selected_boxes.shape[0]):
        y0_i = int(selected_boxes[i, 0].numpy())
        x0_i = int(selected_boxes[i, 1].numpy())
        # Clamp to valid range
        y0_i = max(min(y0_i, ny - ly), 0)
        x0_i = max(min(x0_i, nx - lx), 0)
        patches.append(X[y0_i : y0_i + ly, x0_i : x0_i + lx, :])

    # Forgetting prevention
    if cfg_ap.forgetting_prevention and len(patches) < K:
        rng = np.random.default_rng()
        for _ in range(10):
            ry = rng.integers(0, max(ny - ly, 1))
            rx = rng.integers(0, max(nx - lx, 1))
            thk = state.thk.numpy()[ry : ry + ly, rx : rx + lx]
            if np.max(thk) > 1.0:
                patches.append(X[ry : ry + ly, rx : rx + lx, :])
                break

    dtype = inputs.dtype
    return tf.cast(tf.stack(patches, axis=0), dtype)


# ===========================================================================
# Helpers
# ===========================================================================

def _get_full_field(state: State, inputs: tf.Tensor) -> tf.Tensor:
    """Reconstruct the full [H, W, C] field from state (same as fieldin_state_to_X)."""
    from igm.processes.iceflow.utils.data_preprocessing import fieldin_state_to_X
    from igm.utils.math.precision import normalize_precision
    # We reconstruct from state; the inputs tensor gives us the dtype
    # but we need cfg — instead, just stack from the known input names in state
    # For simplicity, we infer channels from inputs shape
    # Actually, we can reconstruct from the patches if we know the grid...
    # Safest: re-read from state the same way get_solver_inputs_from_state does.
    # We store X on state during selection to avoid recomputation.
    if hasattr(state, "_adaptive_patching_X"):
        return state._adaptive_patching_X
    raise RuntimeError("Full field X not cached on state — call store_full_field first")


def _get_thk_channel_index(state: State) -> int:
    """Return the channel index of 'thk' in the input tensor, or None."""
    # thk is always the first input channel in the default config
    return 0


# ===========================================================================
# Dispatcher
# ===========================================================================

_STRATEGIES = {
    "none": select_none,
    "topk": select_topk,
    "nms": select_nms,
    "maxpool": select_maxpool,
    "tfnms": select_tfnms,
}


def select_patches(
    cfg: DictConfig, state: State, inputs: tf.Tensor
) -> tf.Tensor:
    """Select patches for retraining based on the configured strategy.

    Called from solve_iceflow() between input preparation and optimizer.minimize().
    """
    cfg_ap = cfg.processes.iceflow.unified.adaptive_patching
    strategy = cfg_ap.strategy

    if strategy == "none":
        return inputs

    # Cache the full field on state for strategies that need it (B, C, D)
    if strategy in ("nms", "maxpool", "tfnms"):
        from igm.processes.iceflow.utils.data_preprocessing import fieldin_state_to_X
        from igm.utils.math.precision import normalize_precision
        X = fieldin_state_to_X(cfg, state)
        dtype = normalize_precision(cfg.processes.iceflow.numerics.precision)
        state._adaptive_patching_X = tf.cast(X, dtype)

    # Store framesizemax in cfg_ap for strategies to use
    cfg_ap_with_fmax = cfg_ap
    # We pass framesizemax through — strategies need it to compute patch dims
    # OmegaConf is read-only, so we use a simple namespace
    class _Cfg:
        pass
    ap = _Cfg()
    ap.strategy = cfg_ap.strategy
    ap.max_retrain_patches = cfg_ap.max_retrain_patches
    ap.min_dhdt = cfg_ap.min_dhdt
    ap.forgetting_prevention = cfg_ap.forgetting_prevention
    ap.scoring = cfg_ap.scoring
    ap.framesizemax = cfg.processes.iceflow.unified.data_preparation.framesizemax

    select_fn = _STRATEGIES[strategy]
    selected = select_fn(inputs, state, ap)

    # Store previous thickness for next call
    state._thk_prev = tf.identity(state.thk)

    n_total = inputs.shape[0]
    n_selected = selected.shape[0]
    if n_selected < n_total:
        print(f"  Adaptive patching ({strategy}): {n_selected}/{n_total} patches selected")

    return selected
