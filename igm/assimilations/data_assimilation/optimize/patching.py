#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""
Tiling helpers for the patch-wise (out-of-core) data assimilation.

The full-grid cost function is (almost) spatially local: each cost pixel
depends on the controls only within a finite radius R (the emulator receptive
field + the regularization stencils + the anisotropic-smoothing kernel).
Hence the full-grid gradient can be assembled from per-tile gradient tapes:
run the forward model and the cost on a window = core + halo, take the
gradient w.r.t. the window controls, and keep only the core part (the halo
ring absorbs the influence of the artificial window edges, provided
halo >= R). Cores tile the domain, so stitching the cores reconstructs the
gradient everywhere while peak GPU memory is that of a single window.

All windows have the same shape (near the domain boundary they are shifted
inward rather than clipped), so the jit-compiled iceflow evaluators trace
exactly once.
"""

import math
import numpy as np
import tensorflow as tf

from igm.common import State


def _plan_1d(n, core, halo):
    """1D window plan: list of (w0, w1, c0, c1) with cores [c0,c1) tiling
    [0,n) in steps of `core`, and uniform windows [w0,w1) of width
    min(n, core + 2*halo) shifted inward at the domain boundary."""
    W = min(n, core + 2 * halo)
    K = max(1, math.ceil(n / core))
    plan = []
    for k in range(K):
        c0, c1 = k * core, min((k + 1) * core, n)
        w0 = max(0, min(c0 - halo, n - W))
        plan.append((w0, w0 + W, c0, c1))
    return plan


def plan_windows(ny, nx, core, halo):
    """2D window plan: list of ((wy0,wy1,cy0,cy1), (wx0,wx1,cx0,cx1))."""
    return [(r, c) for r in _plan_1d(ny, core, halo) for c in _plan_1d(nx, core, halo)]


def slice_state(state, ny, nx, y0, y1, x0, x1):
    """Shallow patch view of `state`: every tensor field whose two trailing
    dimensions match the (ny, nx) grid is sliced to the window; everything
    else (scalars, models, config-like objects) is copied by reference."""
    pstate = State()
    for name, value in vars(state).items():
        if name.startswith("_"):
            continue
        if isinstance(value, (tf.Tensor, tf.Variable)):
            shape = value.shape
            if shape.rank is not None and shape.rank >= 2 \
                    and shape[-2] == ny and shape[-1] == nx:
                value = value[..., y0:y1, x0:x1]
        elif isinstance(value, np.ndarray):
            if value.ndim >= 2 and value.shape[-2:] == (ny, nx):
                value = value[..., y0:y1, x0:x1]
        elif name == "x" and getattr(value, "shape", None) is not None:
            value = value[x0:x1]
        elif name == "y" and getattr(value, "shape", None) is not None:
            value = value[y0:y1]
        setattr(pstate, name, value)
    return pstate
