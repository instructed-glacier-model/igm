import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
import tensorflow as tf
from collections.abc import Sequence

def _as_list(x):
    if isinstance(x, str):
        return [x]
    if isinstance(x, Sequence):
        return list(x)
    return [x]


def cell_area_like(dx: tf.Tensor, like: tf.Tensor) -> tf.Tensor:
    """
    Cell area ΔA inferred from dx. Assumes square cells: ΔA = dx^2.
    """
    dx = tf.cast(dx, like.dtype)
    return tf.ones_like(like, dtype=like.dtype) * (dx * dx)

def masked_integral(x: tf.Tensor, mask: tf.Tensor, dx: tf.Tensor) -> tf.Tensor:
    """
    Discrete approximation of ∫ x dA over `mask`, with dA = dx^2.
    """
    area = cell_area_like(dx, x)
    return tf.reduce_sum(tf.where(mask, x * area, tf.zeros_like(x)))


def masked_area(mask: tf.Tensor, dx: tf.Tensor, like: tf.Tensor) -> tf.Tensor:
    """
    Area covered by `mask`, using the same dA = dx^2 convention as masked_integral.
    """
    one = tf.ones_like(like, dtype=like.dtype)
    return masked_integral(one, tf.cast(mask, tf.bool), dx)


def _safe_loss_scale(value: tf.Tensor, dtype: tf.dtypes.DType, floor: float = 1e-6) -> tf.Tensor:
    value = tf.cast(value, dtype)
    return tf.maximum(tf.abs(value), tf.cast(floor, dtype))

    
def _initialize_inverted_fields(cfg, state, dtype) -> set[str]:
    """
    Only initialize fields that are actually inverted for.
    Non-inverted iceflow inputs are cast to ``dtype`` in place so the
    network's input stack is dtype-homogeneous.
    """
    inverted = {
        str(item["name"])
        for item in cfg.assimilations.field_inversion.variables
    }

    # These are not inversion variables, just cast once for consistency.
    state.uvelsurfobs = tf.cast(state.uvelsurfobs, dtype=dtype)
    state.vvelsurfobs = tf.cast(state.vvelsurfobs, dtype=dtype)
    state.usurf = tf.cast(state.usurf, dtype=dtype)
    state.dX = tf.cast(state.dX, dtype=dtype)

    if "thk" in inverted and not cfg.assimilations.field_inversion.use_existing_thk:
        thk0 = initial_thickness(
            s=state.usurf,
            u=state.uvelsurfobs,
            v=state.vvelsurfobs,
            mask=state.icemask,
            dx=state.dX[0, 0],
            dy=state.dX[0, 0],
        )
        state.thk = tf.convert_to_tensor(thk0, dtype=dtype)

    if "tau_ref" in inverted:
        state.tau_ref = (
            tf.ones_like(state.usurf, dtype=dtype)
            * tf.cast(cfg.processes.iceflow.physics.sliding.tau_ref, dtype)
        )

    if "arrhenius" in inverted:
        state.arrhenius = (
            tf.ones_like(state.usurf, dtype=dtype)
            * tf.cast(cfg.processes.iceflow.physics.viscosity.arrhenius, dtype)
        )

    # convert dtype on any other input fields to avoid dtype mismatch
    for field_name in cfg.processes.iceflow.unified.inputs:
        value = getattr(state, field_name)
        if hasattr(value, "dtype") and value.dtype != dtype:
            setattr(state, field_name, tf.cast(value, dtype=dtype))

    return inverted

def initial_thickness(
    s,                 # surface elevation [m], shape (Ny, Nx)
    u, v,              # surface velocities (same shape), units set by speed_units
    mask,              # boolean ice mask (True = ice)
    dx, dy,            # grid spacing [m]
    *,
    speed_units="m/yr",   # "m/s" or "m/yr"
    A=1e-24,              # Pa^-3 s^-1 (tune 1e-24 .. 5e-24)
    n=3,
    rho=917.0, g=9.81,
    kappa=0.8,            # surface->depth-avg factor
    slope_floor=1e-3,     # dimensionless slope floor
    U0=100.0,             # m/yr threshold for blending
    L=5000.0,             # [m] distance length scale
    smooth_sigma_cells=2, # Gaussian sigma in grid cells
    Hmin=10.0, Hmax=3000.0
):

    """
    Computes an initial thickness field by blending between:
    1) SIA-based thickness (reliable where speed is high, but can be very wrong where speed is low or surface slope is shallow)
    2) Distance-based thickness (reliable near the margin, but can be very wrong in the interior)
    The blending is based on speed: where speed >> U0, trust SIA; where speed << U0, trust distance; in between, blend smoothly. The final field is then smoothed and clamped to [Hmin, Hmax].
    """

    s = np.asarray(s)
    u = np.asarray(u)
    v = np.asarray(v)
    mask = np.asarray(mask)

    if s.shape != u.shape or s.shape != v.shape:
        raise ValueError(f"initial_thickness: s, u, v must have the same shape. Got {s.shape}, {u.shape}, {v.shape}.")

    if not (np.isfinite(dx) and np.isfinite(dy) and dx > 0.0 and dy > 0.0):
        raise ValueError(f"initial_thickness: dx and dy must be finite and > 0. Got dx={dx}, dy={dy}.")

    if not (np.isfinite(A) and A > 0.0):
        raise ValueError(f"initial_thickness: A must be finite and > 0. Got A={A}.")

    if not (np.isfinite(n) and n > 0.0):
        raise ValueError(f"initial_thickness: n must be finite and > 0. Got n={n}.")

    # --- sanitize mask: treat NaN/inf as non-ice; then cast to bool
    if mask.dtype != bool:
        mask = np.where(np.isfinite(mask), mask, 0)
        mask = mask.astype(bool)
    else:
        mask = mask.copy()

    # --- fill NaNs in surface elevation using nearest finite neighbor (so gradients are defined)
    if not np.all(np.isfinite(s)):
        finite_s = np.isfinite(s)
        if not np.any(finite_s):
            raise ValueError("initial_thickness: surface elevation 's' is entirely non-finite.")
        # nearest-neighbor fill for non-finite cells
        _, inds = distance_transform_edt(~finite_s, return_indices=True)
        s = s[tuple(inds)]

    # --- speed magnitude (use speed as the robust quantity; replace non-finite speeds with tiny > 0)
    speed = np.hypot(u, v)

    sec_per_year = 365.25 * 24 * 3600.0
    su = speed_units.lower()
    if su in ["m/yr", "m/years", "m/year"]:
        speed = speed / sec_per_year
    elif su not in ["m/s", "mps"]:
        raise ValueError("speed_units must be 'm/s' or 'm/yr'.")

    # tiny positive speed used ONLY to replace non-finite values (and to protect denominators)
    eps_speed = 1e-12  # m/s ~ 3e-5 m/yr
    speed = np.where(np.isfinite(speed), speed, eps_speed)
    speed = np.where(speed > 0.0, speed, 0.0)  # keep true zeros if present (stagnant ice is allowed)

    # --- surface slope |∇s|
    ds_dy, ds_dx = np.gradient(s, dy, dx)
    slope = np.hypot(ds_dx, ds_dy)

    # "maximum" does not clear NaNs, so explicitly fix non-finite slope before flooring
    slope = np.where(np.isfinite(slope), slope, slope_floor)
    slope = np.maximum(slope, slope_floor)

    # --- SIA-based thickness (depth-avg speed ~ k H^{n+1} slope^n)
    k = (2.0 * A / (n + 2.0)) * (rho * g) ** n
    if not (np.isfinite(k) and k > 0.0):
        raise ValueError(f"initial_thickness: computed k must be finite and > 0. Got k={k} (check A, n, rho, g).")

    slope_n = np.power(slope, n)
    slope_n = np.where(np.isfinite(slope_n), slope_n, np.inf)  # if it blows up, ratio->0 below

    denom = k * slope_n
    # protect against denom <= 0 or non-finite; force safe denom so divide won't yield NaN/inf
    denom = np.where((np.isfinite(denom) & (denom > 0.0)), denom, np.inf)

    num = kappa * speed
    num = np.where(np.isfinite(num), num, eps_speed)
    num = np.maximum(num, 0.0)

    ratio = num / denom
    ratio = np.where(np.isfinite(ratio) & (ratio >= 0.0), ratio, 0.0)

    H_sia = np.power(ratio, 1.0 / (n + 1.0))
    H_sia = np.where(np.isfinite(H_sia), H_sia, Hmin)
    H_sia = np.clip(H_sia, Hmin, Hmax)
    H_sia[~mask] = 0.0

    # --- distance inside ice to nearest non-ice
    d = distance_transform_edt(mask, sampling=(dy, dx))
    d = np.where(np.isfinite(d), d, 0.0)

    sia_vals = H_sia[mask]
    finite_pos = sia_vals[np.isfinite(sia_vals) & (sia_vals > 0)]
    beta = np.percentile(finite_pos, 90) if finite_pos.size else 500.0
    if not np.isfinite(beta) or beta <= 0.0:
        beta = 500.0

    # guard L to avoid 0/0 at the margin; if user gives L<=0 or non-finite, fall back to ~one grid cell
    if not (np.isfinite(L) and L > 0.0):
        L_eff = max(float(min(dx, dy)), 1.0)
    else:
        L_eff = float(L)

    H_dist = beta * (1.0 - np.exp(-d / L_eff))
    H_dist = np.where(np.isfinite(H_dist), H_dist, Hmin)
    H_dist = np.clip(H_dist, Hmin, Hmax)

    # --- Blend by speed reliability
    U0_mps = U0 / sec_per_year  # m/s
    if not (np.isfinite(U0_mps) and U0_mps > 0.0):
        U0_mps = eps_speed  # avoids 0/0 if U0 is zero or non-finite

    denom_w = speed + U0_mps
    denom_w = np.where(denom_w > 0.0, denom_w, eps_speed)
    w = speed / denom_w
    w = np.where(np.isfinite(w), w, 0.0)

    H0 = w * H_sia + (1.0 - w) * H_dist
    H0 = np.where(np.isfinite(H0), H0, Hmin)

    # --- Smooth & clamp
    if smooth_sigma_cells and smooth_sigma_cells > 0:
        H0 = gaussian_filter(H0, sigma=smooth_sigma_cells, mode="nearest")
        H0 = np.where(np.isfinite(H0), H0, Hmin)

    H0 = np.clip(H0, Hmin, Hmax)
    H0[~mask] = 0.0

    if np.any(np.isnan(H0)):
        raise ValueError("initial_thickness: NaN values encountered in computed thickness (after safety checks).")

    return H0
