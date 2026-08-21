from .climate import (
    apply_lapse_rate, compute_negative_temperature_ratio,
    compute_integral_positive_temperature, accumulation_from_daily,
    pdd_sum_daily, apply_lapse_rate_daily
)
import tensorflow as tf


# Define SMB parameters directly
smb_oggm_wat_density = 1000.0  # kg/m³
smb_oggm_ice_density = 910.0   # kg/m³


def cosine_temperature_series(ttot, t_freq, T_high=9.0, T_low=7.0):
    """
    Generate a cosine temperature series.

    Returns a tensor of length N = ttot / t_freq with one full cosine cycle:
    starts at T_high, dips to T_low mid-cycle, returns to T_high at the end.

    Parameters:
    -----------
    ttot : float
        Total time period
    t_freq : float
        Frequency of updates
    T_high : float
        Maximum temperature
    T_low : float
        Minimum temperature

    Returns:
    --------
    temps : tf.Tensor
        Temperature series
    """
    N = int(round(float(ttot) / float(t_freq)))
    if N < 2:
        raise ValueError("ttot/t_freq must be >= 2 to form a cosine cycle.")
    mean = 0.5 * (T_high + T_low)
    amp = 0.5 * (T_high - T_low)
    theta = tf.linspace(0.0, 2 * 3.141592653589793, N)
    return mean + amp * tf.cos(theta)


def update_smb_PDD(
    Z_topo,
    precipitation=None,
    T_m_lowest=None,
    T_s=None,
    melt_factor=2.0 / 360,
    smb_wat_rho: float = 1000.0,
    smb_ice_rho: float = 910.0,
    P_daily=None,
    T_daily=None,
    time_dim: int = 0,
    positive_degree_threshold_c: float = 1.0,
):
    """
    Compute SMB (m ice eq / year) using Positive Degree Day (PDD) method.

    Modes:
      - Daily mode: if P_daily and T_daily are provided -> direct PDD and
        accumulation from daily data.
      - Parametric mode: otherwise use precipitation, T_m_lowest, T_s with
        the OGGM-style formulas.

    Parameters:
    -----------
    Z_topo : tf.Tensor
        Surface elevation (ny, nx) in meters
    precipitation : tf.Tensor or float, optional
        Annual precipitation (m w.e./yr) for parametric mode
    T_m_lowest : tf.Tensor or float, optional
        Mean annual temperature at lowest point (°C) for parametric mode
    T_s : tf.Tensor or float, optional
        Seasonal amplitude parameter for parametric mode
    melt_factor : tf.Tensor or float
        Ablation factor
    smb_wat_rho : float
        Water density (kg/m³)
    smb_ice_rho : float
        Ice density (kg/m³)
    P_daily : tf.Tensor, optional
        Daily precipitation for daily mode
    T_daily : tf.Tensor, optional
        Daily temperature for daily mode
    time_dim : int
        Time dimension for daily inputs
    positive_degree_threshold_c : float
        Temperature threshold for PDD calculation

    Returns:
    --------
    smb : tf.Tensor
        Surface mass balance in m ice eq per year
    """
    # --- DAILY MODE ---
    if (P_daily is not None) and (T_daily is not None):
        # Temperature field via lapse-rate
        T_daily_topo = apply_lapse_rate_daily(Z_topo, T_daily)

        # Accumulation: sum precip where T <= 0°C
        accumulation_wat = accumulation_from_daily(P_daily, T_daily_topo, time_dim=time_dim)

        # PDD (with threshold 1°C by default)
        pdd = pdd_sum_daily(T_daily_topo, time_dim=time_dim, threshold_c=positive_degree_threshold_c)

        # Ablation in water equivalent
        melt_factor_tensor = tf.cast(melt_factor, dtype=pdd.dtype)
        ablation_wat = melt_factor_tensor * pdd

    else:
        # --- PARAMETRIC MODE ---
        if any(x is None for x in (precipitation, T_m_lowest, T_s)):
            raise ValueError(
                "Parametric mode requires 'precipitation', 'T_m_lowest', and 'T_s' "
                "when daily inputs are not provided."
            )

        # Build fields on the same dtype as Z_topo
        precipitation = tf.cast(precipitation, dtype=Z_topo.dtype)
        T_s = tf.cast(T_s, dtype=Z_topo.dtype)

        # Temperature field via lapse-rate
        T_m = apply_lapse_rate(Z_topo, T_m_lowest)

        # Accumulation fraction of year with negative temps, times annual precip
        neg_ratio = compute_negative_temperature_ratio(T_m, T_s)
        accumulation_wat = precipitation * neg_ratio

        # Positive degree "integral"
        p_int = compute_integral_positive_temperature(T_m, T_s)

        # Ablation
        melt_factor_tensor = tf.cast(melt_factor, dtype=p_int.dtype)
        ablation_wat = melt_factor_tensor * p_int

    # Convert water-equivalent to ice-equivalent and combine
    rho_w = tf.constant(smb_wat_rho, dtype=Z_topo.dtype)
    rho_i = tf.constant(smb_ice_rho, dtype=Z_topo.dtype)

    smb = (accumulation_wat - ablation_wat) * (rho_w / rho_i)

    return smb


def update_smb_ELA(
    Z_topo,
    ELA,
    grad_b,
    b_max,
):
    """
    Compute SMB (m ice eq / year) using simple ELA (Equilibrium Line Altitude) method.

    The SMB is linearly proportional to the elevation difference from the ELA,
    capped at a maximum accumulation rate.

    SMB = min(grad_b * (Z - ELA), b_max)

    Parameters:
    -----------
    Z_topo : tf.Tensor
        Surface elevation (ny, nx) in meters
    ELA : tf.Tensor or float
        Equilibrium Line Altitude (m)
    grad_b : tf.Tensor or float
        Mass balance gradient (m ice eq / m elevation)
    b_max : tf.Tensor or float
        Maximum accumulation rate (m ice eq / year)

    Returns:
    --------
    smb : tf.Tensor
        Surface mass balance in m ice eq per year (same shape as Z_topo)
    """
    dtype = Z_topo.dtype

    # Convert inputs to tensors on the same dtype as Z_topo
    ELA = tf.cast(ELA, dtype=dtype)
    grad_b = tf.cast(grad_b, dtype=dtype)

    # Compute SMB: grad_b * (Z - ELA)
    smb = grad_b * (Z_topo - ELA)

    return smb


def update_smb_profile(
    Z_topo,
    smb_vec,
    z_min,
    dz,
):
    """
    Compute SMB (m ice eq / year) from an elevation-based profile using
    piecewise-linear (triangular hat basis) interpolation.

    This method maps a 1D SMB profile smb(z) to the 2D topography SMB(x,y)
    by interpolating based on elevation. The interpolation is AD-friendly
    and mask-free.

    Parameters:
    -----------
    Z_topo : tf.Tensor
        Surface elevation (m), shape (H, W) or (..., H, W)
    smb_vec : tf.Tensor
        1D SMB profile values (m ice eq / year) at discrete elevations.
        Shape (N,) where N is the number of elevation bins.
        smb_vec[i] corresponds to elevation z_min + i * dz
    z_min : tf.Tensor or float
        Minimum elevation (m) corresponding to smb_vec[0]
    dz : tf.Tensor or float
        Elevation spacing (m) between consecutive smb_vec values

    Returns:
    --------
    smb : tf.Tensor
        Surface mass balance in m ice eq per year (same shape as Z_topo)

    Example:
    --------
    >>> # Create SMB profile from -15 to 5 m/yr over elevation range
    >>> zmin = tf.reduce_min(topo)
    >>> zmax = tf.reduce_max(topo)
    >>> dz = 100.0  # 100m bins
    >>> N = int(tf.math.ceil((zmax - zmin) / dz).numpy())
    >>> smb_vec = tf.linspace(-15.0, 5.0, N)
    >>> smb_2d = update_smb_profile(topo, smb_vec, zmin, dz)
    """
    dtype = Z_topo.dtype

    # Convert inputs to tensors on the same dtype as Z_topo
    smb_vec = tf.cast(smb_vec, dtype=dtype)
    z_min = tf.cast(z_min, dtype=dtype)
    dz = tf.cast(dz, dtype=dtype)

    # Compute continuous index coordinate
    t = (Z_topo - z_min) / dz

    # Create index array for all SMB values
    N = smb_vec.shape[0]
    k = tf.cast(tf.range(N), dtype=dtype)

    # Compute triangular (hat) basis weights
    # Each point has non-zero weight for the two nearest elevation bins
    # Weight = relu(1 - |t - k|), which is 1 at exact bin and 0 beyond ±1 bin
    w = tf.nn.relu(1.0 - tf.abs(tf.expand_dims(t, axis=-1) - k))

    # Weighted sum over all elevation bins
    smb = tf.reduce_sum(w * smb_vec, axis=-1)

    return smb


# Backward compatibility alias
update_smb = update_smb_PDD
