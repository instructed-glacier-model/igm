import tensorflow as tf
from typing import Union


def apply_lapse_rate(topography, T_m_lowest):
    """
    Apply temperature lapse rate to compute temperature field from topography.

    Parameters:
    -----------
    topography : tf.Tensor
        Elevation field (ny, nx) in meters
    T_m_lowest : tf.Tensor or float
        Mean annual temperature at the lowest point (°C)

    Returns:
    --------
    T_m : tf.Tensor
        Temperature field (ny, nx) in °C
    """
    lapse_rate = 7.0 / 1000.0  # 7°C/km
    min_altitude = tf.reduce_min(topography)
    delta_alt = topography - min_altitude

    T_m = T_m_lowest - lapse_rate * delta_alt

    return T_m


def apply_lapse_rate_daily(topography: tf.Tensor, T_m_lowest: tf.Tensor) -> tf.Tensor:
    """
    Apply temperature lapse rate to daily temperature data.

    Parameters:
    -----------
    topography : tf.Tensor
        Elevation field (H, W) in meters
    T_m_lowest : tf.Tensor
        Daily temperatures at the lowest altitude (366, 1) or (366,)

    Returns:
    --------
    T_m : tf.Tensor
        Daily temperature fields (366, H, W)
    """
    lapse_rate = tf.constant(7.0 / 1000.0, dtype=topography.dtype)  # 7 °C/km

    # Altitude differences relative to minimum
    min_altitude = tf.reduce_min(topography)
    delta_alt = topography - min_altitude  # (H, W)

    # Shape to enable broadcasting
    # T_m_lowest: (366, 1, 1), delta_alt: (1, H, W)
    T_m_lowest = tf.cast(tf.squeeze(T_m_lowest, axis=-1), dtype=topography.dtype)
    T_m_lowest = tf.reshape(T_m_lowest, [-1, 1, 1])
    delta_alt = tf.expand_dims(delta_alt, axis=0)

    # Broadcasted computation -> (366, H, W)
    T_m = T_m_lowest - lapse_rate * delta_alt
    return T_m


def smooth_piecewise(x: tf.Tensor, w: float = 0.1, eps: float = 1e-6) -> tf.Tensor:
    """
    Smooth piecewise function for clipping values between -1 and 1.

    Parameters:
    -----------
    x : tf.Tensor
        Input tensor
    w : float
        Width of smoothing region
    eps : float
        Small epsilon for numerical stability

    Returns:
    --------
    y : tf.Tensor
        Smoothed output
    """
    one = tf.constant(1.0 - eps, dtype=x.dtype)
    mone = -one

    left_p = (x + 1.0 + w) / (2.0 * w)          # (-1-w , -1+w)
    right_p = (x - (1.0 - w)) / (2.0 * w)        # ( 1-w ,  1+w)

    left_poly = mone + w * tf.square(left_p)
    right_poly = (-w) * tf.square(right_p) + 2.0 * w * right_p + (1.0 - w)

    y = tf.where(x <= -1.0 - w, mone,
         tf.where(x >= 1.0 + w, one,
           tf.where(x < -1.0 + w, left_poly,
             tf.where(x > 1.0 - w, right_poly, x))))
    return y


def compute_integral_positive_temperature(T_m, T_s):
    """
    Compute the integral of T_abl(t) over the period where T_abl > 0.

    Parameters:
    -----------
    T_m : tf.Tensor
        Mean annual temperature field (ny, nx) in °C
    T_s : tf.Tensor or float
        Seasonal temperature amplitude in °C

    Returns:
    --------
    integral : tf.Tensor
        Positive temperature integral
    """
    A = 12.0  # months
    pi = tf.constant(3.141592653589793, dtype=T_m.dtype)
    clipped = smooth_piecewise(T_m / T_s)
    return T_m * (A - (A / pi) * tf.acos(clipped)) + (T_s * A / pi) * tf.sqrt(1 - tf.square(clipped))


def compute_negative_temperature_ratio(T_m, T_s):
    """
    Compute the ratio of the year when the temperature is negative.

    Parameters:
    -----------
    T_m : tf.Tensor
        Mean annual temperature field (ny, nx) in °C
    T_s : tf.Tensor or float
        Seasonal temperature amplitude in °C

    Returns:
    --------
    ratio : tf.Tensor
        Negative temperature ratio (values between 0 and 1)
    """
    pi = tf.constant(3.141592653589793, dtype=T_m.dtype)
    return (1.0 / pi) * tf.acos(smooth_piecewise(T_m / T_s))


def _align_time_vector(vec: tf.Tensor, ref: tf.Tensor, time_dim: int) -> tf.Tensor:
    """
    If vec is a time vector of shape (T,), reshape to broadcast along ref's time_dim.

    Parameters:
    -----------
    vec : tf.Tensor
        Time vector to align
    ref : tf.Tensor
        Reference tensor to align with
    time_dim : int
        Time dimension in ref

    Returns:
    --------
    vec : tf.Tensor
        Reshaped vector for broadcasting
    """
    time_dim = time_dim % len(ref.shape)
    if len(vec.shape) == 0:
        # scalar: fine as-is
        return vec
    if len(vec.shape) == 1 and vec.shape[0] == ref.shape[time_dim]:
        view_shape = [1] * len(ref.shape)
        view_shape[time_dim] = vec.shape[0]  # put T on the time axis
        return tf.reshape(vec, view_shape)
    # Otherwise assume it's already broadcastable
    return vec


def pdd_sum_daily(
    T_daily: tf.Tensor,
    time_dim: int = 0,
    threshold_c: Union[float, tf.Tensor] = 0.2
) -> tf.Tensor:
    """
    Compute Positive Degree Days: sum_t max(T - threshold, 0).

    Parameters:
    -----------
    T_daily : tf.Tensor
        Daily temperature field (T, X, Y) or with extra batch dims
    time_dim : int
        Dimension to reduce over (time)
    threshold_c : float or tf.Tensor
        Temperature threshold in °C

    Returns:
    --------
    pdd : tf.Tensor
        Positive degree days (spatial dimensions)
    """
    time_dim = time_dim % len(T_daily.shape)

    if isinstance(threshold_c, tf.Tensor):
        thr = tf.cast(threshold_c, dtype=T_daily.dtype)
        # If it's a (T,) vector, align to time_dim
        if len(thr.shape) == 1 and thr.shape[0] == T_daily.shape[time_dim]:
            thr = _align_time_vector(thr, T_daily, time_dim)
    else:
        thr = tf.constant(threshold_c, dtype=T_daily.dtype)

    pdd = tf.maximum(T_daily - thr, 0.0)
    return tf.reduce_sum(pdd, axis=time_dim)


def accumulation_from_daily(
    P_daily: tf.Tensor,
    T_daily: tf.Tensor,
    time_dim: int = 0,
    snow_temp_c: Union[float, tf.Tensor] = 0.0
) -> tf.Tensor:
    """
    Compute solid precipitation accumulation: sum_t P where T <= snow_temp_c.

    Parameters:
    -----------
    P_daily : tf.Tensor
        Daily precipitation (m w.e. per day), shape (T,) or broadcastable to T_daily
    T_daily : tf.Tensor
        Daily temperature field (T, X, Y)
    time_dim : int
        Dimension to reduce over (time)
    snow_temp_c : float or tf.Tensor
        Snow temperature threshold in °C

    Returns:
    --------
    acc : tf.Tensor
        Accumulated solid precipitation (spatial dimensions)
    """
    time_dim = time_dim % len(T_daily.shape)

    if isinstance(snow_temp_c, tf.Tensor):
        thr = tf.cast(snow_temp_c, dtype=T_daily.dtype)
    else:
        thr = tf.constant(snow_temp_c, dtype=T_daily.dtype)

    # Cold-day mask (same shape as T_daily)
    mask_cold = tf.cast(T_daily <= thr, dtype=T_daily.dtype)

    # Align P_daily to broadcast along T_daily
    P_daily = tf.cast(P_daily, dtype=T_daily.dtype)
    if len(P_daily.shape) == 1 and P_daily.shape[0] == T_daily.shape[time_dim]:
        P_daily = _align_time_vector(P_daily, T_daily, time_dim)

    # Multiply by mask
    acc = P_daily * mask_cold
    return tf.reduce_sum(acc, axis=time_dim)
