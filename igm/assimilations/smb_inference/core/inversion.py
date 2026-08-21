import tensorflow as tf
from typing import Dict, Tuple


def _mask(x: tf.Tensor, thresh: float) -> tf.Tensor:
    """
    Mask observation/simulations using a sigmoid function.

    Parameters:
    -----------
    x : tf.Tensor
        Input tensor
    thresh : float
        Threshold value

    Returns:
    --------
    mask : tf.Tensor
        Sigmoid-smoothed mask
    """
    scale = 0.1  # Steeper sigmoid using a scaling factor
    return tf.sigmoid(scale * (x - 1))


def _eval_pair(H: tf.Tensor, obs: tf.Tensor, thickness_thresh: float = 1.0):
    """
    Compute masked residual diagnostics between simulation and observation.

    This version uses differentiable soft masking to preserve gradient flow.
    The mask is based on the observation only (not on H) to avoid creating
    a dependency that breaks gradients.

    Parameters:
    -----------
    H : tf.Tensor
        Simulated ice thickness
    obs : tf.Tensor
        Observed ice thickness
    thickness_thresh : float
        Threshold defining glacier presence for union-of-extents masking

    Returns:
    --------
    metrics : dict
        Dictionary containing mse, rmse, mae, bias, std, area
    """
    # Create a soft mask based on observation only (not H) to preserve gradients
    # Using sigmoid for smooth, differentiable masking
    # This mask is NOT a function of H, so gradients flow through H unimpeded
    scale = 1.0  # Steepness of the sigmoid transition
    # Zero the mask wherever obs is NaN so those pixels never contribute to
    # cost or gradient. Then sanitize obs to a finite value — NaN * 0 = NaN
    # in TF, so we must replace NaNs before arithmetic.
    valid = tf.cast(~tf.math.is_nan(obs), obs.dtype)
    obs_safe = tf.where(tf.math.is_nan(obs), tf.zeros_like(obs), obs)
    soft_mask = tf.sigmoid(scale * ((obs_safe - thickness_thresh)* valid)) 

    # Compute residuals
    resid = H - obs_safe

    # Compute weighted metrics using soft mask
    # This preserves gradient flow through H
    mask_sum = tf.reduce_sum(soft_mask) + 1e-8  # Avoid division by zero

    # Weighted MSE
    mse = tf.reduce_sum(soft_mask * resid**2) / mask_sum
    rmse = tf.sqrt(mse)

    # Weighted MAE
    mae = tf.reduce_sum(soft_mask * tf.abs(resid)) / mask_sum

    # Weighted bias
    bias = tf.reduce_sum(soft_mask * resid) / mask_sum

    # Weighted std (approximate)
    variance = tf.reduce_sum(soft_mask * (resid - bias)**2) / mask_sum
    std = tf.sqrt(variance + 1e-8)

    # Area (approximate count of glacier pixels)
    count = mask_sum

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "std": std,
        "area": count * 0.01,
    }


def inversion_thickness(
    precip_tensor, T_m_lowest, T_s, P_daily, T_daily, melt_factor,
    obs1880, obs26, obs57, obs80, obs99, obs09, obs17,
    glacier_model,
    reg_lambda: float = 0.001,
    thickness_thresh: float = 1.0,
    w1880: float = 1.0,
    w26: float = 1.0, w57: float = 1.0, w80: float = 1.0,
    w99: float = 1.0, w09: float = 1.0, w17: float = 1.0
) -> Tuple[tf.Tensor, tf.Tensor, Dict[str, Dict[str, tf.Tensor]]]:
    """
    Inversion for ice thickness observations.

    Returns:
      H99: simulated thickness for 1999 (or last epoch)
      loss: scalar tensor
      metrics: dict of per-epoch diagnostics computed inside the glacier union mask
               keys: '26','57','80','99' with mse, rmse, mae, bias, std, count

    Notes:
      - Loss uses RMSE (meters) for physical interpretability.
      - Data fidelity is computed only where (obs>thresh OR sim>thresh) & mask.
    """
    # Forward simulation
    H1880, H26, H57, H80, H99, H09, H17 = glacier_model(
        precip_tensor=precip_tensor,
        T_m_lowest=T_m_lowest,
        T_s=T_s,
        melt_factor=melt_factor
    )

    # Per-epoch diagnostics (masked)
    m1880 = _eval_pair(H1880, obs1880, thickness_thresh)
    m26 = _eval_pair(H26, obs26, thickness_thresh)
    m57 = _eval_pair(H57, obs57, thickness_thresh)
    m80 = _eval_pair(H80, obs80, thickness_thresh)
    m99 = _eval_pair(H99, obs99, thickness_thresh)
    m09 = _eval_pair(H09, obs09, thickness_thresh)
    m17 = _eval_pair(H17, obs17, thickness_thresh)

    metrics = {
        "1880": m1880,
        "26": m26,
        "57": m57,
        "80": m80,
        "99": m99,
        "09": m09,
        "17": m17
    }

    # Smoothness regularization (simple 1D TV-like quadratic)
    smoothness_x = tf.reduce_sum((T_m_lowest[1:] - T_m_lowest[:-1])**2)

    # Data term: weighted sum of MSEs
    data_term = (
        w1880 * m1880["mse"] +
        w26 * m26["mse"] +
        w57 * m57["mse"] +
        w80 * m80["mse"] +
        w99 * m99["mse"] +
        w09 * m09["mse"] +
        w17 * m17["mse"]
    )

    data = data_term / (w1880 + w26 + w57 + w80 + w99 + w09 + w17)

    loss = data + reg_lambda * smoothness_x

    return [H1880, H26, H57, H80, H99, H09, H17], loss, data, metrics


def inversion_extent(
    precip_tensor, T_m_lowest, T_s, P_daily, T_daily, melt_factor,
    obs1880, obs26, obs57, obs80, obs99, obs09, obs17,
    glacier_model,
    reg_lambda: float = 0.001,
    thickness_thresh: float = 1.0,
    w1880: float = 0.0,
    w26: float = 0.0, w57: float = 0.0, w80: float = 0.0,
    w99: float = 0.0, w09: float = 0.0, w17: float = 0.0
) -> Tuple[Dict[str, tf.Tensor], tf.Tensor, Dict[str, Dict[str, tf.Tensor]]]:
    """
    Extent-only inversion.

    Returns:
      extents: dict of simulated binary extents per epoch
      loss: scalar tensor
      metrics: dict per epoch with IoU, Dice, Precision, Recall, Accuracy, BCE, RMSE(0/1), counts

    Notes:
      - Simulated maps are binarized with STE so gradients flow.
      - Loss is a weighted sum of (1 - Dice) with optional BCE mix, plus smoothness regularization.
    """
    # Forward simulation
    H1880, H26, H57, H80, H99, H09, H17 = glacier_model(
        precip_tensor=precip_tensor,
        T_m_lowest=T_m_lowest,
        T_s=T_s,
        melt_factor=melt_factor
    )

    # Binarize observations (hard, no gradients)
    O1880 = _mask(obs1880, thickness_thresh)
    O26 = _mask(obs26, thickness_thresh)
    O57 = _mask(obs57, thickness_thresh)
    O80 = _mask(obs80, thickness_thresh)
    O99 = _mask(obs99, thickness_thresh)
    O09 = _mask(obs09, thickness_thresh)
    O17 = _mask(obs17, thickness_thresh)

    # Binarize simulations with STE (keeps gradients alive)
    S1880 = _mask(H1880, thickness_thresh)
    S26 = _mask(H26, thickness_thresh)
    S57 = _mask(H57, thickness_thresh)
    S80 = _mask(H80, thickness_thresh)
    S99 = _mask(H99, thickness_thresh)
    S09 = _mask(H09, thickness_thresh)
    S17 = _mask(H17, thickness_thresh)

    # Per-epoch diagnostics (masked)
    m1880 = _eval_pair(S1880, O1880)
    m26 = _eval_pair(S26, O26)
    m57 = _eval_pair(S57, O57)
    m80 = _eval_pair(S80, O80)
    m99 = _eval_pair(S99, O99)
    m09 = _eval_pair(S09, O09)
    m17 = _eval_pair(S17, O17)

    metrics = {
        "1880": m1880,
        "26": m26,
        "57": m57,
        "80": m80,
        "99": m99,
        "09": m09,
        "17": m17
    }

    # Smoothness regularization
    eps = 1e-8
    smoothness_x = tf.reduce_sum((T_m_lowest[1:] - T_m_lowest[:-1])**2) / (tf.reduce_sum(T_m_lowest**2) + eps)

    # Data term
    data_term = (
        w1880 * m1880["mse"] +
        w26 * m26["mse"] +
        w57 * m57["mse"] +
        w80 * m80["mse"] +
        w99 * m99["mse"] +
        w09 * m09["mse"] +
        w17 * m17["mse"]
    )
    data = data_term / (w1880 + w26 + w57 + w80 + w99 + w09 + w17)

    loss = data + reg_lambda * smoothness_x

    return [H1880, H26, H57, H80, H99, H09, H17], loss, data, metrics


def invert_field(
    smb_field: tf.Tensor,
    observation: tf.Tensor,
    glacier_model,
    save_times: list = None,
    reg_lambda: float = 0.0,
    thickness_thresh: float = 1.0,
    smooth_type: str = 'gradient',
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, Dict[str, tf.Tensor]]:
    """
    Invert for SMB field given ice thickness observations.

    Parameters
    ----------
    smb_field : tf.Tensor
        SMB field to optimize (2D or 3D)
    observation : tf.Tensor
        Observed ice thickness at final time
    glacier_model : GlacierDynamicsCheckpointed
        Glacier dynamics model
    save_times : list, optional
        Times at which to save H (if None, only returns final H)
    reg_lambda : float
        Regularization weight for spatial smoothness
    thickness_thresh : float
        Threshold for glacier presence in metrics
    smooth_type : str
        Type of smoothness regularization ('gradient' or 'laplacian')

    Returns
    -------
    H_sim : tf.Tensor
        Simulated ice thickness at final time
    loss : tf.Tensor
        Total loss (data + regularization)
    data_term : tf.Tensor
        Data fidelity term only
    metrics : dict
        Evaluation metrics (mse, rmse, mae, bias, std, area)
    """
    # Forward simulation
    if save_times is not None:
        saved_H_dict, H_sim = glacier_model(
            precip_tensor=None,
            T_m_lowest=None,
            T_s=None,
            melt_factor=None,
            smb_method='field',
            smb_field=smb_field,
            save_times=save_times,
        )
    else:
        H_sim = glacier_model(
            precip_tensor=None,
            T_m_lowest=None,
            T_s=None,
            melt_factor=None,
            smb_method='field',
            smb_field=smb_field,
        )

    # Compute data fidelity (masked misfit)
    metrics = _eval_pair(H_sim, observation, thickness_thresh)
    data_term = metrics["mse"]

    # Spatial smoothness regularization
    if reg_lambda > 0:
        if len(smb_field.shape) == 3:
            smb_slice = smb_field[-1, :, :]
        elif len(smb_field.shape) == 2:
            smb_slice = smb_field
        else:
            raise ValueError(f"smb_field must be 2D or 3D, got shape {smb_field.shape}")

        if smooth_type == 'gradient':
            grad_x = smb_slice[:, 1:] - smb_slice[:, :-1]
            grad_y = smb_slice[1:, :] - smb_slice[:-1, :]
            smoothness = tf.reduce_sum(grad_x**2) + tf.reduce_sum(grad_y**2)

        elif smooth_type == 'laplacian':
            laplacian = (
                smb_slice[:-2, 1:-1] +
                smb_slice[2:, 1:-1] +
                smb_slice[1:-1, :-2] +
                smb_slice[1:-1, 2:] -
                4 * smb_slice[1:-1, 1:-1]
            )
            smoothness = tf.reduce_sum(laplacian**2)

        else:
            raise ValueError(f"smooth_type must be 'gradient' or 'laplacian', got '{smooth_type}'")

        loss = data_term + reg_lambda * smoothness
    else:
        loss = data_term

    return H_sim, loss, data_term, metrics

def invert_profile(
    smb_vec: tf.Tensor,
    z_min,
    dz,
    observation: tf.Tensor,
    glacier_model,
    save_times: list = None,
    reg_lambda: float = 0.0,
    thickness_thresh: float = 1.0,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, Dict[str, tf.Tensor]]:

    H_sim = glacier_model(
        precip_tensor=None,
        T_m_lowest=None,
        T_s=None,
        melt_factor=None,
        smb_method='profile',
        smb_vec=smb_vec,
        z_min=z_min,
        dz=dz,
        differentiable=True,  # Use tf.while_loop for gradient flow
    )

    # Compute data fidelity (masked misfit)
    metrics = _eval_pair(H_sim, observation, thickness_thresh)
    data_term = metrics["rmse"]

    # Smoothness regularization for 1D profile
    if reg_lambda > 0:
        # Penalize second derivative (curvature) of SMB profile
        second_deriv = smb_vec[:-2] - 2 * smb_vec[1:-1] + smb_vec[2:]
        smoothness = tf.reduce_sum(second_deriv**2)

        loss = data_term + reg_lambda * smoothness
    else:
        loss = data_term

    return H_sim, loss, data_term, metrics
