#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np  
import tensorflow as tf 
import matplotlib.pyplot as plt
from igm.utils.math.getmag import getmag
from igm.utils.grad.compute_divflux import compute_divflux
from scipy import stats 
from igm.processes.time.time import compute_dt_from_cfl
from igm.processes.iceflow.emulate.emulated import update_iceflow_emulated
from igm.utils.grad.compute_divflux_slope_limiter import compute_divflux_slope_limiter

def compute_rms_std_optimization(state, i):
    I = state.icemaskobs > 0.5

    if i == 0:
        state.rmsthk = []
        state.stdthk = []
        state.rmsvel = []
        state.stdvel = []
        state.rmsusurf = []
        state.stdusurf = []
        state.rmsdiv = []
        state.stddiv = []
        state.vol = []

    if hasattr(state, "thkobs"):
        ACT = ~tf.math.is_nan(state.thkobs)
        if np.sum(ACT) == 0:
            state.rmsthk.append(0)
            state.stdthk.append(0)
        else:
            state.rmsthk.append(np.nanmean(state.thk[ACT] - state.thkobs[ACT]))
            state.stdthk.append(np.nanstd(state.thk[ACT] - state.thkobs[ACT]))

    else:
        state.rmsthk.append(0)
        state.stdthk.append(0)

    if hasattr(state, "uvelsurfobs"):
        velsurf_mag = getmag(state.uvelsurf, state.vvelsurf).numpy()
        velsurfobs_mag = getmag(state.uvelsurfobs, state.vvelsurfobs).numpy()
        ACT = ~np.isnan(velsurfobs_mag)

        state.rmsvel.append(
            np.mean(velsurf_mag[(I & ACT).numpy()] - velsurfobs_mag[(I & ACT).numpy()])
        )
        state.stdvel.append(
            np.std(velsurf_mag[(I & ACT).numpy()] - velsurfobs_mag[(I & ACT).numpy()])
        )
    else:
        state.rmsvel.append(0)
        state.stdvel.append(0)

    if hasattr(state, "divfluxobs"):
        state.rmsdiv.append(np.mean(state.divfluxobs[I] - state.divflux[I]))
        state.stddiv.append(np.std(state.divfluxobs[I] - state.divflux[I]))
    else:
        state.rmsdiv.append(0)
        state.stddiv.append(0)

    if hasattr(state, "usurfobs"):
        state.rmsusurf.append(np.mean(state.usurf[I] - state.usurfobs[I]))
        state.stdusurf.append(np.std(state.usurf[I] - state.usurfobs[I]))
    else:
        state.rmsusurf.append(0)
        state.stdusurf.append(0)

    state.vol.append((np.sum(state.thk) * (state.dx**2) / 1e9).numpy())
 
def create_density_matrix(data, kernel_size):
    # Convert data to binary mask (1 for valid data, 0 for NaN)
    binary_mask = tf.where(tf.math.is_nan(data), tf.zeros_like(data), tf.ones_like(data))

    # Create a kernel for convolution (all ones)
    kernel = tf.ones((kernel_size, kernel_size, 1, 1), dtype=binary_mask.dtype)

    # Apply convolution to count valid data points in the neighborhood
    density = tf.nn.conv2d(tf.expand_dims(tf.expand_dims(binary_mask, 0), -1), 
                           kernel, strides=[1, 1, 1, 1], padding='SAME')

    # Remove the extra dimensions added for convolution
    density = tf.squeeze(density)

    return density

def ave4(A):
    return (A[1:, 1:] + A[:-1, 1:] + A[1:, :-1] + A[:-1, :-1]) / 4.0

def compute_flow_direction_for_anisotropic_smoothing_vel(state):

    state.flowdirx = tf.where(tf.math.is_nan(state.uvelsurf), 0.0, state.uvelsurf)
    state.flowdiry = tf.where(tf.math.is_nan(state.vvelsurf), 0.0, state.vvelsurf)
  
    state.flowdirx = gaussian_filter_tf(state.flowdirx, sigma=20, kernel_size=20, mask=None)
    state.flowdiry = gaussian_filter_tf(state.flowdiry, sigma=20, kernel_size=20, mask=None)
 
    state.flowdirx /= getmag(state.flowdirx, state.flowdiry)
    state.flowdiry /= getmag(state.flowdirx, state.flowdiry)

    state.flowdirx = tf.where(tf.math.is_nan(state.flowdirx), 0.0, state.flowdirx)
    state.flowdiry = tf.where(tf.math.is_nan(state.flowdiry), 0.0, state.flowdiry)

    # state.flowdirx = ave4(state.flowdirx)
    # state.flowdiry = ave4(state.flowdiry)

def compute_flow_direction_for_anisotropic_smoothing_usurf(state):
 
    def _kernels(dx: float):
        """3×3 central-difference stencils."""
        kx  = tf.constant([[0.,0.,0.],[-1.,0.,1.],[0.,0.,0.]], tf.float32) / (2.0*dx)
        ky  = tf.constant([[0.,-1.,0.],[0., 0.,0.],[0., 1.,0.]], tf.float32) / (2.0*dx)
        expand = lambda K: tf.reshape(K, [3,3,1,1])
        return expand(kx), expand(ky)

    def _conv2(x, k):
        """Apply 2D convolution to a 2D tensor."""
        # Convert 2D tensor to 4D for conv2d: [batch, height, width, channels]
        x_4d = tf.expand_dims(tf.expand_dims(x, 0), -1)
        
        # Apply padding and convolution
        pad = [[0,0], [1,1], [1,1], [0,0]]  # 1 pixel on H and W
        x_pad = tf.pad(x_4d, pad, mode="SYMMETRIC")
        result = tf.nn.conv2d(x_pad, k, strides=1, padding='VALID')
        
        # Convert back to 2D
        return tf.squeeze(result, [0, 3])
 
    kx, ky = _kernels(state.dx)
    s_s = gaussian_filter_tf(state.usurfobs, sigma=20, kernel_size=20, mask=state.thk>0) 
    sx = _conv2(s_s, kx); sy = _conv2(s_s, ky)
    mag = tf.maximum(tf.sqrt(sx*sx + sy*sy), 1.0e-12)
 
    state.flowdirx = -sx/mag
    state.flowdiry = -sy/mag

    # state.flowdirx = ave4(state.flowdirx)
    # state.flowdiry = ave4(state.flowdiry)

def gaussian_filter_tf(input_tensor, sigma=1.0, kernel_size=5, mask=None):
    """
    Apply Gaussian filter to a 2D tensor with optional mask support.
    
    Args:
        input_tensor: 2D tensor to filter
        sigma: Standard deviation for Gaussian kernel
        kernel_size: Size of the kernel (should be odd)
        mask: Optional boolean mask - filter only where mask is True
    
    Returns:
        Filtered tensor
    """
    # Ensure kernel_size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Create Gaussian kernel
    x = tf.range(-(kernel_size // 2), kernel_size // 2 + 1, dtype=tf.float32)
    y = tf.range(-(kernel_size // 2), kernel_size // 2 + 1, dtype=tf.float32)
    X, Y = tf.meshgrid(x, y)
    
    # Gaussian formula
    kernel = tf.exp(-(X**2 + Y**2) / (2 * sigma**2))
    kernel = kernel / tf.reduce_sum(kernel)
    
    # Reshape for conv2d: [height, width, in_channels, out_channels]
    kernel = tf.reshape(kernel, [kernel_size, kernel_size, 1, 1])
    
    # Prepare input for conv2d: [batch, height, width, channels]
    input_4d = tf.expand_dims(tf.expand_dims(input_tensor, 0), -1)
    
    if mask is not None:
        # Apply mask-aware filtering
        mask_4d = tf.expand_dims(tf.expand_dims(tf.cast(mask, tf.float32), 0), -1)
        
        # Filter the masked input
        masked_input = input_4d * mask_4d
        filtered_masked = tf.nn.conv2d(masked_input, kernel, strides=1, padding='SAME')
        
        # Filter the mask to get normalization weights
        filtered_mask = tf.nn.conv2d(mask_4d, kernel, strides=1, padding='SAME')
        
        # Normalize by the filtered mask to account for missing values
        filtered_normalized = tf.where(
            filtered_mask > 1e-8,
            filtered_masked / filtered_mask,
            input_4d
        )
        
        # Apply the original mask to the result
        result = tf.where(mask_4d > 0.5, filtered_normalized, input_4d)
    else:
        # Standard convolution without mask
        result = tf.nn.conv2d(input_4d, kernel, strides=1, padding='SAME')
    
    # Remove extra dimensions and return
    return tf.squeeze(result, [0, 3])

########################

def apply_relaxation(cfg, state):
      
#    state.usurf_ref = state.usurf

    # time = 0

    print("-------------- Relaxation steps -----------------")

    for ll in range(cfg.processes.data_assimilation.optimization.nb_relaxation_steps):

        # time += state.dt.numpy()

        # update_iceflow_emulator(cfg, state, ll, pertubate=False)

        update_iceflow_emulated(cfg, state)

        cfl = 0.1 ; step_max = 1

        state.dt = compute_dt_from_cfl(
            state.ubar, state.vbar, cfl, state.dx, step_max
        )

        state.divflux = compute_divflux_slope_limiter(
            state.ubar, state.vbar, state.thk, 
            state.dx, state.dx, state.dt, 
            slope_type=cfg.processes.thk.slope_type
        )

        ACT = state.thk > 0.5
        state.res = stats.linregress( state.usurf[ACT], state.divflux[ACT] )   
        divflux_reg = state.res.intercept + state.res.slope * state.usurf

#        divflux_reg = gaussian_filter_tf(state.divflux, sigma=15.0, kernel_size=25, mask=ACT)

        # conserve mass
        smb = divflux_reg + tf.reduce_mean((state.divflux - divflux_reg)[ACT])

        # smb = smb + 0.25 * gaussian_filter_tf(state.usurfobs - state.usurf, sigma=10.0, kernel_size=20, mask=ACT)

        thk_new  = tf.maximum(state.thk + state.dt * (smb - state.divflux), 0)

        state.thk  = tf.where(state.thk>0, thk_new, 0)

        state.usurf = state.topg + state.thk

        # Calculate the difference field
#        diff_field = state.usurf - state.usurfobs
#        diff  = np.std(diff_field[state.thk>0.0])
#        vol = tf.reduce_sum(state.thk).numpy() * state.dx * state.dx / 1e9  # in km3
#        print(" Relaxation step ", ll, " / ", time, "\n STD :", diff, "\n Volume :", vol)

