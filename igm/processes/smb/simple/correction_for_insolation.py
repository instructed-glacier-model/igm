#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

# Correction of the ELA for insolation (aspect / slope of the surface).
# The ELA is shifted per grid cell depending on the difference between the
# solar incidence angle on the local surface and on a flat surface:
# sun-facing slopes get a higher ELA (more melt), shaded slopes a lower one.

# Details to this correction can be found in Henz et al. 2025, TC, https://doi.org/10.5194/tc-19-5913-2025

import os
import tensorflow as tf
from igm.utils.grad.grad import grad_xy


def correct_ela_for_insolation(cfg, state, ela):
    """Return a 2D ELA field corrected for insolation."""

    # make params more accessible and readable..
    p = cfg.processes.smb.simple.correction_for_insolation

    solar_elevation = p.solar_elevation
    solar_azimuth = p.solar_azimuth
    ela_per_degree = p.ela_per_degree_incidence

    state.angle_of_incidences = calc_incidence_angle(
        state, solar_elevation, solar_azimuth
    )

    # difference in incidence angle compared to a flat surface
    angle_difference = (90.0 - solar_elevation) - state.angle_of_incidences

    ela_matrix = tf.ones_like(state.usurf) * ela + angle_difference * ela_per_degree

    # one-time diagnostic figure to check the orientation of the correction
    if p.plot_insolation_check and not hasattr(
        state, "insolation_check_done"
    ):
        plot_insolation_check(state, ela_matrix - ela)
        state.insolation_check_done = True

    return ela_matrix


def calculate_normal_vector(state):
    # signed grid spacing in y, so that the gradient is correct even if y is decreasing
    dy = state.dX * tf.sign(state.y[1] - state.y[0])
    dzdx, dzdy = grad_xy(state.usurf, state.dX, dy, False, "extrapolate")

    normal_vector = tf.stack([-dzdx, -dzdy, tf.ones_like(dzdx)], axis=-1)

    return normal_vector / tf.norm(normal_vector, axis=-1, keepdims=True)


def calculate_sun_vector(elevation_angle, azimuth_angle):
    elevation_rad = tf_deg2rad(elevation_angle)
    azimuth_rad = tf_deg2rad(azimuth_angle)

    # x: east, y: north, z: up ; azimuth measured clockwise from north
    sun_vector = tf.stack(
        [
            tf.cos(elevation_rad) * tf.sin(azimuth_rad),
            tf.cos(elevation_rad) * tf.cos(azimuth_rad),
            tf.sin(elevation_rad),
        ]
    )

    return sun_vector / tf.norm(sun_vector)


def calc_incidence_angle(state, solar_elevation, solar_azimuth=180.0):
    # angle between sun vector and surface normal vector [degrees]
    sun_vector = calculate_sun_vector(solar_elevation, solar_azimuth)
    normal_vectors = calculate_normal_vector(state)

    dot_product = tf.reduce_sum(normal_vectors * sun_vector, axis=-1)
    dot_product = tf.clip_by_value(dot_product, -1.0, 1.0)

    return tf_rad2deg(tf.acos(dot_product))


def plot_insolation_check(state, ela_correction):
    import numpy as np
    import matplotlib.pyplot as plt

    extent = [
        float(tf.reduce_min(state.x)),
        float(tf.reduce_max(state.x)),
        float(tf.reduce_min(state.y)),
        float(tf.reduce_max(state.y)),
    ]
    # imshow with origin='lower' expects increasing y along axis 0
    flip = bool(state.y[0] > state.y[-1])
    usurf = state.usurf.numpy()[::-1] if flip else state.usurf.numpy()
    corr = ela_correction.numpy()[::-1] if flip else ela_correction.numpy()

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    im = ax[0].imshow(usurf, cmap="terrain", origin="lower", extent=extent)
    ax[0].set_title("Surface topography")
    plt.colorbar(im, ax=ax[0], orientation="horizontal").set_label(
        "Surface elevation [m a.s.l.]"
    )

    # vmin and max are set to symmetric values around zero for better visualization
    vmax = np.percentile(np.abs(corr), 98)
    vmin = -vmax

    im = ax[1].imshow(corr, cmap="RdBu_r", origin="lower", extent=extent, vmin=vmin, vmax=vmax)
    ax[1].set_title("ELA correction for insolation")
    plt.colorbar(im, ax=ax[1], orientation="horizontal").set_label(
        "ELA shift [m] (north is up)"
    )

    plt.savefig(os.path.join(os.getcwd(), "check_insolation_correction.png"))
    plt.close(fig)


def tf_rad2deg(rad):
    return rad / 0.017453292519943295


def tf_deg2rad(deg):
    return deg * 0.017453292519943295
