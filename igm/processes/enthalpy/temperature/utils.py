#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Tuple


@tf.function
def compute_pmp_tf(
    rho_ice: tf.Tensor,
    g: tf.Tensor,
    depth_ice: tf.Tensor,
    beta: tf.Tensor,
    c_ice: tf.Tensor,
    T_pmp_ref: tf.Tensor,
    T_ref: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    TensorFlow function to compute pressure melting point temperature and enthalpy.

    Args:
        rho_ice: Ice density (kg m^-3).
        g: Gravitational acceleration (m s^-2).
        depth_ice: Depth below ice surface (m).
        beta: Clausius-Clapeyron constant (K Pa^-1).
        c_ice: Specific heat capacity of ice (J kg^-1 K^-1).
        T_pmp_ref: Reference pressure melting point temperature (K).
        T_ref: Reference temperature (K).

    Returns:
        Tuple of pressure melting point temperature (K) and enthalpy (J kg^-1).
    """
    p_ice = rho_ice * g * depth_ice
    T_pmp = T_pmp_ref - beta * p_ice
    E_pmp = c_ice * (T_pmp - T_ref)
    return T_pmp, E_pmp


@tf.function
def compute_T_tf(
    E: tf.Tensor,
    E_pmp: tf.Tensor,
    T_ref: tf.Tensor,
    c_ice: tf.Tensor,
) -> tf.Tensor:
    """
    TensorFlow function to compute temperature from enthalpy.

    For cold ice (E < E_pmp), temperature is computed from enthalpy.
    For temperate ice (E >= E_pmp), temperature equals the pressure melting point,
    derived inline as E_pmp / c_ice + T_ref.

    Args:
        E: Enthalpy field (J kg^-1).
        E_pmp: Pressure melting point enthalpy (J kg^-1).
        T_ref: Reference temperature (K).
        c_ice: Specific heat capacity of ice (J kg^-1 K^-1).

    Returns:
        Temperature field (K).
    """
    return tf.minimum(E, E_pmp) / c_ice + T_ref


@tf.function
def compute_omega_tf(
    E: tf.Tensor,
    E_pmp: tf.Tensor,
    L_ice: tf.Tensor,
) -> tf.Tensor:
    """
    TensorFlow function to compute water content from enthalpy.

    For cold ice (E < E_pmp), water content is zero.
    For temperate ice (E >= E_pmp), water content is computed from excess enthalpy.

    Args:
        E: Enthalpy field (J kg^-1).
        E_pmp: Pressure melting point enthalpy (J kg^-1).
        L_ice: Latent heat of fusion for ice (J kg^-1).

    Returns:
        Water content fraction (-).
    """
    return tf.where(E >= E_pmp, (E - E_pmp) / L_ice, 0.0)


@tf.function()
def compute_pa_tf(
    T: tf.Tensor,
    beta: tf.Tensor,
    rho_ice: tf.Tensor,
    g: tf.Tensor,
    depth_ice: tf.Tensor,
) -> tf.Tensor:
    """
    TensorFlow function to compute pressure-adjusted temperature.

    Args:
        T: Temperature field (K).
        beta: Clausius-Clapeyron constant (K Pa^-1).
        rho_ice: Ice density (kg m^-3).
        g: Gravitational acceleration (m s^-2).
        depth_ice: Depth below ice surface (m).

    Returns:
        Pressure-adjusted temperature (K).
    """
    return T + beta * rho_ice * g * depth_ice


@tf.function
def compute_E_cold_tf(
    T: tf.Tensor,
    T_pmp: tf.Tensor,
    T_ref: tf.Tensor,
    c_ice: tf.Tensor,
) -> tf.Tensor:
    """
    TensorFlow function to compute cold ice enthalpy from temperature.

    Computes enthalpy assuming no water content. If temperature exceeds the
    pressure melting point, enthalpy is capped at the melting point value.

    Args:
        T: Temperature field (K).
        T_pmp: Pressure melting point temperature (K).
        T_ref: Reference temperature (K).
        c_ice: Specific heat capacity of ice (J kg^-1 K^-1).

    Returns:
        Cold ice enthalpy (J kg^-1).
    """
    return tf.where(
        T < T_pmp,
        c_ice * (T - T_ref),
        c_ice * (T_pmp - T_ref),
    )
