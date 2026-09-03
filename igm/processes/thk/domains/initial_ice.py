"""Constraint retaining the initially ice-covered cells."""

import tensorflow as tf


def initialize(options, cfg, state):
    threshold = tf.cast(options.get("min_thickness", 0.0), state.thk.dtype)
    state.thk_components.initial_ice_mask = state.thk > threshold


def get_mask(options, cfg, state):
    return state.thk_components.initial_ice_mask
