"""Live grounded-ice constraint."""

import tensorflow as tf


def get_mask(options, cfg, state):
    """Return cells whose ice base is below the bed rather than floating."""
    dtype = state.thk.dtype
    water_level = tf.cast(getattr(state, "water_level", 0.0), dtype)
    ratio_density = tf.cast(cfg.processes.thk.ratio_density, dtype)
    grounded = state.topg + ratio_density * state.thk > water_level
    state.groundedmask = tf.cast(grounded, dtype)
    return grounded
