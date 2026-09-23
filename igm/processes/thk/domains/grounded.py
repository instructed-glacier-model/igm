"""Live grounded-ice constraint."""

import tensorflow as tf

from ..masks import compute_grounded_mask
from ..surfaces import get_density_ratio


def get_mask(options, cfg, state):
    """Return cells whose ice base is below the bed rather than floating."""
    rho_ratio = 1.0 / get_density_ratio(cfg)
    grounded = compute_grounded_mask(
        state.thk, state.topg, state.water_level, rho_ratio
    )
    state.groundedmask = tf.cast(grounded, state.thk.dtype)
    return grounded
