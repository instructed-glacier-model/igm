"""Interior of a state-provided cell mask."""

import tensorflow as tf


def get_mask(options, cfg, state):
    field = str(options.get("field", "")).strip()
    if not field:
        raise ValueError(
            "The thickness 'interior' domain constraint requires a field name."
        )
    if not hasattr(state, field):
        raise RuntimeError(
            f"Thickness domain constraint references missing state.{field}."
        )

    mask = tf.cast(getattr(state, field), tf.bool)
    padded = tf.pad(mask, [[1, 1], [1, 1]], constant_values=False)
    return (
        mask
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
    )
