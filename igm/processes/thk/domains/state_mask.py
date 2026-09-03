"""Constraint backed directly by a named state field."""

import tensorflow as tf


def get_mask(options, cfg, state):
    field = str(options.get("field", "")).strip()
    if not field:
        raise ValueError(
            "The thickness 'state_mask' domain constraint requires a field name."
        )
    if not hasattr(state, field):
        raise RuntimeError(
            f"Thickness domain constraint references missing state.{field}."
        )
    return tf.cast(getattr(state, field), tf.bool)
