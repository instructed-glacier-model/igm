import tensorflow as tf


@tf.function()
def getmag(u: tf.Tensor, v: tf.Tensor) -> tf.Tensor:
    """Magnitude of the vector field ``(u, v)``, element-wise."""
    return tf.norm(tf.stack([u, v], axis=-1), axis=-1)
