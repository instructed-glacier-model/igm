import tensorflow as tf

@tf.function()
def getmag(u, v):
    """
    return the norm of a 2D vector, e.g. to compute velbase_mag
    """
    return tf.norm(
        tf.concat([tf.expand_dims(u, axis=-1), tf.expand_dims(v, axis=-1)], axis=2),
        axis=2,
    )
