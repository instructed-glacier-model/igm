import tensorflow as tf


@tf.function()
def getmag3d(u, v):
    """
    return the norm of a 3D vector, e.g. to compute velbase_mag
    """
    return tf.norm(
        tf.concat([tf.expand_dims(u, axis=0), tf.expand_dims(v, axis=0)], axis=0),
        axis=0,
    )
