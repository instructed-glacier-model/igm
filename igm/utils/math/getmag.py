import tensorflow as tf

@tf.function()
def getmag(u, v):
    return tf.norm(tf.stack([u, v], axis=-1), axis=-1)