
import tensorflow as tf

@tf.function()
def interp1d_tf(xs, ys, x):
    """
    This is a 1D interpolation tensorflow implementation
    """
    x = tf.clip_by_value(x, tf.reduce_min(xs), tf.reduce_max(xs))

    # determine the output data type
    ys = tf.convert_to_tensor(ys)
    dtype = ys.dtype

    # normalize data types
    ys = tf.cast(ys, tf.float64)
    xs = tf.cast(xs, tf.float64)
    x = tf.cast(x, tf.float64)

    # pad control points for extrapolation
    xs = tf.concat([[xs.dtype.min], xs, [xs.dtype.max]], axis=0)
    ys = tf.concat([ys[:1], ys, ys[-1:]], axis=0)

    # compute slopes, pad at the edges to flatten
    ms = (ys[1:] - ys[:-1]) / (xs[1:] - xs[:-1])
    ms = tf.pad(ms[:-1], [(1, 1)])

    # solve for intercepts
    bs = ys - ms * xs

    # search for the line parameters at each input data point
    # create a grid of the inputs and piece breakpoints for thresholding
    # rely on argmax stopping on the first true when there are duplicates,
    # which gives us an index into the parameter vectors
    i = tf.math.argmax(xs[..., tf.newaxis, :] > x[..., tf.newaxis], axis=-1)
    m = tf.gather(ms, i, axis=-1)
    b = tf.gather(bs, i, axis=-1)

    # apply the linear mapping at each input data point
    y = m * x + b
    return tf.cast(tf.reshape(y, tf.shape(x)), dtype)

