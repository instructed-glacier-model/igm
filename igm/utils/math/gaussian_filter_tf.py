import tensorflow as tf 

def gaussian_kernel(size: int, sigma: float):
    """Creates a 2D Gaussian kernel."""
    x = tf.range(-size // 2 + 1, size // 2 + 1, dtype=tf.float32)
    x = tf.cast(x, tf.float32)
    xx, yy = tf.meshgrid(x, x)
    kernel = tf.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel = kernel / tf.reduce_sum(kernel)
    return kernel[:, :, tf.newaxis, tf.newaxis]  # shape (k, k, 1, 1)

def gaussian_filter_tf(qx, sigma=1.0, kernel_size=5):
    """Apply 2D Gaussian filter to qx using TensorFlow."""
    qx = qx[tf.newaxis, :, :, tf.newaxis]  # shape (1, ny, nx, 1)
    kernel = gaussian_kernel(kernel_size, sigma)
    qx_smooth = tf.nn.conv2d(qx, kernel, strides=1, padding='SAME')
    return qx_smooth[0, :, :, 0]  # shape (ny, nx)
