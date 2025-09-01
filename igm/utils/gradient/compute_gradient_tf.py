import tensorflow as tf

# THIS NEED TO REOVED ON THE LONG TERM SINCE EMBEDED IN  compute_gradient.py

@tf.function()
def compute_gradient_tf(s, dx, dy):
    """
    compute spatial 2D gradient of a given field
    """

    # EX = tf.concat([s[:, 0:1], 0.5 * (s[:, :-1] + s[:, 1:]), s[:, -1:]], 1)
    # diffx = (EX[:, 1:] - EX[:, :-1]) / dx

    # EY = tf.concat([s[0:1, :], 0.5 * (s[:-1, :] + s[1:, :]), s[-1:, :]], 0)
    # diffy = (EY[1:, :] - EY[:-1, :]) / dy

    EX = tf.concat(
        [
            1.5 * s[:, 0:1] - 0.5 * s[:, 1:2],
            0.5 * s[:, :-1] + 0.5 * s[:, 1:],
            1.5 * s[:, -1:] - 0.5 * s[:, -2:-1],
        ],
        1,
    )
    diffx = (EX[:, 1:] - EX[:, :-1]) / dx

    EY = tf.concat(
        [
            1.5 * s[0:1, :] - 0.5 * s[1:2, :],
            0.5 * s[:-1, :] + 0.5 * s[1:, :],
            1.5 * s[-1:, :] - 0.5 * s[-2:-1, :],
        ],
        0,
    )
    diffy = (EY[1:, :] - EY[:-1, :]) / dy

    return diffx, diffy