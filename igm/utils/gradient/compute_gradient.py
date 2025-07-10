import tensorflow as tf

@tf.function()
def compute_gradient(s, dx, dy, staggered_grid=False):

    if staggered_grid:

#    compute spatial gradient, outcome on stagerred grid

        E = 2.0 * (s[..., :, 1:] - s[..., :, :-1]) / (dx[..., :, 1:] + dx[..., :, :-1])
        diffx = 0.5 * (E[..., 1:, :] + E[..., :-1, :])

        EE = 2.0 * (s[..., 1:, :] - s[..., :-1, :]) / (dy[..., 1:, :] + dy[..., :-1, :])
        diffy = 0.5 * (EE[..., :, 1:] + EE[..., :, :-1])

        return diffx, diffy

    else:

    #   compute spatial 2D gradient of a given field

        # EX = tf.concat([s[:, 0:1], 0.5 * (s[:, :-1] + s[:, 1:]), s[:, -1:]], 1)
        # diffx = (EX[:, 1:] - EX[:, :-1]) / dx

        # EY = tf.concat([s[0:1, :], 0.5 * (s[:-1, :] + s[1:, :]), s[-1:, :]], 0)
        # diffy = (EY[1:, :] - EY[:-1, :]) / dy

        EX = tf.concat(
            [
                1.5 * s[..., :, 0:1] - 0.5 * s[..., :, 1:2],
                0.5 * s[..., :, :-1] + 0.5 * s[..., :, 1:],
                1.5 * s[..., :, -1:] - 0.5 * s[..., :, -2:-1],
            ],
            -1,
        )
        diffx = (EX[..., :, 1:] - EX[..., :, :-1]) / dx

        EY = tf.concat(
            [
                1.5 * s[..., 0:1, :] - 0.5 * s[..., 1:2, :],
                0.5 * s[..., :-1, :] + 0.5 * s[..., 1:, :],
                1.5 * s[..., -1:, :] - 0.5 * s[..., -2:-1, :],
            ],
            -2,
        )
        diffy = (EY[..., 1:, :] - EY[..., :-1, :]) / dy

        return diffx, diffy