import tensorflow as tf

from igm.utils.grad.compute_divflux_slope_limiter import compute_divflux_slope_limiter
from ...utils.emulator_tools import apply_boundary_condition_tf


def make_pretrained_cnn_step(model, V_bar, Nz, input_fields):
    """
    Factory function that creates a gradient-checkpointed step for a
    pretrained CNN emulator loaded from an IGM artifact directory.

    Unlike the PINN step, the pretrained CNN:
    - Uses the model's own input_normalizer (FixedChannelStandardization)
    - Takes a variable set of inputs defined by the manifest (e.g. [thk, usurf, slidingco])
    - Outputs 2*Nz channels that are depth-averaged with V_bar

    Parameters
    ----------
    model : tf.keras.Model
        Pretrained CNN with input_normalizer attached.
    V_bar : tf.Tensor
        Vertical averaging weights, shape (Nz,).
    Nz : int
        Number of vertical levels.
    input_fields : list of str
        Ordered input field names from the manifest (e.g. ['thk', 'usurf', 'slidingco']).

    Returns
    -------
    pretrained_cnn_step_differentiable : function
        A gradient-checkpointed function for use in tf.while_loop.
    """
    V_bar_1d = tf.squeeze(V_bar)

    # Build a lookup so the step function knows which fields to stack
    field_set = set(input_fields)

    @tf.recompute_grad
    def pretrained_cnn_step_differentiable(H_ice, Z_surf, smb, time, Z_topo, dx, dy,
                                           dtmax, cfl, arrhenius_field, slidingco_field):
        dx = tf.cast(dx, tf.float32)
        dy = tf.cast(dy, tf.float32)
        dtmax = tf.cast(dtmax, tf.float32)
        cfl = tf.cast(cfl, tf.float32)

        # Build input channels in manifest order
        dX_field = tf.ones_like(H_ice) * dx
        channel_map = {
            'thk': H_ice,
            'usurf': Z_surf,
            'arrhenius': arrhenius_field,
            'slidingco': slidingco_field,
            'dX': dX_field,
        }
        channels = [channel_map[f] for f in input_fields]
        input_data = tf.stack(channels, axis=-1)
        input_data = tf.expand_dims(input_data, 0)  # (1, ny, nx, C)

        # Model handles normalization internally via input_normalizer
        velocity_pred = model(input_data, training=False)  # (1, ny, nx, 2*Nz)

        # Split into U and V components, then depth-average
        U_all = velocity_pred[0, :, :, :Nz]   # (ny, nx, Nz)
        V_all = velocity_pred[0, :, :, Nz:]   # (ny, nx, Nz)

        ubar = tf.einsum("ijn,n->ij", U_all, V_bar_1d)
        vbar = tf.einsum("ijn,n->ij", V_all, V_bar_1d)

        # CFL time step
        vel_max = tf.maximum(tf.reduce_max(tf.abs(ubar)), tf.reduce_max(tf.abs(vbar)))
        vel_max = tf.maximum(vel_max, 1e-4)
        dt = tf.minimum(cfl * dx / vel_max, dtmax)

        # Flux divergence and thickness update
        dHdt = -compute_divflux_slope_limiter(ubar, vbar, H_ice, dx, dy, dt, "superbee")

        H_ice_new = H_ice + dt * dHdt
        H_ice_new = tf.maximum(H_ice_new, 0.0)

        # Mass balance
        H_ice_new = H_ice_new + dt * smb

        # Boundary condition
        H_ice_new = apply_boundary_condition_tf(H_ice_new)

        # Update surface
        Z_surf_new = Z_topo + H_ice_new

        return H_ice_new, Z_surf_new, time + dt, ubar, vbar

    return pretrained_cnn_step_differentiable
