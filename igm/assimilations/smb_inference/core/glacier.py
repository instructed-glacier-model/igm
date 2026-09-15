from .smb import update_smb_PDD, update_smb_ELA, update_smb_profile, cosine_temperature_series
from ..visualization.plots import visualize, visualize_velocities, plot_thickness_divflux_velocities
import tensorflow as tf
import warnings
from .forward_schemes.pretrained_cnn_step import make_pretrained_cnn_step
from igm.processes.iceflow.unified.solver.solver import solve_iceflow

class GlacierDynamicsCheckpointed(tf.keras.Model):
    """
    Glacier dynamics model with gradient checkpointing support.

    This class implements a differentiable glacier dynamics simulation
    driven by IGM's pretrained ice-flow emulator.

    Parameters:
    -----------
    Z_topo : tf.Tensor
        Bedrock topography (ny, nx)
    H_init : tf.Tensor
        Initial ice thickness (ny, nx)
    ice_mask : tf.Tensor
        Ice mask (ny, nx)
    cfg : omegaconf.DictConfig
        The full IGM Hydra config (run parameters are read from
        cfg.assimilations.smb_inference and cfg.processes.iceflow.physics)
    dx, dy : float
        Grid spacing (m), derived from state.x / state.y
    model : keras.Model, optional
        Trained emulator model. Only needed if not inheriting from state.
    visualize : bool
        Whether to visualize results during simulation
    V_bar : tf.Tensor, optional
        Vertical averaging weights, shape (Nz,).
        Only needed if not inheriting from state.
    Nz : int, optional
        Number of vertical levels.
        Only needed if not inheriting from state.
    input_fields : list of str
        Ordered emulator input field names, taken from
        cfg.processes.iceflow.unified.inputs. Required: it fixes the channel
        order the emulator is fed.
    """

    def __init__(self, Z_topo, H_init, ice_mask, cfg, dx, dy, model=None,
                 visualize=False, V_bar=None, Nz=None, input_fields=None,
                 state=None):
        super().__init__()
        self.Z_topo = Z_topo
        self.ice_mask = ice_mask
        self.H_init = H_init
        self.do_visualize = visualize
        self.input_fields = input_fields

        # Inherit emulator, V_bar, Nz from state when available
        if state is not None and hasattr(state, 'iceflow_model') and state.iceflow_model is not None:
            self.emulator_model = state.iceflow_model
            self.V_bar = state.iceflow.discr_v.V_bar
            self.Nz = int(self.V_bar.shape[0])
        else:
            self.emulator_model = model
            self.V_bar = V_bar
            self.Nz = Nz
        self.state = state
        self.cfg = cfg

        # Run parameters straight from the Hydra config
        smb_cfg = cfg.assimilations.smb_inference

        self.ttot = tf.constant(float(smb_cfg.ttot), dtype=tf.float32)
        self.t_start = tf.constant(float(smb_cfg.t_start), dtype=tf.float32)
        self.dx = dx
        self.dy = dy
        self.dtmax = float(smb_cfg.dtmax)
        self.cfl = float(smb_cfg.cfl)
        self.vis_freq = float(smb_cfg.output.vis_freq)

        if self.do_visualize:
            warnings.warn(
                "Visualization is enabled. Gradient-based inversion is not "
                "supported when visualize=True because a Python-level loop "
                "is used instead of tf.while_loop.",
                stacklevel=2,
            )

    def call(self, precip_tensor, T_m_lowest, T_s, melt_factor,
             smb_method='PDD', ELA=None, grad_b=None, b_max=None, smb_field=None,
             smb_vec=None, z_min=None, dz=None, save_times=None, forward_scheme=None,
             retrain_mode=False):
        """
        Run the glacier dynamics simulation.

        Parameters:
        -----------
        retrain_mode : bool
            If True, uses a Python-level loop with emulator retraining
            (no gradient tracking). If False, uses tf.while_loop for
            differentiable forward simulation.
        """
        # Use differentiable solver for profile-based inversion
        if smb_method == 'profile':

            if not isinstance(z_min, tf.Tensor):
                z_min = tf.constant(z_min, dtype=tf.float32)
            if not isinstance(dz, tf.Tensor):
                dz = tf.constant(dz, dtype=tf.float32)
        return self.solve_glacier_dynamics_differentiable(
                self.Z_topo, self.ttot, self.t_start, smb_vec, z_min, dz,
                retrain_mode=retrain_mode,
            )



    def _retrain_emulator(self, H_ice, Z_surf):
        """Sync current glacier state to IGM state and retrain the iceflow emulator."""

        # Sync glacier variables → IGM state field names
        self.state.thk = H_ice
        self.state.usurf = Z_surf

        saved_it = getattr(self.state, 'it', 0)
        self.state.it = max(int(saved_it), 1)
        solve_iceflow(self.cfg, self.state, init=True)
        self.state.it = saved_it
        # if hasattr(self.state, 'cost') and len(self.state.cost) > 0:
        #     print(f"[smb_inference] Retrained emulator, cost={self.state.cost}")

    def solve_glacier_dynamics_differentiable(self, Z_topo, ttot, time, smb_vec, z_min, dz,
                                               retrain_mode=False):
        """
        Glacier dynamics solver.

        retrain_mode=False: tf.while_loop (differentiable, for use inside GradientTape).
        retrain_mode=True:  Python-level loop with emulator retraining every
                            retrain_interval simulated years (no gradient tracking).
        """
        # Initialize state
        H_ice = self.H_init
        Z_surf = Z_topo + H_ice

        # Get arrhenius/slidingco from state (set by DA), fall back to config defaults
        if self.state is not None and hasattr(self.state, 'arrhenius') and self.state.arrhenius is not None:
            arrhenius_field = tf.cast(self.state.arrhenius, tf.float32)
        else:
            init_arr = float(self.cfg.processes.iceflow.physics.viscosity.arrhenius)
            arrhenius_field = tf.ones_like(H_ice) * init_arr

        if self.state is not None and hasattr(self.state, 'slidingco') and self.state.slidingco is not None:
            slidingco_field = tf.cast(self.state.slidingco, tf.float32)
        else:
            init_sc = float(self.cfg.processes.iceflow.physics.sliding.slidingco)
            slidingco_field = tf.ones_like(H_ice) * init_sc

        # Initial SMB from profile
        smb = update_smb_profile(Z_surf, smb_vec, z_min, dz) * self.ice_mask
        smb = tf.where((smb < 0) | (self.ice_mask > 0.5), smb, tf.constant(-10.0, dtype=tf.float32))

        # Create emulator step function. The input list comes from the iceflow
        # config, so it is always known; without it the channel order the
        # emulator expects cannot be reconstructed.
        if self.input_fields is None:
            raise ValueError(
                "smb_inference needs the emulator's input field list; set "
                "processes.iceflow.unified.inputs."
            )
        emulator_step_fn = make_pretrained_cnn_step(
            self.emulator_model, self.V_bar, self.Nz, self.input_fields
        )

        # Convert loop scalars to tensors
        time_tf = tf.constant(time, dtype=tf.float32)
        ttot_tf = tf.constant(ttot, dtype=tf.float32)
        t_last_update = tf.constant(self.t_start, dtype=tf.float32)
        vis_freq_tf = tf.constant(self.vis_freq, dtype=tf.float32)
        idx = tf.constant(0, dtype=tf.int32)
        neg_smb_fill = tf.constant(-10.0, dtype=tf.float32)

        # Capture constants for the loop body closure
        ice_mask = self.ice_mask
        dx = tf.constant(self.dx, dtype=tf.float32)
        dy = tf.constant(self.dy, dtype=tf.float32)
        dtmax = tf.constant(self.dtmax, dtype=tf.float32)
        cfl = tf.constant(self.cfl, dtype=tf.float32)

        def cond(H_ice, Z_surf, smb, time_tf, t_last_update, idx):
            return time_tf < ttot_tf

        def body(H_ice, Z_surf, smb, time_tf, t_last_update, idx):
            # Emulator step
            H_ice, Z_surf, time_tf, ubar, vbar = emulator_step_fn(
                H_ice, Z_surf, smb, time_tf,
                Z_topo, dx, dy, dtmax, cfl,
                arrhenius_field, slidingco_field
            )

            # Check if SMB update is due
            def update_smb_fn():
                new_smb = update_smb_profile(Z_surf, smb_vec, z_min, dz) * ice_mask
                new_smb = tf.where((new_smb < 0) | (ice_mask > 0.5), new_smb, neg_smb_fill)
                return new_smb, time_tf, idx + 1

            def keep_smb_fn():
                return smb, t_last_update, idx

            smb, t_last_update, idx = tf.cond(
                (time_tf - t_last_update) >= vis_freq_tf,
                update_smb_fn,
                keep_smb_fn,
            )

            return H_ice, Z_surf, smb, time_tf, t_last_update, idx

        if retrain_mode:
            # Python-level loop with emulator retraining (no gradient tracking)
            retrain_interval = 1.0
            if self.cfg is not None:
                retrain_interval = getattr(
                    self.cfg.assimilations.smb_inference.optimization, 'retrain_interval', 1.0
                )
            retrain_interval_tf = tf.constant(retrain_interval, dtype=tf.float32)
            t_last_retrain = tf.constant(time, dtype=tf.float32)

            # First and last replay steps only — used for the start-vs-end
            # flux-divergence comparison plot. The continuity SMB estimate gets
            # its div(flux) from the DA output, not from this replay.
            self.snapshot_start = None
            self.snapshot_end = None
            first_step = True

            while time_tf < ttot_tf:
                H_ice, Z_surf, time_tf, ubar, vbar = emulator_step_fn(
                    H_ice, Z_surf, smb, time_tf,
                    Z_topo, dx, dy, dtmax, cfl,
                    arrhenius_field, slidingco_field
                )
                plot_thickness_divflux_velocities(
                            H_ice, ubar, vbar,
                            dx=float(self.dx), dy=float(self.dy),
                            time=float(time_tf)
                        )

                snapshot = {
                    "H": tf.identity(H_ice),
                    "ubar": tf.identity(ubar),
                    "vbar": tf.identity(vbar),
                    "Z_surf": tf.identity(Z_surf),
                    "time": float(time_tf.numpy()),
                }

                if first_step:
                    self.snapshot_start = snapshot
                    first_step = False
                self.snapshot_end = snapshot

                if (time_tf - t_last_update) >= vis_freq_tf:
                    new_smb = update_smb_profile(Z_surf, smb_vec, z_min, dz) * ice_mask
                    smb = tf.where((new_smb < 0) | (ice_mask > 0.5), new_smb, neg_smb_fill)
                    t_last_update = time_tf
                    idx = idx + 1

                # Retrain emulator at specified interval
                # self._retrain_emulator(H_ice, Z_surf)
                # if (time_tf - t_last_retrain) >= retrain_interval_tf:
                    # self._retrain_emulator(H_ice, Z_surf)
                    # t_last_retrain = time_tf
        else:
            # tf.while_loop for differentiable forward simulation
            H_ice, Z_surf, smb, time_tf, t_last_update, idx = tf.while_loop(
                cond,
                body,
                loop_vars=[H_ice, Z_surf, smb, time_tf, t_last_update, idx],
                parallel_iterations=1,
                swap_memory=False,
            )

        return H_ice
