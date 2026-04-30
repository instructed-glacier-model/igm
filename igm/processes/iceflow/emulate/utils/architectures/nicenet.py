import tensorflow as tf
import numpy as np


class NiceNet(tf.keras.Model):
    """
    Configurable physics-informed ice flow architecture.

    Builds on the SIA (Shallow Ice Approximation) to pre-compute physically
    meaningful features that guide the neural network. Three physics levels
    control how much SIA structure is injected:

      minimal  (4 ch): Surface gradients + log-thickness
      standard (10 ch): + driving stress + SIA deformation velocity (with A!)
                         + sliding velocity proxy (with C!) + flow direction
      full     (16 ch): + log(A), log(C) spatial maps + surface curvature
                         + thickness-slope interaction + linear thickness

    The SIA depth-averaged deformation velocity is:
        ū_def = -2A/(n+2) · (ρg)^n · |∇s|^(n-1) · ∇s · H^(n+1)

    Unlike SIAFNO/SIANet, this architecture properly includes the Arrhenius
    rate factor A and sliding coefficient C in the physics features, giving
    the network actual velocity estimates rather than partial geometric proxies.

    Configurable hyperparameters:
        emulator.network.nb_layers     : int  (backbone depth)
        emulator.network.nb_out_filter : int  (channel width)
        emulator.network.conv_ker_size : int  (kernel size)
        emulator.network.residual      : bool (residual connections)
    """

    def __init__(self, cfg, nb_inputs, nb_outputs, input_normalizer=None, **kwargs):
        super().__init__(**kwargs)

        cfg_physics = cfg.processes.iceflow.physics
        cfg_numerics = cfg.processes.iceflow.numerics
        cfg_unified = cfg.processes.iceflow.unified
        cfg_emulator = cfg.processes.iceflow.emulator

        self.nb_inputs = nb_inputs
        self.nb_outputs = nb_outputs
        self.Nz = int(cfg_numerics.Nz)
        self.n_glen = float(cfg_physics.exp_glen)
        self.rho = float(cfg_physics.ice_density)
        self.g = float(cfg_physics.gravity_cst)
        self.rho_g = self.rho * self.g

        self.input_normalizer = input_normalizer

        # Input channel indices
        input_names = list(cfg_unified.inputs)
        self.idx_thk = input_names.index("thk")
        self.idx_usurf = input_names.index("usurf")
        self.idx_dX = input_names.index("dX")
        self.idx_arrhenius = (
            input_names.index("arrhenius") if "arrhenius" in input_names else None
        )
        self.idx_slidingco = (
            input_names.index("slidingco") if "slidingco" in input_names else None
        )

        # ── IceFlowNet-specific hyperparameters ──
        # Hardcoded for now — will become configurable via unified.network.*
        self.physics_level = "minimal"
        self.multi_scale = False

        # Network size params
        n_filters = int(cfg_emulator.network.nb_out_filter)
        n_layers = int(cfg_emulator.network.nb_layers)
        ker_size = int(getattr(cfg_emulator.network, "conv_ker_size", 3))
        self.use_residual = bool(
            getattr(cfg_emulator.network, "residual", True)
        )

        # ── Reference values for normalization ──
        self.slope_ref = 0.1       # typical alpine slope
        self.H_ref = 200.0         # reference thickness [m]
        self.tau_ref = self.rho_g * self.H_ref * self.slope_ref  # ~1.8e5 Pa
        self.A_ref = 7.6e-24       # rate factor at ~-10°C [Pa^-n s^-1]
        self.C_ref = 1e4           # typical sliding coefficient

        # Log-reference for SIA deformation velocity:
        #   u_ref = 2/(n+2) * A_ref * (ρg)^n * slope_ref^n * H_ref^(n+1)
        n = self.n_glen
        self.log_u_sia_ref = float(
            np.log(2.0 / (n + 2.0))
            + np.log(self.A_ref)
            + n * np.log(self.rho_g)
            + n * np.log(self.slope_ref)
            + (n + 1.0) * np.log(self.H_ref + 1.0)
        )
        # Log-reference for sliding velocity:
        #   u_slide_ref = C_ref * tau_ref^n
        self.log_u_slide_ref = float(
            np.log(self.C_ref) + n * np.log(self.tau_ref)
        )

        # Physics feature count
        if self.physics_level == "minimal":
            n_physics = 4
        elif self.physics_level == "full":
            n_physics = 16
        else:
            n_physics = 10

        # ── Stage 1: Multi-scale learned gradient filters (optional) ──
        n_grad_features = 0
        if self.multi_scale:
            self.grad_conv_3 = tf.keras.layers.Conv2D(
                8, 3, padding="same", dtype=tf.float32, name="grad_3x3"
            )
            self.grad_conv_5 = tf.keras.layers.Conv2D(
                8, 5, padding="same", dtype=tf.float32, name="grad_5x5"
            )
            self.grad_conv_7 = tf.keras.layers.Conv2D(
                4, 7, padding="same", dtype=tf.float32, name="grad_7x7"
            )
            n_grad_features = 20

        # ── Stage 2: Feature combination ──
        self.combine = tf.keras.layers.Dense(
            n_filters, dtype=tf.float32, name="combine"
        )

        # ── Stage 3: Backbone CNN ──
        self.backbone_convs = []
        for i in range(n_layers):
            self.backbone_convs.append(
                tf.keras.layers.Conv2D(
                    n_filters, ker_size, padding="same", dtype=tf.float32,
                    name=f"backbone_{i}",
                )
            )

        # Skip projection (from combined features to output)
        self.skip_proj = tf.keras.layers.Conv2D(
            n_filters, 1, padding="same", dtype=tf.float32, name="skip_proj"
        )

        # Output projection
        self.output_layer = tf.keras.layers.Conv2D(
            nb_outputs, 1, padding="same", dtype=tf.float32, name="output"
        )

        self.build(input_shape=[None, None, None, nb_inputs])

    def _compute_physics_features(self, raw_inputs):
        """
        Compute normalized SIA-derived features from raw (un-normalized) inputs.

        All features are kept in O(1) range for float32 stability.
        The physics_level controls how many features are computed.
        """
        x = tf.cast(raw_inputs, tf.float32)
        n = self.n_glen

        thk = x[..., self.idx_thk : self.idx_thk + 1]
        usurf = x[..., self.idx_usurf : self.idx_usurf + 1]
        dX = x[..., self.idx_dX : self.idx_dX + 1]

        # ── Surface gradients (central finite differences) ──
        inv_2dx = 1.0 / (2.0 * dX + 1e-10)
        dsdx = (tf.roll(usurf, -1, axis=2) - tf.roll(usurf, 1, axis=2)) * inv_2dx
        dsdy = (tf.roll(usurf, -1, axis=1) - tf.roll(usurf, 1, axis=1)) * inv_2dx
        grad_s = tf.sqrt(dsdx ** 2 + dsdy ** 2 + 1e-20)

        # ═══ MINIMAL: geometry only (4 features) ═══
        features = [
            dsdx / self.slope_ref,       # normalized surface gradient x
            dsdy / self.slope_ref,       # normalized surface gradient y
            grad_s / self.slope_ref,     # normalized slope magnitude
            tf.math.log(thk + 1.0),      # log-thickness (0 to ~6)
        ]

        if self.physics_level in ("standard", "full"):
            # ═══ STANDARD: + SIA velocity terms (6 more = 10 total) ═══

            # Driving stress τ = -ρgH∇s (normalized by tau_ref)
            tau_x = -self.rho_g * thk * dsdx
            tau_y = -self.rho_g * thk * dsdy
            features.append(tau_x / self.tau_ref)
            features.append(tau_y / self.tau_ref)

            # SIA deformation velocity magnitude in log-space (INCLUDES A!)
            # log|ū_def| = log(A) + n·log(ρg) + (n+1)·log(H) + n·log|∇s|
            if self.idx_arrhenius is not None:
                arrhenius = x[..., self.idx_arrhenius : self.idx_arrhenius + 1]
                log_A = tf.math.log(tf.maximum(arrhenius, 1e-30))
            else:
                log_A = tf.constant(
                    np.log(self.A_ref), dtype=tf.float32
                ) * tf.ones_like(thk)

            log_u_sia = (
                log_A
                + n * tf.math.log(tf.constant(self.rho_g, dtype=tf.float32))
                + (n + 1.0) * tf.math.log(thk + 1.0)
                + n * tf.math.log(grad_s + 1e-10)
            )
            # Center and scale to O(1)
            features.append((log_u_sia - self.log_u_sia_ref) / 10.0)

            # Sliding velocity proxy in log-space (INCLUDES C!)
            # log|ū_slide| ≈ log(C) + n·log|τ|
            if self.idx_slidingco is not None:
                slidingco = x[..., self.idx_slidingco : self.idx_slidingco + 1]
                log_C = tf.math.log(tf.maximum(slidingco, 1e-30))
            else:
                log_C = tf.constant(
                    np.log(self.C_ref), dtype=tf.float32
                ) * tf.ones_like(thk)

            tau_mag = tf.sqrt(tau_x ** 2 + tau_y ** 2 + 1e-20)
            log_u_slide = log_C + n * tf.math.log(tau_mag + 1e-10)
            features.append((log_u_slide - self.log_u_slide_ref) / 10.0)

            # Flow direction unit vector (ice flows down-gradient)
            ice_mask = tf.cast(thk > 1.0, tf.float32)
            features.append(-dsdx / (grad_s + 1e-10) * ice_mask)
            features.append(-dsdy / (grad_s + 1e-10) * ice_mask)

        if self.physics_level == "full":
            # ═══ FULL: + material maps + curvature (6 more = 16 total) ═══

            # Spatial log-Arrhenius (reveals temperature patterns)
            if self.idx_arrhenius is not None:
                arrhenius = x[..., self.idx_arrhenius : self.idx_arrhenius + 1]
                features.append(
                    (tf.math.log(tf.maximum(arrhenius, 1e-30))
                     - np.log(self.A_ref)) / 5.0
                )
            else:
                features.append(tf.zeros_like(thk))

            # Spatial log-sliding coefficient
            if self.idx_slidingco is not None:
                slidingco = x[..., self.idx_slidingco : self.idx_slidingco + 1]
                features.append(
                    (tf.math.log(tf.maximum(slidingco, 1e-30))
                     - np.log(self.C_ref)) / 5.0
                )
            else:
                features.append(tf.zeros_like(thk))

            # Surface curvature (proxy for longitudinal stress gradients,
            # which the SIA neglects — helps learn corrections)
            inv_dx2 = 1.0 / (dX ** 2 + 1e-10)
            d2sdx2 = (
                tf.roll(usurf, -1, axis=2) - 2.0 * usurf
                + tf.roll(usurf, 1, axis=2)
            ) * inv_dx2
            d2sdy2 = (
                tf.roll(usurf, -1, axis=1) - 2.0 * usurf
                + tf.roll(usurf, 1, axis=1)
            ) * inv_dx2
            # Typical curvature ~ 1e-4 m^-1, scale to O(1)
            features.append(d2sdx2 * 1000.0)
            features.append(d2sdy2 * 1000.0)

            # Thickness × slope interaction (flux proxy)
            features.append(thk * grad_s / (self.H_ref * self.slope_ref))

            # Normalized linear thickness
            features.append(thk / self.H_ref)

        return tf.concat(features, axis=-1)

    def call(self, inputs, training=False):
        x = tf.cast(inputs, tf.float32)
        raw_inputs = x

        if self.input_normalizer is not None:
            x = self.input_normalizer(x, training=training)

        # Physics features (from raw, un-normalized inputs)
        physics = self._compute_physics_features(raw_inputs)

        # Multi-scale learned gradient features (optional)
        if self.multi_scale:
            g3 = tf.nn.gelu(self.grad_conv_3(x))
            g5 = tf.nn.gelu(self.grad_conv_5(x))
            g7 = tf.nn.gelu(self.grad_conv_7(x))
            x = tf.concat([x, physics, g3, g5, g7], axis=-1)
        else:
            x = tf.concat([x, physics], axis=-1)

        # Combine all features to backbone width
        x = tf.nn.gelu(self.combine(x))
        skip = self.skip_proj(x)

        # Backbone CNN with optional residual connections
        for i, conv in enumerate(self.backbone_convs):
            residual = x
            x = tf.nn.gelu(conv(x))
            if self.use_residual and i >= 1 and i % 2 == 1:
                x = x + residual

        x = x + skip
        return self.output_layer(x)
