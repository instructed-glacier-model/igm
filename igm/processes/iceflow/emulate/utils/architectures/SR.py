from __future__ import annotations

from typing import Dict, Any, Tuple

import tensorflow as tf


class SIADecompNet(tf.keras.Model):
    """
    Physics-guided ice-flow emulator with additive decomposition:

        total velocity = sliding + deformation + residual

    Heads
    -----
    sliding
        Depth-uniform plug-like component.
    deformation
        Depth-dependent deformation profile.
    residual
        Depth-dependent correction.

    The model combines:
      1. explicit physics features from raw inputs
      2. learned context features from normalized inputs

    Output layout
    -------------
    [B, H, W, 2*Nz] with

        output[..., :Nz] = Ux(z)
        output[..., Nz:] = Uy(z)

    Internally, velocities are often represented as [B, H, W, Nz, 2].

    Fixed physical constants are part of the architecture, not the
    reconstruction manifest.
    """

    # ------------------------------------------------------------------
    # Fixed architecture constants
    # ------------------------------------------------------------------
    FIXED_N_GLEN = 3.0
    FIXED_RHO = 917.0
    FIXED_G = 9.81

    FIXED_M_SLIDE = 3.0
    FIXED_U_REF = 100.0

    FIXED_EPS = 1e-8
    FIXED_H_REF = 200.0
    FIXED_SLOPE_REF = 0.1
    FIXED_A_REF = 7.6e-24

    FIXED_H_PROXY_FLOOR = 10.0  # stabilizes thin-ice log proxies and inverse gradients

    def __init__(
        self,
        cfg=None,
        nb_inputs=None,
        nb_outputs=None,
        *,
        input_names: list[str] | None = None,
        Nz: int | None = None,
        network_params: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # ------------------------------------------------------------------
        # Dual calling convention:
        #   - cfg-positional: SIADecompNet(cfg, nb_inputs, nb_outputs)
        #   - kwargs (used by .keras reconstruction):
        #     SIADecompNet(input_names=..., Nz=..., network_params=...)
        # No legacy individual-field fallback for this architecture; the cfg
        # path requires a non-empty network.params mapping.
        # ------------------------------------------------------------------
        if cfg is not None:
            from .utils import parse_cfg_input_names_Nz, parse_cfg_network_params_strict
            input_names, Nz = parse_cfg_input_names_Nz(cfg, nb_inputs, nb_outputs)
            network_params = parse_cfg_network_params_strict(cfg, "SIADecompNet")

        if input_names is None or Nz is None or network_params is None:
            raise ValueError(
                "SIADecompNet: must provide either (cfg, nb_inputs, nb_outputs) or "
                "(input_names, Nz, network_params)."
            )

        # ------------------------------------------------------------------
        # Reconstruction inputs
        # ------------------------------------------------------------------
        self.input_names = list(input_names)
        self.Nz = int(Nz)

        self.nb_inputs = len(self.input_names)
        self.nb_outputs = 2 * self.Nz

        # ------------------------------------------------------------------
        # Fixed physical constants
        # ------------------------------------------------------------------
        self.n_glen = float(self.FIXED_N_GLEN)
        self.rho = float(self.FIXED_RHO)
        self.g = float(self.FIXED_G)

        self.m_slide = float(self.FIXED_M_SLIDE)
        self.u_ref = float(self.FIXED_U_REF)

        self.eps_value = float(self.FIXED_EPS)
        self.H_ref_value = float(self.FIXED_H_REF)
        self.slope_ref_value = float(self.FIXED_SLOPE_REF)
        self.A_ref_value = float(self.FIXED_A_REF)
        self.tau_ref_scale_value = (
            self.rho * self.g * self.H_ref_value * self.slope_ref_value
        )

        self.H_proxy_floor_value = float(self.FIXED_H_PROXY_FLOOR)
        self.H_proxy_floor = tf.constant(self.H_proxy_floor_value, dtype=tf.float32)

        self.eps = tf.constant(self.eps_value, dtype=tf.float32)
        self.H_ref = tf.constant(self.H_ref_value, dtype=tf.float32)
        self.slope_ref = tf.constant(self.slope_ref_value, dtype=tf.float32)
        self.tau_ref_scale = tf.constant(self.tau_ref_scale_value, dtype=tf.float32)
        self.A_ref = tf.constant(self.A_ref_value, dtype=tf.float32)

        # ------------------------------------------------------------------
        # Input channel bookkeeping
        # ------------------------------------------------------------------
        self.idx_thk = self.input_names.index("thk")
        self.idx_usurf = self.input_names.index("usurf")

        self.idx_slidingco = (
            self.input_names.index("slidingco")
            if "slidingco" in self.input_names
            else None
        )
        self.idx_arrhenius = (
            self.input_names.index("arrhenius")
            if "arrhenius" in self.input_names
            else None
        )
        if "dX" not in self.input_names:
            raise ValueError(
                "SIADecompNet requires 'dX' in input_names "
                "(grid spacing is read per-pixel from the dX channel)."
            )
        self.idx_dX = self.input_names.index("dX")

        # ------------------------------------------------------------------
        # Input normalizer for the context branch
        # ------------------------------------------------------------------
        self.input_normalizer = None

        # ------------------------------------------------------------------
        # Architecture parameters
        # ------------------------------------------------------------------
        params = dict(network_params)

        allowed_keys = {"nb_layers", "nb_out_filter", "context_dilation_schedule"}
        unexpected = sorted(set(params.keys()) - allowed_keys)
        if unexpected:
            raise ValueError(
                f"Unexpected keys in network_params: {unexpected}. "
                f"Allowed keys are: {sorted(allowed_keys)}"
            )

        if "nb_layers" not in params:
            raise ValueError("network_params must contain 'nb_layers'")
        if "nb_out_filter" not in params:
            raise ValueError("network_params must contain 'nb_out_filter'")
        if "context_dilation_schedule" not in params:
            raise ValueError("network_params must contain 'context_dilation_schedule'")

        self.nb_layers = int(params["nb_layers"])
        self.nb_out_filter = int(params["nb_out_filter"])
        self.context_dilation_schedule = list(params["context_dilation_schedule"])

        self.n_context_blocks = max(2, self.nb_layers // 2)

        if len(self.context_dilation_schedule) != self.n_context_blocks:
            raise ValueError(
                f"context_dilation_schedule must have length {self.n_context_blocks} "
                f"(because nb_layers={self.nb_layers} -> "
                f"n_context_blocks={self.n_context_blocks}), "
                f"but got {len(self.context_dilation_schedule)}"
            )

        # Canonical architecture parameters used for reconstruction
        self.network_params = {
            "nb_layers": int(self.nb_layers),
            "nb_out_filter": int(self.nb_out_filter),
            "context_dilation_schedule": list(self.context_dilation_schedule),
        }

        # ------------------------------------------------------------------
        # Fixed scaling for explicit physics features
        # ------------------------------------------------------------------
        self.log_H_ref = tf.math.log(self.H_ref + 1.0)
        self.log_tau_ref_scale = tf.math.log(self.tau_ref_scale + self.eps)
        self.log_A_ref = tf.math.log(self.A_ref + self.eps)

        self.B_ref = 2.0 * tf.pow(self.A_ref, -1.0 / self.n_glen)
        self.log_B_ref = tf.math.log(self.B_ref + self.eps)

        self.log_u_slide_ref = tf.math.log(
            tf.constant(self.u_ref, dtype=tf.float32) + self.eps
        )
        self.log_u_def_ref = (
            self.log_A_ref
            + (self.n_glen + 1.0) * tf.math.log(self.H_ref + self.H_proxy_floor)
            + self.n_glen * tf.math.log(self.slope_ref + self.eps)
        )

        # ------------------------------------------------------------------
        # Shared context encoder
        # ------------------------------------------------------------------
        self.context_in = tf.keras.layers.Conv2D(
            self.nb_out_filter,
            3,
            padding="same",
            dtype=tf.float32,
            name="context_in",
        )

        self.context_blocks = []
        for i, dilation in enumerate(self.context_dilation_schedule):
            block = {
                "conv1": tf.keras.layers.Conv2D(
                    self.nb_out_filter,
                    3,
                    padding="same",
                    dilation_rate=int(dilation),
                    dtype=tf.float32,
                    name=f"context_block_{i}_conv1",
                ),
                "conv2": tf.keras.layers.Conv2D(
                    self.nb_out_filter,
                    3,
                    padding="same",
                    dilation_rate=int(dilation),
                    dtype=tf.float32,
                    name=f"context_block_{i}_conv2",
                ),
                "act1": tf.keras.layers.Activation(
                    tf.nn.gelu, name=f"context_block_{i}_gelu1"
                ),
                "act2": tf.keras.layers.Activation(
                    tf.nn.gelu, name=f"context_block_{i}_gelu2"
                ),
            }
            self.context_blocks.append(block)

        # ------------------------------------------------------------------
        # Sliding head
        # Output: [B, H, W, 2] = depth-uniform sliding velocity
        # ------------------------------------------------------------------
        self.slide_head_conv1 = tf.keras.layers.Conv2D(
            self.nb_out_filter,
            3,
            padding="same",
            dtype=tf.float32,
            name="slide_head_conv1",
        )
        self.slide_head_act1 = tf.keras.layers.Activation(
            tf.nn.gelu, name="slide_head_gelu1"
        )
        self.slide_head_conv2 = tf.keras.layers.Conv2D(
            self.nb_out_filter,
            3,
            padding="same",
            dtype=tf.float32,
            name="slide_head_conv2",
        )
        self.slide_head_act2 = tf.keras.layers.Activation(
            tf.nn.gelu, name="slide_head_gelu2"
        )
        self.slide_head_out = tf.keras.layers.Conv2D(
            2, 1, padding="same", dtype=tf.float32, name="slide_head_out"
        )

        # ------------------------------------------------------------------
        # Deformation head
        # Output: [B, H, W, 2*Nz] = depth-dependent deformation
        # ------------------------------------------------------------------
        self.def_head_conv1 = tf.keras.layers.Conv2D(
            self.nb_out_filter,
            3,
            padding="same",
            dtype=tf.float32,
            name="def_head_conv1",
        )
        self.def_head_act1 = tf.keras.layers.Activation(
            tf.nn.gelu, name="def_head_gelu1"
        )
        self.def_head_conv2 = tf.keras.layers.Conv2D(
            self.nb_out_filter,
            3,
            padding="same",
            dtype=tf.float32,
            name="def_head_conv2",
        )
        self.def_head_act2 = tf.keras.layers.Activation(
            tf.nn.gelu, name="def_head_gelu2"
        )
        self.def_head_out = tf.keras.layers.Conv2D(
            2 * self.Nz, 1, padding="same", dtype=tf.float32, name="def_head_out"
        )

        # ------------------------------------------------------------------
        # Residual head
        # Output: [B, H, W, 2*Nz] = depth-dependent correction
        # Initialized at zero so the model starts from the structured heads.
        # ------------------------------------------------------------------
        self.res_head_filters = max(self.nb_out_filter // 2, 8)

        self.res_head_conv1 = tf.keras.layers.Conv2D(
            self.res_head_filters,
            3,
            padding="same",
            dtype=tf.float32,
            name="res_head_conv1",
        )
        self.res_head_act1 = tf.keras.layers.Activation(
            tf.nn.gelu, name="res_head_gelu1"
        )
        self.res_head_conv2 = tf.keras.layers.Conv2D(
            self.res_head_filters,
            3,
            padding="same",
            dtype=tf.float32,
            name="res_head_conv2",
        )
        self.res_head_act2 = tf.keras.layers.Activation(
            tf.nn.gelu, name="res_head_gelu2"
        )
        self.res_head_out = tf.keras.layers.Conv2D(
            2 * self.Nz,
            1,
            padding="same",
            dtype=tf.float32,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="res_head_out",
        )

    # ----------------------------------------------------------------------
    # Reconstruction payload
    # ----------------------------------------------------------------------
    def resolved_params(self) -> Dict[str, Any]:
        """Return the constructor payload needed to rebuild the model."""
        return {
            "input_names": list(self.input_names),
            "Nz": int(self.Nz),
            "network_params": {
                "nb_layers": int(self.nb_layers),
                "nb_out_filter": int(self.nb_out_filter),
                "context_dilation_schedule": [
                    int(v) for v in self.context_dilation_schedule
                ],
            },
        }

    # ----------------------------------------------------------------------
    # Deterministic build
    # ----------------------------------------------------------------------
    def build(self, input_shape) -> None:
        """Build all weights in a fixed order using a dummy forward pass."""
        if self.built:
            return

        input_shape = tf.TensorShape(input_shape)
        if input_shape.rank != 4:
            raise ValueError(
                f"SIADecompNet expects input_shape rank 4 [B, H, W, C], got {input_shape}"
            )

        channel_dim = input_shape[-1]
        if channel_dim is None:
            channel_dim = self.nb_inputs
        else:
            channel_dim = int(channel_dim)

        if channel_dim != self.nb_inputs:
            raise ValueError(
                f"Input channel mismatch: model expects {self.nb_inputs} channels "
                f"from input_names={self.input_names}, but build got C={channel_dim}."
            )

        batch_dim = 1 if input_shape[0] is None else int(input_shape[0])
        height_dim = 4 if input_shape[1] is None else int(input_shape[1])
        width_dim = 4 if input_shape[2] is None else int(input_shape[2])

        dummy = tf.zeros(
            shape=(batch_dim, height_dim, width_dim, channel_dim),
            dtype=tf.float32,
        )

        # Build sublayers in forward-pass order.
        _ = self.call(dummy, training=False, return_components=False)

        super().build(input_shape)

    # ----------------------------------------------------------------------
    # Public helpers
    # ----------------------------------------------------------------------
    def set_input_normalizer(self, layer: tf.keras.layers.Layer) -> None:
        """Attach the external normalizer used by the context branch."""
        self.input_normalizer = layer

    # ----------------------------------------------------------------------
    # Tensor utilities
    # ----------------------------------------------------------------------
    def _split_xy_channels(self, uv_flat: tf.Tensor) -> tf.Tensor:
        """Convert [B, H, W, 2*Nz] to [B, H, W, Nz, 2]."""
        ux = uv_flat[..., : self.Nz]
        uy = uv_flat[..., self.Nz :]
        return tf.stack([ux, uy], axis=-1)

    def _merge_xy_channels(self, uv: tf.Tensor) -> tf.Tensor:
        """Convert [B, H, W, Nz, 2] to [B, H, W, 2*Nz]."""
        ux = uv[..., 0]
        uy = uv[..., 1]
        return tf.concat([ux, uy], axis=-1)

    def _broadcast_slide(self, slide_xy: tf.Tensor) -> tf.Tensor:
        """Broadcast [B, H, W, 2] to [B, H, W, Nz, 2]."""
        slide_xy = slide_xy[..., tf.newaxis, :]  # [B, H, W, 1, 2]
        multiples = tf.stack(
            [
                tf.constant(1, dtype=tf.int32),
                tf.constant(1, dtype=tf.int32),
                tf.constant(1, dtype=tf.int32),
                tf.constant(self.Nz, dtype=tf.int32),
                tf.constant(1, dtype=tf.int32),
            ]
        )
        return tf.tile(slide_xy, multiples)

    # ----------------------------------------------------------------------
    # Finite differences with symmetric padding
    # ----------------------------------------------------------------------
    def _get_dx_field(self, x: tf.Tensor) -> tf.Tensor:
        """Return dX with shape [B, H, W, 1]."""
        return tf.cast(x[..., self.idx_dX : self.idx_dX + 1], tf.float32)

    def _central_diff_x(self, field: tf.Tensor, dx: tf.Tensor) -> tf.Tensor:
        """Central difference in x with symmetric padding."""
        fpad = tf.pad(field, [[0, 0], [0, 0], [1, 1], [0, 0]], mode="SYMMETRIC")
        return (fpad[:, :, 2:, :] - fpad[:, :, :-2, :]) / (2.0 * dx + self.eps)

    def _central_diff_y(self, field: tf.Tensor, dx: tf.Tensor) -> tf.Tensor:
        """Central difference in y with symmetric padding."""
        fpad = tf.pad(field, [[0, 0], [1, 1], [0, 0], [0, 0]], mode="SYMMETRIC")
        return (fpad[:, 2:, :, :] - fpad[:, :-2, :, :]) / (2.0 * dx + self.eps)

    def _second_diff_x(self, field: tf.Tensor, dx: tf.Tensor) -> tf.Tensor:
        """Second derivative in x with symmetric padding."""
        fpad = tf.pad(field, [[0, 0], [0, 0], [1, 1], [0, 0]], mode="SYMMETRIC")
        return (fpad[:, :, 2:, :] - 2.0 * field + fpad[:, :, :-2, :]) / (
            dx * dx + self.eps
        )

    def _second_diff_y(self, field: tf.Tensor, dx: tf.Tensor) -> tf.Tensor:
        """Second derivative in y with symmetric padding."""
        fpad = tf.pad(field, [[0, 0], [1, 1], [0, 0], [0, 0]], mode="SYMMETRIC")
        return (fpad[:, 2:, :, :] - 2.0 * field + fpad[:, :-2, :, :]) / (
            dx * dx + self.eps
        )

    # ----------------------------------------------------------------------
    # Explicit physics features from raw inputs
    # ----------------------------------------------------------------------
    def _physics_features(
        self, raw_inputs: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, Dict[str, tf.Tensor]]:
        """Build explicit physics features for the three heads."""
        x = tf.cast(raw_inputs, tf.float32)

        H = tf.maximum(x[..., self.idx_thk : self.idx_thk + 1], 0.0)
        s = x[..., self.idx_usurf : self.idx_usurf + 1]
        dx = self._get_dx_field(x)

        # Smooth thickness only for bed-gradient features to suppress pixel-scale DA spikes.
        # Use a 3x3 binomial blur with reflect padding.
        binomial_kernel = tf.constant(
            [[1.0, 2.0, 1.0],
            [2.0, 4.0, 2.0],
            [1.0, 2.0, 1.0]],
            dtype=tf.float32,
        ) / 16.0
        binomial_kernel = binomial_kernel[:, :, tf.newaxis, tf.newaxis]  # [3, 3, 1, 1]

        H_for_bed_grad = tf.nn.depthwise_conv2d(
            tf.pad(H, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT"),
            binomial_kernel,
            strides=[1, 1, 1, 1],
            padding="VALID",
        )

        b = s - H
        b_for_bed_grad = s - H_for_bed_grad

        tau_ref = None
        if self.idx_slidingco is not None:
            tau_ref = tf.maximum(
                x[..., self.idx_slidingco : self.idx_slidingco + 1], self.eps
            )

        A = None
        if self.idx_arrhenius is not None:
            A = tf.maximum(
                x[..., self.idx_arrhenius : self.idx_arrhenius + 1], self.eps
            )

        # ------------------------------------------------------------------
        # Geometry
        # ------------------------------------------------------------------
        dsdx = self._central_diff_x(s, dx)
        dsdy = self._central_diff_y(s, dx)
        dbdx = self._central_diff_x(b_for_bed_grad, dx)
        dbdy = self._central_diff_y(b_for_bed_grad, dx)

        d2sdx2 = self._second_diff_x(s, dx)
        d2sdy2 = self._second_diff_y(s, dx)

        grad_s = tf.sqrt(dsdx**2 + dsdy**2 + self.eps)
        grad_b = tf.sqrt(dbdx**2 + dbdy**2 + self.eps)

        # ------------------------------------------------------------------
        # Driving-stress proxy
        # ------------------------------------------------------------------
        tau_dx = -self.rho * self.g * H * dsdx
        tau_dy = -self.rho * self.g * H * dsdy
        tau_d = tf.sqrt(tau_dx**2 + tau_dy**2 + self.eps)

        # ------------------------------------------------------------------
        # Log-safe stress proxy
        # ------------------------------------------------------------------
        tau_dx = -self.rho * self.g * H * dsdx
        tau_dy = -self.rho * self.g * H * dsdy
        tau_d = tf.sqrt(tau_dx**2 + tau_dy**2 + self.eps)

        # Use floored thickness only in log-based proxies.
        # Keep linear stress features on raw H.
        H_for_log_proxy = H + self.H_proxy_floor
        tau_dx_for_log = -self.rho * self.g * H_for_log_proxy * dsdx
        tau_dy_for_log = -self.rho * self.g * H_for_log_proxy * dsdy
        tau_d_for_log = tf.sqrt(tau_dx_for_log**2 + tau_dy_for_log**2 + self.eps)

        # ------------------------------------------------------------------
        # Fixed scaling to O(1)
        # ------------------------------------------------------------------
        log_H = (tf.math.log(H + 1.0) - self.log_H_ref) / 3.0
        H_lin = H / (self.H_ref + self.eps)
        H_grad_interaction = (H * grad_s) / (self.H_ref * self.slope_ref + self.eps)

        dsdx_n = dsdx / self.slope_ref
        dsdy_n = dsdy / self.slope_ref
        dbdx_n = dbdx / self.slope_ref
        dbdy_n = dbdy / self.slope_ref
        grad_s_n = grad_s / self.slope_ref
        grad_b_n = grad_b / self.slope_ref

        tau_dx_n = tau_dx / (self.tau_ref_scale + self.eps)
        tau_dy_n = tau_dy / (self.tau_ref_scale + self.eps)
        tau_d_n = tau_d / (self.tau_ref_scale + self.eps)

        log_tau_d_raw = tf.math.log(tau_d_for_log + self.eps)
        log_tau_d = (log_tau_d_raw - self.log_tau_ref_scale) / 5.0

        # Downhill unit vector, masked off ice
        ice_mask = tf.cast(H > 1.0, tf.float32)
        dir_x = -dsdx / (grad_s + self.eps) * ice_mask
        dir_y = -dsdy / (grad_s + self.eps) * ice_mask

        # Curvature channels for the residual stack
        curv_x_n = d2sdx2 * 1000.0
        curv_y_n = d2sdy2 * 1000.0

        # ------------------------------------------------------------------
        # Sliding-head features
        # ------------------------------------------------------------------
        slide_feats = [
            log_H,
            dsdx_n,
            dsdy_n,
            grad_s_n,
            dbdx_n,
            dbdy_n,
            grad_b_n,
            tau_dx_n,
            tau_dy_n,
            tau_d_n,
            log_tau_d,
            dir_x,
            dir_y,
        ]

        log_tau_ref = None
        log_u_slide_proxy = None

        if tau_ref is not None:
            log_tau_ref_raw = tf.math.log(tau_ref + self.eps)
            log_tau_ref = (log_tau_ref_raw - self.log_tau_ref_scale) / 5.0

            log_u_slide_proxy_raw = (
                tf.math.log(tf.constant(self.u_ref, dtype=tf.float32) + self.eps)
                + self.m_slide * (log_tau_d_raw - log_tau_ref_raw)
            )
            log_u_slide_proxy = (
                log_u_slide_proxy_raw - self.log_u_slide_ref
            ) / 5.0

            slide_feats.extend([log_tau_ref, log_u_slide_proxy])
        else:
            log_tau_ref_raw = None
            log_u_slide_proxy_raw = None

        slide_feats = tf.concat(slide_feats, axis=-1)

        # ------------------------------------------------------------------
        # Deformation-head features
        # ------------------------------------------------------------------
        def_feats = [
            log_H,
            dsdx_n,
            dsdy_n,
            grad_s_n,
            tau_dx_n,
            tau_dy_n,
            tau_d_n,
            log_tau_d,
            H_lin,
            H_grad_interaction,
        ]

        log_A = None
        log_B = None
        log_u_def_proxy = None

        if A is not None:
            log_A_raw = tf.math.log(A + self.eps)
            B = 2.0 * tf.pow(A, -1.0 / self.n_glen)
            log_B_raw = tf.math.log(B + self.eps)

            log_u_def_proxy_raw = (
                log_A_raw
                + (self.n_glen + 1.0) * tf.math.log(H_for_log_proxy)
                + self.n_glen * tf.math.log(grad_s + self.eps)
            )

            log_A = (log_A_raw - self.log_A_ref) / 5.0
            log_B = (log_B_raw - self.log_B_ref) / 5.0
            log_u_def_proxy = (log_u_def_proxy_raw - self.log_u_def_ref) / 5.0

            def_feats.extend([log_A, log_B, log_u_def_proxy])
        else:
            log_A_raw = None
            log_B_raw = None
            log_u_def_proxy_raw = None

        def_feats = tf.concat(def_feats, axis=-1)

        # ------------------------------------------------------------------
        # Residual-head features
        # ------------------------------------------------------------------
        all_feats = tf.concat([slide_feats, def_feats, curv_x_n, curv_y_n], axis=-1)

        aux = {
            "H": H,
            "s": s,
            "b": b,
            "dx": dx,
            "dsdx": dsdx,
            "dsdy": dsdy,
            "dbdx": dbdx,
            "dbdy": dbdy,
            "d2sdx2": d2sdx2,
            "d2sdy2": d2sdy2,
            "grad_s": grad_s,
            "grad_b": grad_b,
            "tau_dx": tau_dx,
            "tau_dy": tau_dy,
            "tau_d": tau_d,
            "tau_d_for_log": tau_d_for_log,
            "dir_x": dir_x,
            "dir_y": dir_y,
            "H_for_bed_grad": H_for_bed_grad,
            "b_for_bed_grad": b_for_bed_grad,
            "H_for_log_proxy": H_for_log_proxy,
        }

        if tau_ref is not None:
            aux["tau_ref"] = tau_ref
            aux["log_tau_ref"] = log_tau_ref_raw
            aux["log_u_slide_proxy"] = log_u_slide_proxy_raw

        if A is not None:
            aux["A"] = A
            aux["log_A"] = log_A_raw
            aux["log_B"] = log_B_raw
            aux["log_u_def_proxy"] = log_u_def_proxy_raw

        return slide_feats, def_feats, all_feats, aux

    # ----------------------------------------------------------------------
    # Learned context branch
    # ----------------------------------------------------------------------
    def _context_features(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:
        """Compute learned context features from normalized inputs."""
        x = tf.cast(inputs, tf.float32)

        if self.input_normalizer is not None:
            x = self.input_normalizer(x, training=training)

        h = self.context_in(x)
        h = tf.nn.gelu(h)

        for block in self.context_blocks:
            residual = h
            h = block["conv1"](h)
            h = block["act1"](h)
            h = block["conv2"](h)
            h = block["act2"](h)
            h = h + residual

        return h

    # ----------------------------------------------------------------------
    # Forward pass
    # ----------------------------------------------------------------------
    def call(
        self,
        inputs: tf.Tensor,
        training: bool = False,
        return_components: bool = False,
    ) -> tf.Tensor | Dict[str, Any]:
        """Return total velocity, or the decomposed components if requested."""
        raw_inputs = tf.cast(inputs, tf.float32)

        # 1. Explicit physics features from raw inputs
        slide_phys, def_phys, all_phys, aux = self._physics_features(raw_inputs)

        # 2. Learned context features from normalized inputs
        context = self._context_features(raw_inputs, training=training)

        # 3. Sliding head
        slide_in = tf.concat([slide_phys, context], axis=-1)
        slide_h = self.slide_head_conv1(slide_in)
        slide_h = self.slide_head_act1(slide_h)
        slide_h = self.slide_head_conv2(slide_h)
        slide_h = self.slide_head_act2(slide_h)
        slide_xy = self.slide_head_out(slide_h)  # [B, H, W, 2]

        # 4. Deformation head
        def_in = tf.concat([def_phys, context], axis=-1)
        def_h = self.def_head_conv1(def_in)
        def_h = self.def_head_act1(def_h)
        def_h = self.def_head_conv2(def_h)
        def_h = self.def_head_act2(def_h)
        def_flat = self.def_head_out(def_h)  # [B, H, W, 2*Nz]
        def_uv = self._split_xy_channels(def_flat)  # [B, H, W, Nz, 2]

        # 5. Residual head
        res_in = tf.concat([all_phys, context], axis=-1)
        res_h = self.res_head_conv1(res_in)
        res_h = self.res_head_act1(res_h)
        res_h = self.res_head_conv2(res_h)
        res_h = self.res_head_act2(res_h)
        res_flat = self.res_head_out(res_h)  # [B, H, W, 2*Nz]
        res_uv = self._split_xy_channels(res_flat)  # [B, H, W, Nz, 2]

        # 6. Broadcast sliding over depth and sum the three components
        slide_uv = self._broadcast_slide(slide_xy)  # [B, H, W, Nz, 2]
        total_uv = slide_uv + def_uv + res_uv

        # Flatten back to [B, H, W, 2*Nz]
        total_flat = self._merge_xy_channels(total_uv)

        if not return_components:
            return total_flat

        return {
            "total": total_flat,
            "total_uv": total_uv,
            "slide_xy": slide_xy,
            "slide_uv": slide_uv,
            "deformation_uv": def_uv,
            "residual_uv": res_uv,
            "physics_aux": aux,
        }