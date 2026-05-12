from __future__ import annotations

from typing import Optional, Dict, Any, Tuple, List

import tensorflow as tf


class SIADecompNetV2(tf.keras.Model):
    """
    Extension of SIADecompNet with independently configurable heads.

    The context encoder (shared trunk) is identical to SIADecompNet.  The
    three prediction heads — sliding, deformation, and residual — each accept
    their own ``*_filters`` and ``*_layers`` parameters so that head capacity
    can be tuned independently of the shared encoder.

    ------------------------------------------------------------------
    New network_params keys (all optional; defaults reproduce the
    original SIADecompNet behaviour)
    ------------------------------------------------------------------

    slide_head_filters : int, default nb_out_filter
        Width of the sliding-velocity head.
    slide_head_layers : int, default 2
        Number of Conv2D+GELU pairs in the sliding head.

    def_head_filters : int, default nb_out_filter
        Width of the deformation head.
    def_head_layers : int, default 2
        Number of Conv2D+GELU pairs in the deformation head.

    res_head_filters : int, default max(nb_out_filter // 2, 8)
        Width of the residual head.
    res_head_layers : int, default 2
        Number of Conv2D+GELU pairs in the residual head.

    ------------------------------------------------------------------
    All other parameters and behaviour are identical to SIADecompNet.
    See that class for full documentation.
    ------------------------------------------------------------------
    """

    # ------------------------------------------------------------------
    # Fixed architecture constants  (identical to SIADecompNet)
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

    FIXED_H_PROXY_FLOOR = 10.0

    def __init__(
        self,
        *,
        input_names: list[str],
        Nz: int,
        network_params: dict[str, Any],
        dx_const: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # ------------------------------------------------------------------
        # Minimal reconstruction inputs
        # ------------------------------------------------------------------
        self.input_names = list(input_names)
        self.Nz = int(Nz)

        self.nb_inputs = len(self.input_names)
        self.nb_outputs = 2 * self.Nz

        # ------------------------------------------------------------------
        # Fixed physics constants
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
        self.idx_dX = self.input_names.index("dX") if "dX" in self.input_names else None

        if self.idx_dX is None:
            self.dx_const_value = 90.0 if dx_const is None else float(dx_const)
            self.dx_const = tf.constant(self.dx_const_value, dtype=tf.float32)
        else:
            self.dx_const_value = None
            self.dx_const = None

        # ------------------------------------------------------------------
        # Learned normalizer for the context branch
        # ------------------------------------------------------------------
        self.input_normalizer = None

        # ------------------------------------------------------------------
        # Parse and validate network_params
        # ------------------------------------------------------------------
        params = dict(network_params)

        allowed_keys = {
            "nb_layers",
            "nb_out_filter",
            "context_dilation_schedule",
            "slide_head_filters",
            "slide_head_layers",
            "def_head_filters",
            "def_head_layers",
            "res_head_filters",
            "res_head_layers",
            "anchor_deformation_at_bed",
            "zero_mean_residual_over_depth",
        }

        # ------------------------------------------------------------------
        # Settings that might help identifiability such that each head
        # focuses on its intended component of the solution.  These are
        # not strictly necessary but may help training stability and
        # interpretability.
        # ------------------------------------------------------------------
        self.anchor_deformation_at_bed = bool(params.get("anchor_deformation_at_bed", True))
        self.zero_mean_residual_over_depth = bool(
            params.get("zero_mean_residual_over_depth", True)
        )
        self.bed_index = int(0)   # TO DO: for different vertical discretizations this may not work
        unexpected = sorted(set(params.keys()) - allowed_keys)
        if unexpected:
            raise ValueError(
                f"Unexpected keys in network_params: {unexpected}. "
                f"Allowed keys are: {sorted(allowed_keys)}"
            )

        for required in ("nb_layers", "nb_out_filter", "context_dilation_schedule"):
            if required not in params:
                raise ValueError(f"network_params must contain '{required}'")

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

        # Per-head parameters with defaults matching original SIADecompNet
        self.slide_head_n_filters = int(
            params.get("slide_head_filters", self.nb_out_filter)
        )
        self.slide_head_n_layers = int(params.get("slide_head_layers", 2))

        self.def_head_n_filters = int(
            params.get("def_head_filters", self.nb_out_filter)
        )
        self.def_head_n_layers = int(params.get("def_head_layers", 2))

        self.res_head_n_filters = int(
            params.get("res_head_filters", max(self.nb_out_filter // 2, 8))
        )
        self.res_head_n_layers = int(params.get("res_head_layers", 2))

        self.network_params = {
            "nb_layers": int(self.nb_layers),
            "nb_out_filter": int(self.nb_out_filter),
            "context_dilation_schedule": list(self.context_dilation_schedule),
            "slide_head_filters": int(self.slide_head_n_filters),
            "slide_head_layers": int(self.slide_head_n_layers),
            "def_head_filters": int(self.def_head_n_filters),
            "def_head_layers": int(self.def_head_n_layers),
            "res_head_filters": int(self.res_head_n_filters),
            "res_head_layers": int(self.res_head_n_layers),
            "anchor_deformation_at_bed": bool(self.anchor_deformation_at_bed),
            "zero_mean_residual_over_depth": bool(self.zero_mean_residual_over_depth),
        }

        # ------------------------------------------------------------------
        # Fixed physics scaling / centering
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
        # Shared context encoder  (identical structure to SIADecompNet)
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
        # Sliding head  — configurable depth and width
        # Output: [B, H, W, 2] = [ubx, uby]
        # ------------------------------------------------------------------
        self.slide_head_body = self._make_head_body(
            "slide_head", self.slide_head_n_layers, self.slide_head_n_filters
        )
        self.slide_head_out = tf.keras.layers.Conv2D(
            2, 1, padding="same", dtype=tf.float32, name="slide_head_out"
        )

        # ------------------------------------------------------------------
        # Deformation head  — configurable depth and width
        # Output: [B, H, W, 2*Nz]
        # ------------------------------------------------------------------
        self.def_head_body = self._make_head_body(
            "def_head", self.def_head_n_layers, self.def_head_n_filters
        )
        self.def_head_out = tf.keras.layers.Conv2D(
            2 * self.Nz, 1, padding="same", dtype=tf.float32, name="def_head_out"
        )

        # ------------------------------------------------------------------
        # Residual head  — configurable depth and width
        # Output: [B, H, W, 2*Nz]
        # Zero-initialised output so it starts inactive.
        # ------------------------------------------------------------------
        self.res_head_body = self._make_head_body(
            "res_head", self.res_head_n_layers, self.res_head_n_filters
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
    # Head construction helper
    # ----------------------------------------------------------------------
    def _make_head_body(
        self, name: str, n_layers: int, filters: int
    ) -> tf.keras.Sequential:
        """
        Build ``n_layers`` Conv2D(filters, 3×3) + GELU pairs for one head,
        returned as a ``tf.keras.Sequential``.

        """
        body: List[tf.keras.layers.Layer] = []
        for i in range(n_layers):
            body.append(
                tf.keras.layers.Conv2D(
                    filters,
                    3,
                    padding="same",
                    dtype=tf.float32,
                    name=f"{name}_conv{i + 1}",
                )
            )
            body.append(
                tf.keras.layers.Activation(
                    tf.nn.gelu, name=f"{name}_gelu{i + 1}"
                )
            )
        return tf.keras.Sequential(body, name=f"{name}_body")

    @staticmethod
    def _apply_head_body(
        body: tf.keras.Sequential, x: tf.Tensor
    ) -> tf.Tensor:
        return body(x)

    # ----------------------------------------------------------------------
    # Reconstruction manifest
    # ----------------------------------------------------------------------
    def resolved_params(self) -> Dict[str, Any]:
        return {
            "input_names": list(self.input_names),
            "Nz": int(self.Nz),
            "network_params": {
                "nb_layers": int(self.nb_layers),
                "nb_out_filter": int(self.nb_out_filter),
                "context_dilation_schedule": [
                    int(v) for v in self.context_dilation_schedule
                ],
                "slide_head_filters": int(self.slide_head_n_filters),
                "slide_head_layers": int(self.slide_head_n_layers),
                "def_head_filters": int(self.def_head_n_filters),
                "def_head_layers": int(self.def_head_n_layers),
                "res_head_filters": int(self.res_head_n_filters),
                "res_head_layers": int(self.res_head_n_layers),
                "anchor_deformation_at_bed": bool(self.anchor_deformation_at_bed),
                "zero_mean_residual_over_depth": bool(self.zero_mean_residual_over_depth),
            },
            "dx_const": None
            if self.dx_const_value is None
            else float(self.dx_const_value),
        }

    # ----------------------------------------------------------------------
    # Keras build
    # ----------------------------------------------------------------------
    def build(self, input_shape) -> None:
        if self.built:
            return

        input_shape = tf.TensorShape(input_shape)
        if input_shape.rank != 4:
            raise ValueError(
                f"SIADecompNetV2 expects input_shape rank 4 [B, H, W, C], "
                f"got {input_shape}"
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
        _ = self.call(dummy, training=False, return_components=False)
        super().build(input_shape)

    # ----------------------------------------------------------------------
    # Public helpers
    # ----------------------------------------------------------------------
    def set_input_normalizer(self, layer: tf.keras.layers.Layer) -> None:
        self.input_normalizer = layer

    # ----------------------------------------------------------------------
    # Tensor utilities  (identical to SIADecompNet)
    # ----------------------------------------------------------------------
    def _split_xy_channels(self, uv_flat: tf.Tensor) -> tf.Tensor:
        ux = uv_flat[..., : self.Nz]
        uy = uv_flat[..., self.Nz :]
        return tf.stack([ux, uy], axis=-1)

    def _merge_xy_channels(self, uv: tf.Tensor) -> tf.Tensor:
        ux = uv[..., 0]
        uy = uv[..., 1]
        return tf.concat([ux, uy], axis=-1)

    def _broadcast_slide(self, slide_xy: tf.Tensor) -> tf.Tensor:
        slide_xy = slide_xy[..., tf.newaxis, :]
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
    # Finite differences  (identical to SIADecompNet)
    # ----------------------------------------------------------------------
    def _get_dx_field(self, x: tf.Tensor) -> tf.Tensor:
        if self.idx_dX is not None:
            return tf.cast(x[..., self.idx_dX : self.idx_dX + 1], tf.float32)
        thk = x[..., self.idx_thk : self.idx_thk + 1]
        return tf.ones_like(thk, dtype=tf.float32) * self.dx_const

    def _central_diff_x(self, field: tf.Tensor, dx: tf.Tensor) -> tf.Tensor:
        fpad = tf.pad(field, [[0, 0], [0, 0], [1, 1], [0, 0]], mode="SYMMETRIC")
        return (fpad[:, :, 2:, :] - fpad[:, :, :-2, :]) / (2.0 * dx + self.eps)

    def _central_diff_y(self, field: tf.Tensor, dx: tf.Tensor) -> tf.Tensor:
        fpad = tf.pad(field, [[0, 0], [1, 1], [0, 0], [0, 0]], mode="SYMMETRIC")
        return (fpad[:, 2:, :, :] - fpad[:, :-2, :, :]) / (2.0 * dx + self.eps)

    def _second_diff_x(self, field: tf.Tensor, dx: tf.Tensor) -> tf.Tensor:
        fpad = tf.pad(field, [[0, 0], [0, 0], [1, 1], [0, 0]], mode="SYMMETRIC")
        return (fpad[:, :, 2:, :] - 2.0 * field + fpad[:, :, :-2, :]) / (
            dx * dx + self.eps
        )

    def _second_diff_y(self, field: tf.Tensor, dx: tf.Tensor) -> tf.Tensor:
        fpad = tf.pad(field, [[0, 0], [1, 1], [0, 0], [0, 0]], mode="SYMMETRIC")
        return (fpad[:, 2:, :, :] - 2.0 * field + fpad[:, :-2, :, :]) / (
            dx * dx + self.eps
        )

    # ----------------------------------------------------------------------
    # Explicit physics features  (identical to SIADecompNet)
    # ----------------------------------------------------------------------
    def _physics_features(
        self, raw_inputs: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, Dict[str, tf.Tensor]]:
        x = tf.cast(raw_inputs, tf.float32)

        H = tf.maximum(x[..., self.idx_thk : self.idx_thk + 1], 0.0)
        s = x[..., self.idx_usurf : self.idx_usurf + 1]
        dx = self._get_dx_field(x)

        binomial_kernel = tf.constant(
            [[1.0, 2.0, 1.0],
             [2.0, 4.0, 2.0],
             [1.0, 2.0, 1.0]],
            dtype=tf.float32,
        ) / 16.0
        binomial_kernel = binomial_kernel[:, :, tf.newaxis, tf.newaxis]

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

        dsdx = self._central_diff_x(s, dx)
        dsdy = self._central_diff_y(s, dx)
        dbdx = self._central_diff_x(b_for_bed_grad, dx)
        dbdy = self._central_diff_y(b_for_bed_grad, dx)

        d2sdx2 = self._second_diff_x(s, dx)
        d2sdy2 = self._second_diff_y(s, dx)

        grad_s = tf.sqrt(dsdx**2 + dsdy**2 + self.eps)
        grad_b = tf.sqrt(dbdx**2 + dbdy**2 + self.eps)

        tau_dx = -self.rho * self.g * H * dsdx
        tau_dy = -self.rho * self.g * H * dsdy
        tau_d = tf.sqrt(tau_dx**2 + tau_dy**2 + self.eps)

        H_for_log_proxy = H + self.H_proxy_floor
        tau_dx_for_log = -self.rho * self.g * H_for_log_proxy * dsdx
        tau_dy_for_log = -self.rho * self.g * H_for_log_proxy * dsdy
        tau_d_for_log = tf.sqrt(
            tau_dx_for_log**2 + tau_dy_for_log**2 + self.eps
        )

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

        ice_mask = tf.cast(H > 1.0, tf.float32)
        dir_x = -dsdx / (grad_s + self.eps) * ice_mask
        dir_y = -dsdy / (grad_s + self.eps) * ice_mask

        curv_x_n = d2sdx2 * 1000.0
        curv_y_n = d2sdy2 * 1000.0

        slide_feats = [
            log_H, dsdx_n, dsdy_n, grad_s_n,
            dbdx_n, dbdy_n, grad_b_n,
            tau_dx_n, tau_dy_n, tau_d_n, log_tau_d,
            dir_x, dir_y,
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
            log_u_slide_proxy = (log_u_slide_proxy_raw - self.log_u_slide_ref) / 5.0
            slide_feats.extend([log_tau_ref, log_u_slide_proxy])
        else:
            log_tau_ref_raw = None
            log_u_slide_proxy_raw = None

        slide_feats = tf.concat(slide_feats, axis=-1)

        def_feats = [
            log_H, dsdx_n, dsdy_n, grad_s_n,
            tau_dx_n, tau_dy_n, tau_d_n, log_tau_d,
            H_lin, H_grad_interaction,
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

        all_feats = tf.concat([slide_feats, def_feats, curv_x_n, curv_y_n], axis=-1)

        aux = {
            "H": H, "s": s, "b": b, "dx": dx,
            "dsdx": dsdx, "dsdy": dsdy, "dbdx": dbdx, "dbdy": dbdy,
            "d2sdx2": d2sdx2, "d2sdy2": d2sdy2,
            "grad_s": grad_s, "grad_b": grad_b,
            "tau_dx": tau_dx, "tau_dy": tau_dy, "tau_d": tau_d,
            "tau_d_for_log": tau_d_for_log,
            "dir_x": dir_x, "dir_y": dir_y,
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
    # Learned context encoder  (identical to SIADecompNet)
    # ----------------------------------------------------------------------
    def _context_features(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:
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
        raw_inputs = tf.cast(inputs, tf.float32)

        slide_phys, def_phys, all_phys, aux = self._physics_features(raw_inputs)
        context = self._context_features(raw_inputs, training=training)

        # Sliding head
        slide_h = self._apply_head_body(
            self.slide_head_body, tf.concat([slide_phys, context], axis=-1)
        )
        slide_xy = self.slide_head_out(slide_h)          # [B, H, W, 2]

        # Deformation head
        def_h = self._apply_head_body(
            self.def_head_body, tf.concat([def_phys, context], axis=-1)
        )
        def_flat = self.def_head_out(def_h)              # [B, H, W, 2*Nz]
        def_uv = self._split_xy_channels(def_flat)       # [B, H, W, Nz, 2]
        if self.anchor_deformation_at_bed:
            def_uv = def_uv - def_uv[..., self.bed_index:self.bed_index+1, :]

        # Residual head
        res_h = self._apply_head_body(
            self.res_head_body, tf.concat([all_phys, context], axis=-1)
        )
        res_flat = self.res_head_out(res_h)              # [B, H, W, 2*Nz]
        res_uv = self._split_xy_channels(res_flat)       # [B, H, W, Nz, 2]
        if self.zero_mean_residual_over_depth:
            res_uv = res_uv - tf.reduce_mean(res_uv, axis=-2, keepdims=True)

        # Combine
        slide_uv = self._broadcast_slide(slide_xy)       # [B, H, W, Nz, 2]
        total_uv = slide_uv + def_uv + res_uv
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


class SIADecompNetV2SharedHead(SIADecompNetV2):
    """
    Shared-spatial-body variant of SIADecompNetV2.

    This architecture keeps the same explicit physics decomposition as
    ``SIADecompNetV2`` at the output level:

      total = broadcast(sliding) + anchored_deformation + zero_mean_residual

    but removes the three separate spatial head bodies.  All learned spatial
    processing after input normalization happens in one shared dilated context
    encoder.  The three component heads are deliberately lightweight 1x1
    physics-conditioned projections:

      slide head: shared context + slide-relevant physics -> 2 channels
      def head:   shared context + deformation-relevant physics -> 2*Nz channels
      res head:   shared context + all physics features -> 2*Nz channels

    The point is to test whether the SIADV2 decomposition needs separate
    component-specific spatial processing, or whether the decomposition can be
    enforced mostly through output constraints plus local physics conditioning.

    network_params
    --------------
    nb_layers : int
        Nominal context depth.  As in SIADecompNetV2, this creates
        ``max(2, nb_layers // 2)`` residual context blocks, each with two
        3x3 convolutions.
    nb_out_filter : int
        Width of the shared context encoder.
    context_dilation_schedule : list[int]
        One dilation value per residual context block.
    head_filters : int, optional
        Width of the optional per-head 1x1 bottleneck.  Defaults to
        ``max(nb_out_filter // 2, 32)``.
    head_layers : int, optional
        Number of Conv2D(1x1)+GELU bottleneck layers before each output
        projection.  Defaults to 1.  Set to 0 for pure linear projections.
    anchor_deformation_at_bed : bool, optional
        Same constraint as SIADecompNetV2.  Default True.
    zero_mean_residual_over_depth : bool, optional
        Same constraint as SIADecompNetV2.  Default True.
    """

    def __init__(
        self,
        *,
        input_names: list[str],
        Nz: int,
        network_params: dict[str, Any],
        dx_const: Optional[float] = None,
        **kwargs,
    ):
        tf.keras.Model.__init__(self, **kwargs)

        # ------------------------------------------------------------------
        # Minimal reconstruction inputs
        # ------------------------------------------------------------------
        self.input_names = list(input_names)
        self.Nz = int(Nz)
        self.nb_inputs = len(self.input_names)
        self.nb_outputs = 2 * self.Nz

        # ------------------------------------------------------------------
        # Fixed physics constants; keep bitwise-compatible definitions with
        # SIADecompNetV2 so _physics_features can be inherited unchanged.
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
        self.idx_dX = self.input_names.index("dX") if "dX" in self.input_names else None

        if self.idx_dX is None:
            self.dx_const_value = 90.0 if dx_const is None else float(dx_const)
            self.dx_const = tf.constant(self.dx_const_value, dtype=tf.float32)
        else:
            self.dx_const_value = None
            self.dx_const = None

        self.input_normalizer = None

        # ------------------------------------------------------------------
        # Parse and validate the deliberately small configuration surface.
        # ------------------------------------------------------------------
        params = dict(network_params)
        allowed_keys = {
            "nb_layers",
            "nb_out_filter",
            "context_dilation_schedule",
            "head_filters",
            "head_layers",
            "anchor_deformation_at_bed",
            "zero_mean_residual_over_depth",
        }
        unexpected = sorted(set(params.keys()) - allowed_keys)
        if unexpected:
            raise ValueError(
                f"Unexpected keys in network_params: {unexpected}. "
                f"Allowed keys are: {sorted(allowed_keys)}"
            )

        for required in ("nb_layers", "nb_out_filter", "context_dilation_schedule"):
            if required not in params:
                raise ValueError(f"network_params must contain '{required}'")

        self.nb_layers = int(params["nb_layers"])
        self.nb_out_filter = int(params["nb_out_filter"])
        self.context_dilation_schedule = [
            int(v) for v in params["context_dilation_schedule"]
        ]
        self.n_context_blocks = max(2, self.nb_layers // 2)

        if len(self.context_dilation_schedule) != self.n_context_blocks:
            raise ValueError(
                f"context_dilation_schedule must have length {self.n_context_blocks} "
                f"(because nb_layers={self.nb_layers} -> "
                f"n_context_blocks={self.n_context_blocks}), "
                f"but got {len(self.context_dilation_schedule)}"
            )

        self.head_filters = int(params.get("head_filters", max(self.nb_out_filter // 2, 32)))
        self.head_layers = int(params.get("head_layers", 1))
        if self.head_layers < 0:
            raise ValueError("head_layers must be >= 0")
        if self.head_layers > 0 and self.head_filters <= 0:
            raise ValueError("head_filters must be > 0 when head_layers > 0")

        self.anchor_deformation_at_bed = bool(params.get("anchor_deformation_at_bed", True))
        self.zero_mean_residual_over_depth = bool(
            params.get("zero_mean_residual_over_depth", True)
        )
        self.bed_index = int(0)

        self.network_params = {
            "nb_layers": int(self.nb_layers),
            "nb_out_filter": int(self.nb_out_filter),
            "context_dilation_schedule": list(self.context_dilation_schedule),
            "head_filters": int(self.head_filters),
            "head_layers": int(self.head_layers),
            "anchor_deformation_at_bed": bool(self.anchor_deformation_at_bed),
            "zero_mean_residual_over_depth": bool(self.zero_mean_residual_over_depth),
        }

        # ------------------------------------------------------------------
        # Fixed physics scaling / centering used by inherited _physics_features.
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
        # One shared spatial context body.
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
        # Lightweight local component heads.  Each head sees shared context plus
        # only its relevant physics conditioning.  The residual head sees the
        # full physics set because it is explicitly tasked with correcting what
        # the slide/deformation proxies miss.
        # ------------------------------------------------------------------
        self.slide_head = self._make_pointwise_head(
            "slide_head", out_channels=2, zero_init=False
        )
        self.def_head = self._make_pointwise_head(
            "def_head", out_channels=2 * self.Nz, zero_init=False
        )
        self.res_head = self._make_pointwise_head(
            "res_head", out_channels=2 * self.Nz, zero_init=True
        )

    def _make_pointwise_head(
        self,
        name: str,
        *,
        out_channels: int,
        zero_init: bool,
    ) -> tf.keras.Sequential:
        layers: List[tf.keras.layers.Layer] = []
        for i in range(self.head_layers):
            layers.append(
                tf.keras.layers.Conv2D(
                    self.head_filters,
                    1,
                    padding="same",
                    dtype=tf.float32,
                    name=f"{name}_pw{i + 1}",
                )
            )
            layers.append(tf.keras.layers.Activation(tf.nn.gelu, name=f"{name}_gelu{i + 1}"))

        initializer = "zeros" if zero_init else "glorot_uniform"
        layers.append(
            tf.keras.layers.Conv2D(
                out_channels,
                1,
                padding="same",
                dtype=tf.float32,
                kernel_initializer=initializer,
                bias_initializer="zeros" if zero_init else "zeros",
                name=f"{name}_out",
            )
        )
        return tf.keras.Sequential(layers, name=name)

    def resolved_params(self) -> Dict[str, Any]:
        return {
            "input_names": [str(v) for v in self.input_names],
            "Nz": int(self.Nz),
            "network_params": {
                "nb_layers": int(self.nb_layers),
                "nb_out_filter": int(self.nb_out_filter),
                "context_dilation_schedule": [
                    int(v) for v in list(self.context_dilation_schedule)
                ],
                "head_filters": int(self.head_filters),
                "head_layers": int(self.head_layers),
                "anchor_deformation_at_bed": bool(self.anchor_deformation_at_bed),
                "zero_mean_residual_over_depth": bool(
                    self.zero_mean_residual_over_depth
                ),
            },
            "dx_const": None
            if self.dx_const_value is None
            else float(self.dx_const_value),
        }

    def build(self, input_shape) -> None:
        if self.built:
            return

        input_shape = tf.TensorShape(input_shape)
        if input_shape.rank != 4:
            raise ValueError(
                f"SIADecompNetV2SharedHead expects input_shape rank 4 [B, H, W, C], "
                f"got {input_shape}"
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
        _ = self.call(dummy, training=False, return_components=False)
        super().build(input_shape)

    def call(
        self,
        inputs: tf.Tensor,
        training: bool = False,
        return_components: bool = False,
    ) -> tf.Tensor | Dict[str, Any]:
        raw_inputs = tf.cast(inputs, tf.float32)

        slide_phys, def_phys, all_phys, aux = self._physics_features(raw_inputs)
        context = self._context_features(raw_inputs, training=training)

        slide_xy = self.slide_head(tf.concat([context, slide_phys], axis=-1))

        def_flat = self.def_head(tf.concat([context, def_phys], axis=-1))
        def_uv = self._split_xy_channels(def_flat)
        if self.anchor_deformation_at_bed:
            def_uv = def_uv - def_uv[..., self.bed_index:self.bed_index + 1, :]

        res_flat = self.res_head(tf.concat([context, all_phys], axis=-1))
        res_uv = self._split_xy_channels(res_flat)
        if self.zero_mean_residual_over_depth:
            res_uv = res_uv - tf.reduce_mean(res_uv, axis=-2, keepdims=True)

        slide_uv = self._broadcast_slide(slide_xy)
        total_uv = slide_uv + def_uv + res_uv
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
