from __future__ import annotations

from typing import Optional, Dict, Any, Tuple, List

import tensorflow as tf



def _as_hw_tuple(size: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    if isinstance(size, int):
        return (int(size), int(size))
    if isinstance(size, (tuple, list)) and len(size) == 2:
        return (int(size[0]), int(size[1]))
    raise ValueError(f"Expected int or (H, W) pair for size, got {size!r}")


class CNO_LReLU(tf.keras.layers.Layer):
    """
    TensorFlow equivalent of the simplified official CNO activation:
        bicubic upsample -> LeakyReLU -> bicubic resize to target size.
    """

    def __init__(
        self,
        *,
        in_size: tuple[int, int],
        out_size: tuple[int, int],
        negative_slope: float = 0.01,
        name: Optional[str] = None,
    ):
        super().__init__(name=name, dtype=tf.float32)
        self.in_size = _as_hw_tuple(in_size)
        self.out_size = _as_hw_tuple(out_size)
        self.up_size = (2 * self.in_size[0], 2 * self.in_size[1])
        self.negative_slope = float(negative_slope)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = tf.image.resize(
            x,
            size=self.up_size,
            method="bilinear",
            antialias=False,
        )
        x = tf.nn.leaky_relu(x, alpha=self.negative_slope)
        x = tf.image.resize(
            x,
            size=self.out_size,
            method="bilinear",
            antialias=False,
        )
        return x


class CNOBlock(tf.keras.layers.Layer):
    """Conv -> BN(optional) -> CNO_LReLU."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        in_size: tuple[int, int],
        out_size: tuple[int, int],
        use_bn: bool = True,
        name: Optional[str] = None,
    ):
        super().__init__(name=name, dtype=tf.float32)
        self.convolution = tf.keras.layers.Conv2D(
            filters=int(out_channels),
            kernel_size=3,
            padding="same",
            dtype=tf.float32,
            name=None if name is None else f"{name}_conv",
        )
        self.batch_norm = (
            tf.keras.layers.BatchNormalization(
                axis=-1,
                momentum=0.99,
                epsilon=1e-5,
                dtype=tf.float32,
                name=None if name is None else f"{name}_bn",
            )
            if use_bn
            else None
        )
        self.act = CNO_LReLU(
            in_size=in_size,
            out_size=out_size,
            name=None if name is None else f"{name}_act",
        )

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.convolution(x)
        if self.batch_norm is not None:
            x = self.batch_norm(x, training=training)
        return self.act(x)


class LiftProjectBlock(tf.keras.layers.Layer):
    """Official simplified CNO lift/project block."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        size: tuple[int, int],
        latent_dim: int = 64,
        name: Optional[str] = None,
    ):
        super().__init__(name=name, dtype=tf.float32)
        self.inter_block = CNOBlock(
            in_channels=in_channels,
            out_channels=int(latent_dim),
            in_size=size,
            out_size=size,
            use_bn=False,
            name=None if name is None else f"{name}_inter",
        )
        self.convolution = tf.keras.layers.Conv2D(
            filters=int(out_channels),
            kernel_size=3,
            padding="same",
            dtype=tf.float32,
            name=None if name is None else f"{name}_conv",
        )

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.inter_block(x, training=training)
        x = self.convolution(x)
        return x


class ResidualBlock(tf.keras.layers.Layer):
    """Official simplified CNO residual block."""

    def __init__(
        self,
        *,
        channels: int,
        size: tuple[int, int],
        use_bn: bool = True,
        name: Optional[str] = None,
    ):
        super().__init__(name=name, dtype=tf.float32)
        self.convolution1 = tf.keras.layers.Conv2D(
            filters=int(channels),
            kernel_size=3,
            padding="same",
            dtype=tf.float32,
            name=None if name is None else f"{name}_conv1",
        )
        self.convolution2 = tf.keras.layers.Conv2D(
            filters=int(channels),
            kernel_size=3,
            padding="same",
            dtype=tf.float32,
            name=None if name is None else f"{name}_conv2",
        )
        self.batch_norm1 = (
            tf.keras.layers.BatchNormalization(
                axis=-1,
                momentum=0.99,
                epsilon=1e-5,
                dtype=tf.float32,
                name=None if name is None else f"{name}_bn1",
            )
            if use_bn
            else None
        )
        self.batch_norm2 = (
            tf.keras.layers.BatchNormalization(
                axis=-1,
                momentum=0.99,
                epsilon=1e-5,
                dtype=tf.float32,
                name=None if name is None else f"{name}_bn2",
            )
            if use_bn
            else None
        )
        self.act = CNO_LReLU(
            in_size=size,
            out_size=size,
            name=None if name is None else f"{name}_act",
        )

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        out = self.convolution1(x)
        if self.batch_norm1 is not None:
            out = self.batch_norm1(out, training=training)
        out = self.act(out)
        out = self.convolution2(out)
        if self.batch_norm2 is not None:
            out = self.batch_norm2(out, training=training)
        return x + out


class ResNetStage(tf.keras.layers.Layer):
    def __init__(
        self,
        *,
        channels: int,
        size: tuple[int, int],
        num_blocks: int,
        use_bn: bool = True,
        name: Optional[str] = None,
    ):
        super().__init__(name=name, dtype=tf.float32)
        self.blocks = [
            ResidualBlock(
                channels=channels,
                size=size,
                use_bn=use_bn,
                name=None if name is None else f"{name}_block_{i}",
            )
            for i in range(int(num_blocks))
        ]

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        for block in self.blocks:
            x = block(x, training=training)
        return x


class CNOContextEncoder(tf.keras.layers.Layer):
    """
    CNO-style encoder-decoder context branch adapted from the official simplified
    CNO2d implementation, but used here only as the learned context pathway.
    """

    def __init__(
        self,
        *,
        in_dim: int,
        out_dim: int,
        input_size: tuple[int, int],
        n_layers: int,
        n_res: int = 4,
        n_res_neck: int = 4,
        channel_multiplier: int = 16,
        use_bn: bool = True,
        lift_project_latent_dim: int = 64,
        name: str = "cno_context",
    ):
        super().__init__(name=name, dtype=tf.float32)

        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.input_size = _as_hw_tuple(input_size)
        self.n_layers = int(n_layers)
        self.n_res = int(n_res)
        self.n_res_neck = int(n_res_neck)
        self.channel_multiplier = int(channel_multiplier)
        self.use_bn = bool(use_bn)
        self.lift_project_latent_dim = int(lift_project_latent_dim)

        if self.n_layers < 1:
            raise ValueError("CNOContextEncoder requires n_layers >= 1")
        if min(self.input_size) // (2 ** self.n_layers) < 1:
            raise ValueError(
                f"CNO n_layers={self.n_layers} is too deep for input_size={self.input_size}; "
                "the bottleneck would collapse below 1 pixel."
            )

        self.lift_dim = max(1, self.channel_multiplier // 2)

        self.encoder_features = [self.lift_dim]
        for i in range(self.n_layers):
            self.encoder_features.append((2 ** i) * self.channel_multiplier)

        self.decoder_features_in = list(self.encoder_features[1:])
        self.decoder_features_in.reverse()

        self.decoder_features_out = list(self.encoder_features[:-1])
        self.decoder_features_out.reverse()

        for i in range(1, self.n_layers):
            self.decoder_features_in[i] = 2 * self.decoder_features_in[i]

        self.encoder_sizes: list[tuple[int, int]] = []
        self.decoder_sizes: list[tuple[int, int]] = []
        h0, w0 = self.input_size
        for i in range(self.n_layers + 1):
            self.encoder_sizes.append((h0 // (2 ** i), w0 // (2 ** i)))
            self.decoder_sizes.append((h0 // (2 ** (self.n_layers - i)), w0 // (2 ** (self.n_layers - i))))

        self.lift = LiftProjectBlock(
            in_channels=self.in_dim,
            out_channels=self.encoder_features[0],
            size=self.input_size,
            latent_dim=self.lift_project_latent_dim,
            name="lift",
        )
        self.project = LiftProjectBlock(
            in_channels=self.encoder_features[0] + self.decoder_features_out[-1],
            out_channels=self.out_dim,
            size=self.input_size,
            latent_dim=self.lift_project_latent_dim,
            name="project",
        )

        self.encoder = [
            CNOBlock(
                in_channels=self.encoder_features[i],
                out_channels=self.encoder_features[i + 1],
                in_size=self.encoder_sizes[i],
                out_size=self.encoder_sizes[i + 1],
                use_bn=self.use_bn,
                name=f"encoder_{i}",
            )
            for i in range(self.n_layers)
        ]

        self.ed_expansion = [
            CNOBlock(
                in_channels=self.encoder_features[i],
                out_channels=self.encoder_features[i],
                in_size=self.encoder_sizes[i],
                out_size=self.decoder_sizes[self.n_layers - i],
                use_bn=self.use_bn,
                name=f"ed_expansion_{i}",
            )
            for i in range(self.n_layers + 1)
        ]

        self.decoder = [
            CNOBlock(
                in_channels=self.decoder_features_in[i],
                out_channels=self.decoder_features_out[i],
                in_size=self.decoder_sizes[i],
                out_size=self.decoder_sizes[i + 1],
                use_bn=self.use_bn,
                name=f"decoder_{i}",
            )
            for i in range(self.n_layers)
        ]

        self.res_nets = [
            ResNetStage(
                channels=self.encoder_features[l],
                size=self.encoder_sizes[l],
                num_blocks=self.n_res,
                use_bn=self.use_bn,
                name=f"resnet_{l}",
            )
            for l in range(self.n_layers)
        ]
        self.res_net_neck = ResNetStage(
            channels=self.encoder_features[self.n_layers],
            size=self.encoder_sizes[self.n_layers],
            num_blocks=self.n_res_neck,
            use_bn=self.use_bn,
            name="resnet_neck",
        )

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.lift(x, training=training)

        skip: list[tf.Tensor] = []
        for i in range(self.n_layers):
            y = self.res_nets[i](x, training=training)
            skip.append(y)
            x = self.encoder[i](x, training=training)

        x = self.res_net_neck(x, training=training)

        for i in range(self.n_layers):
            if i == 0:
                x = self.ed_expansion[self.n_layers - i](x, training=training)
            else:
                x = tf.concat(
                    [x, self.ed_expansion[self.n_layers - i](skip[-i], training=training)],
                    axis=-1,
                )
            x = self.decoder[i](x, training=training)

        x = tf.concat([x, self.ed_expansion[0](skip[0], training=training)], axis=-1)
        x = self.project(x, training=training)
        return x


class CNO_DecompNet(tf.keras.Model):
    """
    Physics-guided ice-flow emulator with additive decomposition:

        total velocity = sliding head + deformation head + residual head

    The explicit physics feature pathway is kept intact. The learned context
    pathway is replaced by a CNO-style encoder-decoder adapted from the
    official simplified CNO2d implementation.
    """

    FIXED_N_GLEN = 3.0
    FIXED_RHO = 917.0
    FIXED_G = 9.81

    FIXED_M_SLIDE = 3.0
    FIXED_U_REF = 100.0

    FIXED_EPS = 1e-8
    FIXED_H_REF = 200.0
    FIXED_SLOPE_REF = 0.1
    FIXED_A_REF = 7.6e-24
    FIXED_DX_REF = 100.0

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

        self.input_names = list(input_names)
        self.Nz = int(Nz)
        self.nb_inputs = len(self.input_names)
        self.nb_outputs = 2 * self.Nz

        self.n_glen = float(self.FIXED_N_GLEN)
        self.rho = float(self.FIXED_RHO)
        self.g = float(self.FIXED_G)

        self.m_slide = float(self.FIXED_M_SLIDE)
        self.u_ref = float(self.FIXED_U_REF)

        self.eps_value = float(self.FIXED_EPS)
        self.H_ref_value = float(self.FIXED_H_REF)
        self.slope_ref_value = float(self.FIXED_SLOPE_REF)
        self.A_ref_value = float(self.FIXED_A_REF)
        self.dx_ref_value = float(self.FIXED_DX_REF)
        self.tau_ref_scale_value = (
            self.rho * self.g * self.H_ref_value * self.slope_ref_value
        )

        self.eps = tf.constant(self.eps_value, dtype=tf.float32)
        self.H_ref = tf.constant(self.H_ref_value, dtype=tf.float32)
        self.slope_ref = tf.constant(self.slope_ref_value, dtype=tf.float32)
        self.tau_ref_scale = tf.constant(self.tau_ref_scale_value, dtype=tf.float32)
        self.A_ref = tf.constant(self.A_ref_value, dtype=tf.float32)
        self.dx_ref = tf.constant(self.dx_ref_value, dtype=tf.float32)

        self.idx_thk = self.input_names.index("thk")
        self.idx_usurf = self.input_names.index("usurf")
        self.idx_slidingco = self.input_names.index("slidingco") if "slidingco" in self.input_names else None
        self.idx_arrhenius = self.input_names.index("arrhenius") if "arrhenius" in self.input_names else None
        self.idx_dX = self.input_names.index("dX") if "dX" in self.input_names else None

        if self.idx_dX is None:
            self.dx_const_value = 90.0 if dx_const is None else float(dx_const)
            self.dx_const = tf.constant(self.dx_const_value, dtype=tf.float32)
        else:
            self.dx_const_value = None
            self.dx_const = None

        self.input_normalizer = None

        params = dict(network_params)
        allowed_keys = {
            "nb_out_filter",
            "cno_n_layers",
            "cno_channel_multiplier",
            "cno_n_res",
            "cno_n_res_neck",
            "cno_use_bn",
            "cno_lift_project_latent_dim",
            "context_include_dx",
            "context_include_coords",
        }
        unexpected = sorted(set(params.keys()) - allowed_keys)
        if unexpected:
            raise ValueError(
                f"Unexpected keys in network_params: {unexpected}. "
                f"Allowed keys are: {sorted(allowed_keys)}"
            )

        if "nb_out_filter" not in params:
            raise ValueError("network_params must contain 'nb_out_filter'")
        if "cno_n_layers" not in params:
            raise ValueError("network_params must contain 'cno_n_layers'")

        self.nb_out_filter = int(params["nb_out_filter"])
        self.cno_n_layers = int(params["cno_n_layers"])
        self.cno_channel_multiplier = int(params.get("cno_channel_multiplier", self.nb_out_filter))
        self.cno_n_res = int(params.get("cno_n_res", 4))
        self.cno_n_res_neck = int(params.get("cno_n_res_neck", 4))
        self.cno_use_bn = bool(params.get("cno_use_bn", True))
        self.cno_lift_project_latent_dim = int(params.get("cno_lift_project_latent_dim", 64))
        self.context_include_dx = bool(params.get("context_include_dx", True))
        self.context_include_coords = bool(params.get("context_include_coords", False))

        self.network_params = {
            "nb_out_filter": int(self.nb_out_filter),
            "cno_n_layers": int(self.cno_n_layers),
            "cno_channel_multiplier": int(self.cno_channel_multiplier),
            "cno_n_res": int(self.cno_n_res),
            "cno_n_res_neck": int(self.cno_n_res_neck),
            "cno_use_bn": bool(self.cno_use_bn),
            "cno_lift_project_latent_dim": int(self.cno_lift_project_latent_dim),
            "context_include_dx": bool(self.context_include_dx),
            "context_include_coords": bool(self.context_include_coords),
        }

        self.log_H_ref = tf.math.log(self.H_ref + 1.0)
        self.log_tau_ref_scale = tf.math.log(self.tau_ref_scale + self.eps)
        self.log_A_ref = tf.math.log(self.A_ref + self.eps)
        self.log_dx_ref = tf.math.log(self.dx_ref + self.eps)

        self.B_ref = 2.0 * tf.pow(self.A_ref, -1.0 / self.n_glen)
        self.log_B_ref = tf.math.log(self.B_ref + self.eps)

        self.log_u_slide_ref = tf.math.log(
            tf.constant(self.u_ref, dtype=tf.float32) + self.eps
        )
        self.log_u_def_ref = (
            self.log_A_ref
            + (self.n_glen + 1.0) * tf.math.log(self.H_ref + self.eps)
            + self.n_glen * tf.math.log(self.slope_ref + self.eps)
        )

        self.context_encoder: Optional[CNOContextEncoder] = None
        self.context_extra_channels = 0
        if self.context_include_dx:
            self.context_extra_channels += 1
        if self.context_include_coords:
            self.context_extra_channels += 2

        self.slide_head_conv1 = tf.keras.layers.Conv2D(
            self.nb_out_filter, 3, padding="same", dtype=tf.float32, name="slide_head_conv1"
        )
        self.slide_head_act1 = tf.keras.layers.Activation(tf.nn.gelu, name="slide_head_gelu1")
        self.slide_head_conv2 = tf.keras.layers.Conv2D(
            self.nb_out_filter, 3, padding="same", dtype=tf.float32, name="slide_head_conv2"
        )
        self.slide_head_act2 = tf.keras.layers.Activation(tf.nn.gelu, name="slide_head_gelu2")
        self.slide_head_out = tf.keras.layers.Conv2D(2, 1, padding="same", dtype=tf.float32, name="slide_head_out")

        self.def_head_conv1 = tf.keras.layers.Conv2D(
            self.nb_out_filter, 3, padding="same", dtype=tf.float32, name="def_head_conv1"
        )
        self.def_head_act1 = tf.keras.layers.Activation(tf.nn.gelu, name="def_head_gelu1")
        self.def_head_conv2 = tf.keras.layers.Conv2D(
            self.nb_out_filter, 3, padding="same", dtype=tf.float32, name="def_head_conv2"
        )
        self.def_head_act2 = tf.keras.layers.Activation(tf.nn.gelu, name="def_head_gelu2")
        self.def_head_out = tf.keras.layers.Conv2D(
            2 * self.Nz, 1, padding="same", dtype=tf.float32, name="def_head_out"
        )

        self.res_head_filters = max(self.nb_out_filter // 2, 8)
        self.res_head_conv1 = tf.keras.layers.Conv2D(
            self.res_head_filters, 3, padding="same", dtype=tf.float32, name="res_head_conv1"
        )
        self.res_head_act1 = tf.keras.layers.Activation(tf.nn.gelu, name="res_head_gelu1")
        self.res_head_conv2 = tf.keras.layers.Conv2D(
            self.res_head_filters, 3, padding="same", dtype=tf.float32, name="res_head_conv2"
        )
        self.res_head_act2 = tf.keras.layers.Activation(tf.nn.gelu, name="res_head_gelu2")
        self.res_head_out = tf.keras.layers.Conv2D(
            2 * self.Nz,
            1,
            padding="same",
            dtype=tf.float32,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="res_head_out",
        )

    def resolved_params(self) -> Dict[str, Any]:
        return {
            "input_names": list(self.input_names),
            "Nz": int(self.Nz),
            "network_params": dict(self.network_params),
            "dx_const": None if self.dx_const_value is None else float(self.dx_const_value),
        }

    def _init_context_encoder(self, input_hw: tuple[int, int]) -> None:
        context_in_dim = self.nb_inputs + self.context_extra_channels
        self.context_encoder = CNOContextEncoder(
            in_dim=context_in_dim,
            out_dim=self.nb_out_filter,
            input_size=input_hw,
            n_layers=self.cno_n_layers,
            n_res=self.cno_n_res,
            n_res_neck=self.cno_n_res_neck,
            channel_multiplier=self.cno_channel_multiplier,
            use_bn=self.cno_use_bn,
            lift_project_latent_dim=self.cno_lift_project_latent_dim,
            name="context_cno",
        )

    def build(self, input_shape) -> None:
        if self.built:
            return

        input_shape = tf.TensorShape(input_shape)
        if input_shape.rank != 4:
            raise ValueError(
                f"CNO_DecompNet expects input_shape rank 4 [B, H, W, C], got {input_shape}"
            )

        channel_dim = input_shape[-1]
        channel_dim = self.nb_inputs if channel_dim is None else int(channel_dim)
        if channel_dim != self.nb_inputs:
            raise ValueError(
                f"Input channel mismatch: model expects {self.nb_inputs} channels "
                f"from input_names={self.input_names}, but build got C={channel_dim}."
            )

        batch_dim = 1 if input_shape[0] is None else int(input_shape[0])
        height_dim = 4 if input_shape[1] is None else int(input_shape[1])
        width_dim = 4 if input_shape[2] is None else int(input_shape[2])

        self._init_context_encoder((height_dim, width_dim))

        dummy = tf.zeros((batch_dim, height_dim, width_dim, channel_dim), dtype=tf.float32)
        _ = self.call(dummy, training=False, return_components=False)
        super().build(input_shape)

    def set_input_normalizer(self, layer: tf.keras.layers.Layer) -> None:
        self.input_normalizer = layer

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
        multiples = tf.stack([
            tf.constant(1, dtype=tf.int32),
            tf.constant(1, dtype=tf.int32),
            tf.constant(1, dtype=tf.int32),
            tf.constant(self.Nz, dtype=tf.int32),
            tf.constant(1, dtype=tf.int32),
        ])
        return tf.tile(slide_xy, multiples)

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
        return (fpad[:, :, 2:, :] - 2.0 * field + fpad[:, :, :-2, :]) / (dx * dx + self.eps)

    def _second_diff_y(self, field: tf.Tensor, dx: tf.Tensor) -> tf.Tensor:
        fpad = tf.pad(field, [[0, 0], [1, 1], [0, 0], [0, 0]], mode="SYMMETRIC")
        return (fpad[:, 2:, :, :] - 2.0 * field + fpad[:, :-2, :, :]) / (dx * dx + self.eps)

    def _physics_features(self, raw_inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, Dict[str, tf.Tensor]]:
        x = tf.cast(raw_inputs, tf.float32)

        # ------------------------------------------------------------------
        # Raw fields
        # ------------------------------------------------------------------
        H = tf.maximum(x[..., self.idx_thk : self.idx_thk + 1], 0.0)
        s = x[..., self.idx_usurf : self.idx_usurf + 1]
        b = s - H
        dx = self._get_dx_field(x)

        tau_ref = None
        if self.idx_slidingco is not None:
            tau_ref = tf.maximum(x[..., self.idx_slidingco : self.idx_slidingco + 1], self.eps)

        A = None
        if self.idx_arrhenius is not None:
            A = tf.maximum(x[..., self.idx_arrhenius : self.idx_arrhenius + 1], self.eps)

        # ------------------------------------------------------------------
        # Conservative stabilizers for thickness-derivative paths
        #
        # Kept local here so you only need to edit this one method for testing.
        # For a cleaner long-term version, these could become class constants.
        # ------------------------------------------------------------------
        H_proxy_floor = tf.constant(10.0, dtype=tf.float32)

        # 3x3 binomial blur:
        #   [1 2 1]
        #   [2 4 2] / 16
        #   [1 2 1]
        #
        # Use REFLECT padding to avoid zero-padding edge artifacts.
        binomial_kernel = tf.constant(
            [[1.0, 2.0, 1.0],
            [2.0, 4.0, 2.0],
            [1.0, 2.0, 1.0]],
            dtype=tf.float32,
        ) / 16.0
        binomial_kernel = binomial_kernel[:, :, tf.newaxis, tf.newaxis]

        H_for_bed_grad = tf.nn.depthwise_conv2d(
            tf.pad(H, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT"),
            filter=binomial_kernel,
            strides=[1, 1, 1, 1],
            padding="VALID",
        )
        b_for_bed_grad = s - H_for_bed_grad

        # Use a floored thickness only inside the log-based proxies.
        # This leaves the linear-in-H physics terms untouched.
        H_for_log_proxy = H + H_proxy_floor

        # ------------------------------------------------------------------
        # Geometry
        # ------------------------------------------------------------------
        dsdx = self._central_diff_x(s, dx)
        dsdy = self._central_diff_y(s, dx)

        # Bed gradients now use blurred thickness rather than raw thickness.
        dbdx = self._central_diff_x(b_for_bed_grad, dx)
        dbdy = self._central_diff_y(b_for_bed_grad, dx)

        # Keep curvature channels unchanged: they do not depend on H anyway.
        d2sdx2 = self._second_diff_x(s, dx)
        d2sdy2 = self._second_diff_y(s, dx)

        grad_s = tf.sqrt(dsdx**2 + dsdy**2 + self.eps)
        grad_b = tf.sqrt(dbdx**2 + dbdy**2 + self.eps)

        # ------------------------------------------------------------------
        # Driving stress proxy
        # ------------------------------------------------------------------
        tau_dx = -self.rho * self.g * H * dsdx
        tau_dy = -self.rho * self.g * H * dsdy
        tau_d = tf.sqrt(tau_dx**2 + tau_dy**2 + self.eps)

        # Floored version used only for the log-stress proxies.
        tau_dx_for_log = -self.rho * self.g * H_for_log_proxy * dsdx
        tau_dy_for_log = -self.rho * self.g * H_for_log_proxy * dsdy
        tau_d_for_log = tf.sqrt(tau_dx_for_log**2 + tau_dy_for_log**2 + self.eps)

        # ------------------------------------------------------------------
        # Fixed physical scaling / centering to O(1)
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

        # IMPORTANT: log stress uses the floored proxy, not the raw tau_d.
        log_tau_d_raw = tf.math.log(tau_d_for_log + self.eps)
        log_tau_d = (log_tau_d_raw - self.log_tau_ref_scale) / 5.0

        ice_mask = tf.cast(H > 1.0, tf.float32)
        dir_x = -dsdx / (grad_s + self.eps) * ice_mask
        dir_y = -dsdy / (grad_s + self.eps) * ice_mask

        # Surface-only curvature channels: not a dJ/dthk culprit.
        curv_x_n = d2sdx2 * 1000.0
        curv_y_n = d2sdy2 * 1000.0

        # ------------------------------------------------------------------
        # Sliding features
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

            # IMPORTANT: sliding log proxy now uses floored log_tau_d_raw
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

        # ------------------------------------------------------------------
        # Deformation features
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

            # IMPORTANT: deformation log proxy now uses the same thickness floor
            # that was already added in SR.py.
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

        # Keep the all-features tensor shape exactly the same as before.
        all_feats = tf.concat([slide_feats, def_feats, curv_x_n, curv_y_n], axis=-1)

        aux = {
            "H": H,
            "s": s,
            "b": b,
            "dx": dx,
            "H_for_bed_grad": H_for_bed_grad,
            "b_for_bed_grad": b_for_bed_grad,
            "H_for_log_proxy": H_for_log_proxy,
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

    def _coord_channels(self, x: tf.Tensor) -> tf.Tensor:
        shape = tf.shape(x)
        B = shape[0]
        H = shape[1]
        W = shape[2]

        xi = tf.linspace(tf.constant(-1.0, tf.float32), tf.constant(1.0, tf.float32), W)
        yi = tf.linspace(tf.constant(-1.0, tf.float32), tf.constant(1.0, tf.float32), H)
        X, Y = tf.meshgrid(xi, yi)
        X = tf.broadcast_to(X[tf.newaxis, ..., tf.newaxis], [B, H, W, 1])
        Y = tf.broadcast_to(Y[tf.newaxis, ..., tf.newaxis], [B, H, W, 1])
        return tf.concat([X, Y], axis=-1)

    def _context_features(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:
        if self.context_encoder is None:
            raise RuntimeError("Context encoder has not been initialized. Call build() first.")

        raw_x = tf.cast(inputs, tf.float32)
        x = raw_x
        if self.input_normalizer is not None:
            x = self.input_normalizer(x, training=training)

        extras = []
        if self.context_include_dx:
            dx = self._get_dx_field(raw_x)
            log_dx = tf.math.log(dx + self.eps)
            extras.append(log_dx - self.log_dx_ref)
        if self.context_include_coords:
            extras.append(self._coord_channels(raw_x))

        if extras:
            x = tf.concat([x] + extras, axis=-1)

        return self.context_encoder(x, training=training)

    def call(self, inputs: tf.Tensor, training: bool = False, return_components: bool = False) -> tf.Tensor | Dict[str, Any]:
        raw_inputs = tf.cast(inputs, tf.float32)

        slide_phys, def_phys, all_phys, aux = self._physics_features(raw_inputs)
        context = self._context_features(raw_inputs, training=training)

        slide_in = tf.concat([slide_phys, context], axis=-1)
        slide_h = self.slide_head_conv1(slide_in)
        slide_h = self.slide_head_act1(slide_h)
        slide_h = self.slide_head_conv2(slide_h)
        slide_h = self.slide_head_act2(slide_h)
        slide_xy = self.slide_head_out(slide_h)

        def_in = tf.concat([def_phys, context], axis=-1)
        def_h = self.def_head_conv1(def_in)
        def_h = self.def_head_act1(def_h)
        def_h = self.def_head_conv2(def_h)
        def_h = self.def_head_act2(def_h)
        def_flat = self.def_head_out(def_h)
        def_uv = self._split_xy_channels(def_flat)

        res_in = tf.concat([all_phys, context], axis=-1)
        res_h = self.res_head_conv1(res_in)
        res_h = self.res_head_act1(res_h)
        res_h = self.res_head_conv2(res_h)
        res_h = self.res_head_act2(res_h)
        res_flat = self.res_head_out(res_h)
        res_uv = self._split_xy_channels(res_flat)

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
            "context": context,
        }
