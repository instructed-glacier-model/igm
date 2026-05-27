from __future__ import annotations

from typing import Any, Dict
import tensorflow as tf


# --------------------------------------------------------------------
# SpectralConv2D: 2D Fourier layer
# --------------------------------------------------------------------
class SpectralConv2D(tf.keras.layers.Layer):
    """
    2D Fourier layer.

    x: [B, C_in, H, W] channels-first, real
    -> rFFT
    -> multiply low modes with learned complex weights
    -> irFFT
    -> [B, C_out, H, W]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)

        if self.in_channels <= 0:
            raise ValueError(f"in_channels must be > 0, got {self.in_channels}")
        if self.out_channels <= 0:
            raise ValueError(f"out_channels must be > 0, got {self.out_channels}")
        if self.modes1 <= 0:
            raise ValueError(f"modes1 must be > 0, got {self.modes1}")
        if self.modes2 <= 0:
            raise ValueError(f"modes2 must be > 0, got {self.modes2}")

        self.scale = 1.0 / (self.in_channels * self.out_channels)

        self.w1_real = None
        self.w1_imag = None
        self.w2_real = None
        self.w2_imag = None

    def build(self, input_shape) -> None:
        input_shape = tf.TensorShape(input_shape)

        if input_shape.rank != 4:
            raise ValueError(
                f"SpectralConv2D expects rank-4 input [B, C, H, W], got {input_shape}"
            )

        if input_shape[1] is not None:
            got_channels = int(input_shape[1])
            if got_channels != self.in_channels:
                raise ValueError(
                    f"SpectralConv2D expected {self.in_channels} input channels, "
                    f"got {got_channels}"
                )

        if input_shape[2] is not None:
            height = int(input_shape[2])
            if self.modes1 > height:
                raise ValueError(
                    f"SpectralConv2D modes1={self.modes1} exceeds input height "
                    f"H={height}. modes1 must be <= H."
                )

        if input_shape[3] is not None:
            width = int(input_shape[3])
            width_rfft = width // 2 + 1
            if self.modes2 > width_rfft:
                raise ValueError(
                    f"SpectralConv2D modes2={self.modes2} exceeds rFFT width "
                    f"W//2+1={width_rfft} for W={width}. "
                    f"modes2 must be <= W//2 + 1."
                )

        limit = tf.math.sqrt(tf.cast(self.scale, self.compute_dtype))
        init = tf.keras.initializers.RandomUniform(minval=-limit, maxval=limit)

        weight_shape = (
            self.in_channels,
            self.out_channels,
            self.modes1,
            self.modes2,
        )

        self.w1_real = self.add_weight(
            name="w1_real",
            shape=weight_shape,
            initializer=init,
            trainable=True,
            dtype=self.compute_dtype,
        )
        self.w1_imag = self.add_weight(
            name="w1_imag",
            shape=weight_shape,
            initializer=init,
            trainable=True,
            dtype=self.compute_dtype,
        )
        self.w2_real = self.add_weight(
            name="w2_real",
            shape=weight_shape,
            initializer=init,
            trainable=True,
            dtype=self.compute_dtype,
        )
        self.w2_imag = self.add_weight(
            name="w2_imag",
            shape=weight_shape,
            initializer=init,
            trainable=True,
            dtype=self.compute_dtype,
        )

        super().build(input_shape)

    def _compl_mul2d(
        self,
        x_ft: tf.Tensor,
        w_real: tf.Tensor,
        w_imag: tf.Tensor,
    ) -> tf.Tensor:
        """
        Complex multiplication.

        x_ft: [B, C_in, m1, m2]
        w_*:  [C_in, C_out, m1, m2]
        ->    [B, C_out, m1, m2]
        """
        weights = tf.complex(w_real, w_imag)
        return tf.einsum("bixy,ioxy->boxy", x_ft, weights)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        x: [B, C_in, H, W], real
        """
        x = tf.cast(x, self.compute_dtype)

        height = tf.shape(x)[2]
        width = tf.shape(x)[3]

        x_ft = tf.signal.rfft2d(x)  # [B, C_in, H, W//2+1]
        h_ft = tf.shape(x_ft)[2]
        w_r = tf.shape(x_ft)[3]

        x_ft_top = x_ft[:, :, : self.modes1, : self.modes2]
        out_ft_top = self._compl_mul2d(x_ft_top, self.w1_real, self.w1_imag)

        x_ft_bottom = x_ft[:, :, -self.modes1 :, : self.modes2]
        out_ft_bottom = self._compl_mul2d(x_ft_bottom, self.w2_real, self.w2_imag)

        out_ft_top_full = tf.pad(
            out_ft_top,
            paddings=[
                [0, 0],
                [0, 0],
                [0, h_ft - self.modes1],
                [0, w_r - self.modes2],
            ],
        )
        out_ft_bottom_full = tf.pad(
            out_ft_bottom,
            paddings=[
                [0, 0],
                [0, 0],
                [h_ft - self.modes1, 0],
                [0, w_r - self.modes2],
            ],
        )

        out_ft = out_ft_top_full + out_ft_bottom_full

        return tf.signal.irfft2d(out_ft, fft_length=[height, width])

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "in_channels": int(self.in_channels),
                "out_channels": int(self.out_channels),
                "modes1": int(self.modes1),
                "modes2": int(self.modes2),
            }
        )
        return config


# --------------------------------------------------------------------
# FNO: 2D Fourier Neural Operator (renamed from FNO2)
# --------------------------------------------------------------------
class FNO(tf.keras.Model):
    """
    2D Fourier Neural Operator emulator.

    Constructor:
        FNO(input_names=[...], Nz=..., network_params={...})

    network_params is a flat dict of architecture hyperparameters. All keys
    are optional; missing keys fall back to ``_DEFAULTS`` below.

    Output convention:
        output[..., :Nz] = U_x(z)
        output[..., Nz:] = U_y(z)
    """

    _DEFAULTS: Dict[str, tuple] = {
        "width":            (32,   int),
        "modes1":           (8,    int),
        "modes2":           (8,    int),
        "padding":          (9,    int),
        "use_grid":         (True, bool),
        "projection_width": (128,  int),
    }

    def __init__(
        self,
        *,
        input_names: list[str],
        Nz: int,
        network_params: Dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.input_names = [str(x) for x in input_names]
        self.Nz = int(Nz)
        if self.Nz <= 0:
            raise ValueError(f"Nz must be > 0, got {self.Nz}")

        self.nb_inputs = len(self.input_names)
        self.nb_outputs = 2 * self.Nz
        self.input_normalizer = None

        params = dict(network_params) if network_params else {}
        unexpected = sorted(set(params) - set(self._DEFAULTS))
        if unexpected:
            raise ValueError(
                f"Unexpected keys in network_params: {unexpected}. "
                f"Allowed keys: {sorted(self._DEFAULTS)}"
            )
        for k, (default, cast) in self._DEFAULTS.items():
            setattr(self, k, cast(params.get(k, default)))

        if self.width <= 0:
            raise ValueError(f"width must be > 0, got {self.width}")
        if self.modes1 <= 0:
            raise ValueError(f"modes1 must be > 0, got {self.modes1}")
        if self.modes2 <= 0:
            raise ValueError(f"modes2 must be > 0, got {self.modes2}")
        if self.padding < 0:
            raise ValueError(f"padding must be >= 0, got {self.padding}")
        if self.projection_width <= 0:
            raise ValueError(f"projection_width must be > 0, got {self.projection_width}")

        self.lift_input_channels = self.nb_inputs + (2 if self.use_grid else 0)
        self._dummy_H = max(16, self.modes1 + 1)
        self._dummy_W = max(16, 2 * self.modes2 + 2)

        self.fc0 = tf.keras.layers.Dense(self.width, dtype=self.compute_dtype, name="fc0")
        self.convs = [
            SpectralConv2D(self.width, self.width, self.modes1, self.modes2, name=f"spectral_conv_{i}")
            for i in range(4)
        ]
        self.ws = [
            tf.keras.layers.Conv2D(
                self.width,
                kernel_size=1,
                data_format="channels_first",
                use_bias=True,
                dtype=self.compute_dtype,
                name=f"pointwise_skip_{i}",
            )
            for i in range(4)
        ]
        self.fc1 = tf.keras.layers.Dense(self.projection_width, dtype=self.compute_dtype, name="fc1")
        self.fc2 = tf.keras.layers.Dense(self.nb_outputs, dtype=self.compute_dtype, name="fc2")

    # ----------------------------------------------------------------------
    # Minimal reconstruction manifest payload
    # ----------------------------------------------------------------------
    def resolved_params(self) -> Dict[str, Any]:
        return {
            "input_names": [str(n) for n in self.input_names],
            "Nz": int(self.Nz),
            "network_params": {k: getattr(self, k) for k in self._DEFAULTS},
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
                f"FNO expects input_shape rank 4 [B, H, W, C], got {input_shape}"
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
        height_dim = self._dummy_H if input_shape[1] is None else max(self._dummy_H, int(input_shape[1]))
        width_dim = self._dummy_W if input_shape[2] is None else max(self._dummy_W, int(input_shape[2]))

        dummy = tf.zeros(
            shape=(batch_dim, height_dim, width_dim, channel_dim),
            dtype=self.compute_dtype,
        )
        _ = self.call(dummy, training=False)
        super().build(input_shape)

    # ----------------------------------------------------------------------
    # Utilities
    # ----------------------------------------------------------------------
    def _get_grid(self, x: tf.Tensor) -> tf.Tensor:
        shape = tf.shape(x)
        batch_size = shape[0]
        size_x = shape[1]
        size_y = shape[2]

        gridx = tf.cast(tf.linspace(0.0, 1.0, size_x), x.dtype)
        gridx = tf.reshape(gridx, [1, size_x, 1, 1])
        gridx = tf.tile(gridx, [batch_size, 1, size_y, 1])

        gridy = tf.cast(tf.linspace(0.0, 1.0, size_y), x.dtype)
        gridy = tf.reshape(gridy, [1, 1, size_y, 1])
        gridy = tf.tile(gridy, [batch_size, size_x, 1, 1])

        return tf.concat([gridx, gridy], axis=-1)

    # ----------------------------------------------------------------------
    # Forward pass
    # ----------------------------------------------------------------------
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = tf.cast(inputs, self.compute_dtype)

        if self.input_normalizer is not None:
            x = self.input_normalizer(x, training=training)
            x = tf.cast(x, self.compute_dtype)

        if self.use_grid:
            grid = self._get_grid(x)
            x = tf.concat([x, grid], axis=-1)

        x = self.fc0(x)
        x = tf.transpose(x, [0, 3, 1, 2])

        if self.padding > 0:
            x = tf.pad(
                x,
                paddings=[[0, 0], [0, 0], [0, self.padding], [0, self.padding]],
            )

        height_pad = tf.shape(x)[2]
        width_pad = tf.shape(x)[3]

        for i, (spectral_conv, pointwise_conv) in enumerate(zip(self.convs, self.ws)):
            x1 = spectral_conv(x)
            x2 = pointwise_conv(x)
            if i < len(self.convs) - 1:
                x = tf.nn.gelu(x1 + x2)
            else:
                x = x1 + x2

        if self.padding > 0:
            x = x[:, :, : height_pad - self.padding, : width_pad - self.padding]

        x = tf.transpose(x, [0, 2, 3, 1])
        x = self.fc1(x)
        x = tf.nn.gelu(x)
        x = self.fc2(x)

        return x

    # ----------------------------------------------------------------------
    # Keras serialization compatibility
    # ----------------------------------------------------------------------
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(self.resolved_params())
        return config
