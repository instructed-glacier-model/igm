from __future__ import annotations

from typing import Any, Dict
import tensorflow as tf
from igm.utils.math.precision import normalize_precision


class FNO(tf.keras.Model):
    """
    Simplified Fourier Neural Operator - operates in frequency domain for global information
    """

    def __init__(self, cfg, nb_inputs, nb_outputs, input_normalizer=None):
        super(FNO, self).__init__()

        precision = cfg.processes.iceflow.numerics.precision
        self.dtype_model = normalize_precision(precision)

        self.input_normalizer = input_normalizer

        nb_filters = cfg.processes.iceflow.emulator.network.nb_out_filter
        nb_layers = int(cfg.processes.iceflow.emulator.network.nb_layers)

        # Lifting layer: project input to hidden dimension
        self.lift = tf.keras.layers.Conv2D(nb_filters, 1, dtype=self.dtype_model)

        # Fourier layers
        self.fourier_layers = []
        self.conv_layers = []
        for i in range(nb_layers):
            # We'll use Conv2D as a simplified spectral convolution
            self.fourier_layers.append(
                tf.keras.layers.Conv2D(nb_filters, 1, dtype=self.dtype_model)
            )
            self.conv_layers.append(
                tf.keras.layers.Conv2D(nb_filters, 1, dtype=self.dtype_model)
            )

        self.activation = tf.keras.layers.Activation(
            cfg.processes.iceflow.emulator.network.activation
        )

        # Projection layer: project back to output dimension
        self.project = tf.keras.layers.Conv2D(
            nb_outputs, 1, activation=None, dtype=self.dtype_model
        )

        self.build(input_shape=[None, None, None, nb_inputs])

    def call(self, inputs):
        x = inputs

        if self.input_normalizer is not None:
            x = self.input_normalizer(x)

        # Lift to higher dimension
        x = self.lift(x)

        # Fourier layers
        for fourier_layer, conv_layer in zip(self.fourier_layers, self.conv_layers):
            residual = x

            # FFT to frequency domain
            x_freq = tf.signal.fft2d(tf.cast(x, tf.complex64))

            # Apply spectral convolution (simplified: using real part)
            x_freq_real = tf.cast(tf.math.real(x_freq), self.dtype_model)
            x_freq_imag = tf.cast(tf.math.imag(x_freq), self.dtype_model)

            # Process in frequency domain
            x_freq_processed = fourier_layer(x_freq_real)

            # IFFT back to spatial domain
            x_freq_complex = tf.cast(x_freq_processed, tf.complex64)
            x = tf.cast(
                tf.math.real(tf.signal.ifft2d(x_freq_complex)), self.dtype_model
            )

            # Add skip connection in spatial domain
            x = x + conv_layer(residual)
            x = self.activation(x)

        # Project to output
        outputs = self.project(x)

        return outputs

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

        limit = tf.math.sqrt(tf.cast(self.scale, tf.float32))
        init = tf.keras.initializers.RandomUniform(minval=-limit, maxval=limit)

        weight_shape = (
            self.in_channels,
            self.out_channels,
            self.modes1,
            self.modes2,
        )

        # Top-frequency block
        self.w1_real = self.add_weight(
            name="w1_real",
            shape=weight_shape,
            initializer=init,
            trainable=True,
            dtype=tf.float32,
        )
        self.w1_imag = self.add_weight(
            name="w1_imag",
            shape=weight_shape,
            initializer=init,
            trainable=True,
            dtype=tf.float32,
        )

        # Bottom-frequency block
        self.w2_real = self.add_weight(
            name="w2_real",
            shape=weight_shape,
            initializer=init,
            trainable=True,
            dtype=tf.float32,
        )
        self.w2_imag = self.add_weight(
            name="w2_imag",
            shape=weight_shape,
            initializer=init,
            trainable=True,
            dtype=tf.float32,
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
        x = tf.cast(x, tf.float32)

        height = tf.shape(x)[2]
        width = tf.shape(x)[3]

        x_ft = tf.signal.rfft2d(x)  # [B, C_in, H, W//2+1]
        h_ft = tf.shape(x_ft)[2]
        w_r = tf.shape(x_ft)[3]

        # Top modes
        x_ft_top = x_ft[:, :, : self.modes1, : self.modes2]
        out_ft_top = self._compl_mul2d(
            x_ft_top,
            self.w1_real,
            self.w1_imag,
        )

        # Bottom modes
        x_ft_bottom = x_ft[:, :, -self.modes1 :, : self.modes2]
        out_ft_bottom = self._compl_mul2d(
            x_ft_bottom,
            self.w2_real,
            self.w2_imag,
        )

        # Pad top block into full Fourier tensor
        out_ft_top_full = tf.pad(
            out_ft_top,
            paddings=[
                [0, 0],
                [0, 0],
                [0, h_ft - self.modes1],
                [0, w_r - self.modes2],
            ],
        )

        # Pad bottom block into full Fourier tensor
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

        return tf.signal.irfft2d(
            out_ft,
            fft_length=[height, width],
        )

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
# FNO2: new-format architecture
# --------------------------------------------------------------------
class FNO2(tf.keras.Model):
    """
    2D Fourier Neural Operator emulator.

    New-format constructor:

        FNO2(
            input_names=[...],
            Nz=...,
            network_params={...},
        )

    Output convention:
        output[..., :Nz] = U_x(z)
        output[..., Nz:] = U_y(z)

    Therefore nb_outputs is fixed to 2 * Nz.
    """

    def __init__(
        self,
        *,
        input_names: list[str],
        Nz: int,
        network_params: Dict[str, Any],
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        # ------------------------------------------------------------------
        # Minimal reconstruction inputs
        # ------------------------------------------------------------------
        self.input_names = [str(x) for x in input_names]
        self.Nz = int(Nz)

        if self.Nz <= 0:
            raise ValueError(f"Nz must be > 0, got {self.Nz}")

        self.nb_inputs = len(self.input_names)
        self.nb_outputs = 2 * self.Nz

        # External normalizer, attached elsewhere.
        self.input_normalizer = None

        # ------------------------------------------------------------------
        # Architecture parameters from network_params
        # ------------------------------------------------------------------
        params = dict(network_params)

        allowed_keys = {
            "width",
            "modes1",
            "modes2",
            "padding",
            "use_grid",
            "projection_width",
        }

        unexpected = sorted(set(params.keys()) - allowed_keys)
        if unexpected:
            raise ValueError(
                f"Unexpected keys in network_params: {unexpected}. "
                f"Allowed keys are: {sorted(allowed_keys)}"
            )

        self.width = int(params.get("width", 32))
        self.modes1 = int(params.get("modes1", 8))
        self.modes2 = int(params.get("modes2", self.modes1))
        self.padding = int(params.get("padding", 9))
        self.use_grid = bool(params.get("use_grid", True))
        self.projection_width = int(params.get("projection_width", 128))

        if self.width <= 0:
            raise ValueError(f"width must be > 0, got {self.width}")
        if self.modes1 <= 0:
            raise ValueError(f"modes1 must be > 0, got {self.modes1}")
        if self.modes2 <= 0:
            raise ValueError(f"modes2 must be > 0, got {self.modes2}")
        if self.padding < 0:
            raise ValueError(f"padding must be >= 0, got {self.padding}")
        if self.projection_width <= 0:
            raise ValueError(
                f"projection_width must be > 0, got {self.projection_width}"
            )

        self.lift_input_channels = self.nb_inputs + (2 if self.use_grid else 0)

        # Store normalized architecture dict exactly as used.
        self.network_params = {
            "width": int(self.width),
            "modes1": int(self.modes1),
            "modes2": int(self.modes2),
            "padding": int(self.padding),
            "use_grid": bool(self.use_grid),
            "projection_width": int(self.projection_width),
        }

        # Dummy sizes for deterministic build.
        #
        # SpectralConv2D sees the padded size, so ensure the unpadded dummy plus
        # padding is large enough for the requested modes.
        self._dummy_H = max(16, self.modes1 + 1)
        self._dummy_W = max(16, 2 * self.modes2 + 2)

        # ------------------------------------------------------------------
        # Layers
        # ------------------------------------------------------------------
        self.fc0 = tf.keras.layers.Dense(
            self.width,
            dtype=tf.float32,
            name="fc0",
        )

        self.convs = [
            SpectralConv2D(
                self.width,
                self.width,
                self.modes1,
                self.modes2,
                name=f"spectral_conv_{i}",
            )
            for i in range(4)
        ]

        self.ws = [
            tf.keras.layers.Conv2D(
                self.width,
                kernel_size=1,
                data_format="channels_first",
                use_bias=True,
                dtype=tf.float32,
                name=f"pointwise_skip_{i}",
            )
            for i in range(4)
        ]

        self.fc1 = tf.keras.layers.Dense(
            self.projection_width,
            dtype=tf.float32,
            name="fc1",
        )
        self.fc2 = tf.keras.layers.Dense(
            self.nb_outputs,
            dtype=tf.float32,
            name="fc2",
        )

    # ----------------------------------------------------------------------
    # Minimal reconstruction manifest payload
    # ----------------------------------------------------------------------
    def resolved_params(self) -> Dict[str, Any]:
        """
        Return exactly the minimal constructor payload needed to rebuild the
        model structure before attaching weights / external normalizer.
        """
        return {
            "input_names": [str(n) for n in self.input_names],
            "Nz": int(self.Nz),
            "network_params": dict(self.network_params),
        }

    # ----------------------------------------------------------------------
    # Keras build
    # ----------------------------------------------------------------------
    def build(self, input_shape) -> None:
        """
        Explicit deterministic build for subclassed-model compatibility.

        Creates weights in forward-pass order via a dummy call.
        """
        if self.built:
            return

        input_shape = tf.TensorShape(input_shape)
        if input_shape.rank != 4:
            raise ValueError(
                f"FNO2 expects input_shape rank 4 [B, H, W, C], got {input_shape}"
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

        if input_shape[1] is None:
            height_dim = self._dummy_H
        else:
            height_dim = max(self._dummy_H, int(input_shape[1]))

        if input_shape[2] is None:
            width_dim = self._dummy_W
        else:
            width_dim = max(self._dummy_W, int(input_shape[2]))

        dummy = tf.zeros(
            shape=(batch_dim, height_dim, width_dim, channel_dim),
            dtype=tf.float32,
        )

        _ = self.call(dummy, training=False)
        super().build(input_shape)

    # ----------------------------------------------------------------------
    # Public helper methods
    # ----------------------------------------------------------------------
    def set_input_normalizer(self, layer: tf.keras.layers.Layer) -> None:
        """
        Attach an external input normalizer.

        This is not part of the architecture reconstruction contract.
        """
        self.input_normalizer = layer

    # ----------------------------------------------------------------------
    # Utilities
    # ----------------------------------------------------------------------
    def _get_grid(self, x: tf.Tensor) -> tf.Tensor:
        """
        Generate [B, H, W, 2] grid with coordinates in [0, 1].
        """
        shape = tf.shape(x)
        batch_size = shape[0]
        size_x = shape[1]
        size_y = shape[2]

        gridx = tf.linspace(0.0, 1.0, size_x)
        gridx = tf.reshape(gridx, [1, size_x, 1, 1])
        gridx = tf.tile(gridx, [batch_size, 1, size_y, 1])

        gridy = tf.linspace(0.0, 1.0, size_y)
        gridy = tf.reshape(gridy, [1, 1, size_y, 1])
        gridy = tf.tile(gridy, [batch_size, size_x, 1, 1])

        return tf.concat([gridx, gridy], axis=-1)

    # ----------------------------------------------------------------------
    # Forward pass
    # ----------------------------------------------------------------------
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        inputs:  [B, H, W, C_in]
        returns: [B, H, W, 2*Nz]
        """
        x = tf.cast(inputs, tf.float32)

        if self.input_normalizer is not None:
            x = self.input_normalizer(x, training=training)
            x = tf.cast(x, tf.float32)

        if self.use_grid:
            grid = self._get_grid(x)
            x = tf.concat([x, grid], axis=-1)

        # Lift to hidden width
        x = self.fc0(x)  # [B, H, W, width]

        # Channels-last -> channels-first
        x = tf.transpose(x, [0, 3, 1, 2])  # [B, width, H, W]

        if self.padding > 0:
            x = tf.pad(
                x,
                paddings=[
                    [0, 0],
                    [0, 0],
                    [0, self.padding],
                    [0, self.padding],
                ],
            )

        height_pad = tf.shape(x)[2]
        width_pad = tf.shape(x)[3]

        # Four Fourier blocks.
        for i, (spectral_conv, pointwise_conv) in enumerate(zip(self.convs, self.ws)):
            x1 = spectral_conv(x)
            x2 = pointwise_conv(x)

            if i < len(self.convs) - 1:
                x = tf.nn.gelu(x1 + x2)
            else:
                x = x1 + x2

        if self.padding > 0:
            x = x[
                :,
                :,
                : height_pad - self.padding,
                : width_pad - self.padding,
            ]

        # Channels-first -> channels-last
        x = tf.transpose(x, [0, 2, 3, 1])  # [B, H, W, width]

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

# TensorFlow CNO2d — converted from the PyTorch CNO2d tutorial code
# (ETH Zurich, AI in the Sciences and Engineering).
#
# Interface mirrors FNO2 / CNN: accepts [B, H, W, C_in] (channels-last),
# uses cfg-based configuration, optional input_normalizer.
#
# Performance notes:
#   - Channels-last (NHWC) throughout — no transpose overhead
#   - All spatial sizes are Python ints (static shapes for XLA)
#   - Layer collections stored as named Keras-tracked attributes
#   - Bilinear resize for XLA compatibility
#   - No BatchNorm / no training flag — matches FNO2 usage

import tensorflow as tf
import numpy as np


# =============================================================================
# Helpers
# =============================================================================

def _ceil_to_multiple(n, divisor):
    """Round *n* up to the nearest multiple of *divisor*."""
    return int(np.ceil(n / divisor) * divisor)


# =============================================================================
# Bandlimited activation: upsample 2x -> LeakyReLU -> resample
# =============================================================================

class CNOActivation(tf.keras.layers.Layer):

    def __init__(self, in_h, in_w, out_h, out_w, **kwargs):
        super().__init__(**kwargs)
        self._up_size = [2 * int(in_h), 2 * int(in_w)]
        self._out_size = [int(out_h), int(out_w)]

    def call(self, x):
        x = tf.image.resize(x, self._up_size, method="bilinear")
        x = tf.nn.leaky_relu(x)
        x = tf.image.resize(x, self._out_size, method="bilinear")
        return x


# =============================================================================
# CNOBlock:  Conv2D -> CNOActivation
# =============================================================================

class CNOBlock(tf.keras.layers.Layer):

    def __init__(self, out_channels, in_h, in_w, out_h, out_w, **kwargs):
        super().__init__(**kwargs)
        self.convolution = tf.keras.layers.Conv2D(
            int(out_channels), kernel_size=3, padding="same",
            dtype=tf.float32,
        )
        self.act = CNOActivation(in_h, in_w, out_h, out_w)

    def call(self, x):
        return self.act(self.convolution(x))


# =============================================================================
# LiftProjectBlock:  CNOBlock -> Conv2D
# =============================================================================

class LiftProjectBlock(tf.keras.layers.Layer):

    def __init__(self, out_channels, size_h, size_w,
                 latent_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.inter_block = CNOBlock(
            out_channels=latent_dim,
            in_h=size_h, in_w=size_w, out_h=size_h, out_w=size_w,
        )
        self.convolution = tf.keras.layers.Conv2D(
            int(out_channels), kernel_size=3, padding="same",
            dtype=tf.float32,
        )

    def call(self, x):
        return self.convolution(self.inter_block(x))


# =============================================================================
# ResidualBlock:  Conv -> Activation -> Conv + skip
# =============================================================================

class ResidualBlock(tf.keras.layers.Layer):

    def __init__(self, channels, size_h, size_w, **kwargs):
        super().__init__(**kwargs)
        ch = int(channels)
        self.convolution1 = tf.keras.layers.Conv2D(
            ch, kernel_size=3, padding="same", dtype=tf.float32,
        )
        self.convolution2 = tf.keras.layers.Conv2D(
            ch, kernel_size=3, padding="same", dtype=tf.float32,
        )
        self.act = CNOActivation(size_h, size_w, size_h, size_w)

    def call(self, x):
        out = self.act(self.convolution1(x))
        out = self.convolution2(out)
        return x + out


# =============================================================================
# ResNet:  stack of ResidualBlocks
# =============================================================================

class ResNet(tf.keras.layers.Layer):

    def __init__(self, channels, size_h, size_w, num_blocks, **kwargs):
        super().__init__(**kwargs)
        self.blocks = [
            ResidualBlock(channels, size_h, size_w)
            for _ in range(int(num_blocks))
        ]

    def call(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


# =============================================================================
# CNO2d
# =============================================================================

class CNO2d(tf.keras.Model):
    """Continuous Neural Operator (2-D), TensorFlow / XLA-compatible.

    Parameters from ``cfg`` (Hydra / OmegaConf):
        cfg.processes.iceflow.unified.network.N_layers           (default 3)
        cfg.processes.iceflow.unified.network.N_res              (default 4)
        cfg.processes.iceflow.unified.network.N_res_neck         (default 4)
        cfg.processes.iceflow.unified.network.channel_multiplier (default 16)

    I/O convention (channels-last, same as FNO2 / CNN):
        input  -> [B, H, W, C_in]
        output -> [B, H, W, C_out]   (same H, W as input)
    """

    def __init__(
        self, cfg, nb_inputs, nb_outputs, input_normalizer=None,
        name="CNO2D", **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        cfg_unified = cfg.processes.iceflow.unified
        cfg_numerics = cfg.processes.iceflow.numerics

        N_layers = int(getattr(cfg_unified.network, "N_layers", 3))
        N_res = int(getattr(cfg_unified.network, "N_res", 4))
        N_res_neck = int(getattr(cfg_unified.network, "N_res_neck", 4))
        channel_multiplier = int(getattr(cfg_unified.network, "channel_multiplier", 16))
        use_grid = bool(getattr(cfg_unified.network, "use_grid", True))

        Nz = cfg_numerics.Nz

        self.N_layers = N_layers
        self.lift_dim = channel_multiplier // 2
        self.in_dim = int(nb_inputs)
        self.out_dim = int(nb_outputs)
        self.channel_multiplier = channel_multiplier
        self.use_grid = use_grid
        self.input_normalizer = input_normalizer

        self._N_res = N_res
        self._N_res_neck = N_res_neck
        self._built_cno = False

        # Channel evolution
        encoder_features = [self.lift_dim]
        for i in range(N_layers):
            encoder_features.append(2 ** i * channel_multiplier)

        decoder_features_in = list(encoder_features[1:])
        decoder_features_in.reverse()
        decoder_features_out = list(encoder_features[:-1])
        decoder_features_out.reverse()
        for i in range(1, N_layers):
            decoder_features_in[i] = 2 * decoder_features_in[i]

        self._encoder_features = encoder_features
        self._decoder_features_in = decoder_features_in
        self._decoder_features_out = decoder_features_out

    # ------------------------------------------------------------------
    def _build_cno_layers(self, H, W):
        """Build all sublayers with static Python-int spatial sizes."""
        N = self.N_layers
        ef = self._encoder_features
        dfo = self._decoder_features_out

        eff_in = self.in_dim + (2 if self.use_grid else 0)

        divisor = 2 ** N
        pH = _ceil_to_multiple(H, divisor)
        pW = _ceil_to_multiple(W, divisor)
        self._padded_H = pH
        self._padded_W = pW

        enc_h = [pH // (2 ** i) for i in range(N + 1)]
        enc_w = [pW // (2 ** i) for i in range(N + 1)]
        dec_h = [pH // (2 ** (N - i)) for i in range(N + 1)]
        dec_w = [pW // (2 ** (N - i)) for i in range(N + 1)]
        self._enc_h = enc_h
        self._enc_w = enc_w
        self._dec_h = dec_h
        self._dec_w = dec_w

        # Lift & Project
        self.lift = LiftProjectBlock(out_channels=ef[0], size_h=pH, size_w=pW)
        self.project = LiftProjectBlock(
            out_channels=self.out_dim, size_h=pH, size_w=pW,
        )

        # Encoder
        for i in range(N):
            setattr(self, f"enc_{i}", CNOBlock(
                out_channels=ef[i + 1],
                in_h=enc_h[i], in_w=enc_w[i],
                out_h=enc_h[i + 1], out_w=enc_w[i + 1],
            ))

        # ED expansion
        for i in range(N + 1):
            setattr(self, f"ed_{i}", CNOBlock(
                out_channels=ef[i],
                in_h=enc_h[i], in_w=enc_w[i],
                out_h=dec_h[N - i], out_w=dec_w[N - i],
            ))

        # Decoder
        for i in range(N):
            setattr(self, f"dec_{i}", CNOBlock(
                out_channels=dfo[i],
                in_h=dec_h[i], in_w=dec_w[i],
                out_h=dec_h[i + 1], out_w=dec_w[i + 1],
            ))

        # ResNets
        for l in range(N):
            setattr(self, f"resnet_{l}", ResNet(
                channels=ef[l], size_h=enc_h[l], size_w=enc_w[l],
                num_blocks=self._N_res,
            ))

        self.resnet_neck = ResNet(
            channels=ef[N], size_h=enc_h[N], size_w=enc_w[N],
            num_blocks=self._N_res_neck,
        )

        self._built_cno = True

        # Dummy pass to initialize weights
        dummy = tf.zeros((1, pH, pW, eff_in), dtype=tf.float32)
        self._forward_body(dummy)

    # ------------------------------------------------------------------
    def _get_grid(self, x):
        shape = tf.shape(x)
        bsz, sx, sy = shape[0], shape[1], shape[2]
        gx = tf.reshape(tf.linspace(0.0, 1.0, sx), [1, sx, 1, 1])
        gx = tf.tile(gx, [bsz, 1, sy, 1])
        gy = tf.reshape(tf.linspace(0.0, 1.0, sy), [1, 1, sy, 1])
        gy = tf.tile(gy, [bsz, sx, 1, 1])
        return tf.concat([gx, gy], axis=-1)

    # ------------------------------------------------------------------
    def _forward_body(self, x):
        N = self.N_layers

        x = self.lift(x)
        skip = []

        for i in range(N):
            skip.append(getattr(self, f"resnet_{i}")(x))
            x = getattr(self, f"enc_{i}")(x)

        x = self.resnet_neck(x)

        for i in range(N):
            if i == 0:
                x = getattr(self, f"ed_{N - i}")(x)
            else:
                x = tf.concat([x, getattr(self, f"ed_{N - i}")(skip[-i])],
                              axis=-1)
            x = getattr(self, f"dec_{i}")(x)

        x = tf.concat([x, getattr(self, f"ed_0")(skip[0])], axis=-1)
        x = self.project(x)

        return x

    # ------------------------------------------------------------------
    def call(self, inputs):
        x = inputs
        if x.dtype != tf.float32:
            x = tf.cast(x, tf.float32)

        if self.input_normalizer is not None:
            x = self.input_normalizer(x)

        if self.use_grid:
            x = tf.concat([x, self._get_grid(x)], axis=-1)

        if not self._built_cno:
            H = int(x.shape[1]) if x.shape[1] is not None else int(tf.shape(x)[1])
            W = int(x.shape[2]) if x.shape[2] is not None else int(tf.shape(x)[2])
            self._build_cno_layers(H, W)

        orig_H = tf.shape(x)[1]
        orig_W = tf.shape(x)[2]
        pad_h = self._padded_H - orig_H
        pad_w = self._padded_W - orig_W
        x = tf.pad(x, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]],
                    mode="REFLECT")

        x = self._forward_body(x)

        x = x[:, :orig_H, :orig_W, :]
        return x