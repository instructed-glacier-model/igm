import tensorflow as tf
from igm.utils.math.precision import normalize_precision


def _complex_dtype_for(real_dtype):
    """Return the matching complex dtype for a given real dtype."""
    if real_dtype == tf.float64:
        return tf.complex128
    return tf.complex64


class FNO(tf.keras.Model):
    """
    Simplified Fourier Neural Operator - operates in frequency domain for global information
    """

    def __init__(self, cfg, nb_inputs, nb_outputs, input_normalizer=None):
        super(FNO, self).__init__()

        precision = cfg.processes.iceflow.numerics.precision
        self.dtype_model = normalize_precision(precision)
        self.complex_dtype = _complex_dtype_for(self.dtype_model)

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
            x_freq = tf.signal.fft2d(tf.cast(x, self.complex_dtype))

            # Apply spectral convolution (simplified: using real part)
            x_freq_real = tf.cast(tf.math.real(x_freq), self.dtype_model)
            x_freq_imag = tf.cast(tf.math.imag(x_freq), self.dtype_model)

            # Process in frequency domain
            x_freq_processed = fourier_layer(x_freq_real)

            # IFFT back to spatial domain
            x_freq_complex = tf.cast(x_freq_processed, self.complex_dtype)
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
# SpectralConv2D: 2D Fourier layer (Li et al. FNO2d style)
# --------------------------------------------------------------------
class SpectralConv2D(tf.keras.layers.Layer):
    """
    2D Fourier layer.

    x: [B, C_in, H, W] (channels-first, real)
    -> rFFT -> multiply low modes with learned complex weights -> irFFT
    -> [B, C_out, H, W]
    """

    def __init__(
        self, in_channels, out_channels, modes1, modes2, real_dtype=tf.float32, **kwargs
    ):
        super().__init__(**kwargs)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)
        self.real_dtype = real_dtype
        self.complex_dtype = _complex_dtype_for(real_dtype)

        # scale is as in Li's implementation
        self.scale = 1.0 / (self.in_channels * self.out_channels)

        self.w1_real = None
        self.w1_imag = None
        self.w2_real = None
        self.w2_imag = None

    def build(self, input_shape):
        # input_shape: (B, C_in, H, W)
        if input_shape[1] is not None:
            if int(input_shape[1]) != self.in_channels:
                raise ValueError(
                    f"SpectralConv2D: expected {self.in_channels} input channels, "
                    f"got {input_shape[1]}"
                )

        # Validate modes against input dimensions
        # For rfft2d: W_r = W//2 + 1
        if input_shape[2] is not None:  # Height known
            H = int(input_shape[2])
            if self.modes1 > H:
                raise ValueError(
                    f"SpectralConv2D: modes1={self.modes1} exceeds input height H={H}. "
                    f"modes1 must be <= H for proper spectral truncation."
                )

        if input_shape[3] is not None:  # Width known
            W = int(input_shape[3])
            W_r = W // 2 + 1  # rfft output size
            if self.modes2 > W_r:
                raise ValueError(
                    f"SpectralConv2D: modes2={self.modes2} exceeds rfft width W_r={W_r} (W={W}). "
                    f"modes2 must be <= W//2 + 1 for proper spectral truncation."
                )

        limit = tf.math.sqrt(tf.cast(self.scale, self.real_dtype))
        init = tf.keras.initializers.RandomUniform(minval=-limit, maxval=limit)

        # weights1: top modes [:modes1, :modes2]
        self.w1_real = self.add_weight(
            name="w1_real",
            shape=(self.in_channels, self.out_channels, self.modes1, self.modes2),
            initializer=init,
            dtype=self.real_dtype,
            trainable=True,
        )
        self.w1_imag = self.add_weight(
            name="w1_imag",
            shape=(self.in_channels, self.out_channels, self.modes1, self.modes2),
            initializer=init,
            dtype=self.real_dtype,
            trainable=True,
        )

        # weights2: bottom modes [-modes1:, :modes2]
        self.w2_real = self.add_weight(
            name="w2_real",
            shape=(self.in_channels, self.out_channels, self.modes1, self.modes2),
            initializer=init,
            dtype=self.real_dtype,
            trainable=True,
        )
        self.w2_imag = self.add_weight(
            name="w2_imag",
            shape=(self.in_channels, self.out_channels, self.modes1, self.modes2),
            initializer=init,
            dtype=self.real_dtype,
            trainable=True,
        )

        super().build(input_shape)

    def _compl_mul2d(self, x_ft, w_real, w_imag):
        """
        Complex multiplication:
          x_ft: [B, C_in, m1, m2]
          w_*:  [C_in, C_out, m1, m2]
          ->    [B, C_out, m1, m2]
        """
        weights = tf.complex(w_real, w_imag)
        return tf.einsum("bixy,ioxy->boxy", x_ft, weights)

    def call(self, x):
        """
        x: [B, C_in, H, W], real
        """
        if x.dtype != self.real_dtype:
            x = tf.cast(x, self.real_dtype)

        height = tf.shape(x)[2]
        width = tf.shape(x)[3]

        # rFFT over last two dims (no fftshift)
        x_ft = tf.signal.rfft2d(x)  # [B, C_in, H, W_r], W_r = W//2 + 1
        h_ft = tf.shape(x_ft)[2]
        w_r = tf.shape(x_ft)[3]

        # Top modes
        x_ft_top = x_ft[:, :, : self.modes1, : self.modes2]  # [B, C_in, m1, m2]
        out_ft_top = self._compl_mul2d(x_ft_top, self.w1_real, self.w1_imag)

        # Bottom modes
        x_ft_bottom = x_ft[:, :, -self.modes1 :, : self.modes2]  # [B, C_in, m1, m2]
        out_ft_bottom = self._compl_mul2d(x_ft_bottom, self.w2_real, self.w2_imag)

        # Pad into full [B, C_out, H, W_r]
        pad_top = [
            [0, 0],
            [0, 0],
            [0, h_ft - self.modes1],
            [0, w_r - self.modes2],
        ]
        out_ft_top_full = tf.pad(out_ft_top, pad_top)

        pad_bottom = [
            [0, 0],
            [0, 0],
            [h_ft - self.modes1, 0],
            [0, w_r - self.modes2],
        ]
        out_ft_bottom_full = tf.pad(out_ft_bottom, pad_bottom)

        out_ft = out_ft_top_full + out_ft_bottom_full  # [B, C_out, H, W_r]

        # Inverse rFFT back to real space
        x_out = tf.signal.irfft2d(
            out_ft, fft_length=[height, width]
        )  # [B, C_out, H, W]
        return x_out


class FNO2(tf.keras.Model):

    def __init__(
        self, cfg, nb_inputs, nb_outputs, input_normalizer=None, name="FNO2D", **kwargs
    ):
        super().__init__(name=name, **kwargs)

        cfg_unified = cfg.processes.iceflow.unified
        cfg_numerics = cfg.processes.iceflow.numerics

        precision = cfg_numerics.precision
        self.dtype_model = normalize_precision(precision)

        width = getattr(cfg_unified.network, "width", 32)
        modes1 = getattr(cfg_unified.network, "modes1", 8)
        modes2 = getattr(cfg_unified.network, "modes2", modes1)
        padding = getattr(cfg_unified.network, "padding", modes1 + 1)
        use_grid = getattr(cfg_unified.network, "use_grid", False)

        Nz = cfg_numerics.Nz

        self.modes1 = int(modes1)
        self.modes2 = int(modes2)
        self.width = int(width)
        self.input_channels = int(nb_inputs)
        self.output_channels = int(nb_outputs)
        self.nz = Nz
        self.padding = int(padding)
        self.use_grid = bool(use_grid)
        self.input_normalizer = input_normalizer

        # Lifting: linear map on last channel dimension
        self.fc0 = tf.keras.layers.Dense(self.width, dtype=self.dtype_model)

        # Fourier layers (channels-first)
        self.conv0 = SpectralConv2D(
            self.width,
            self.width,
            self.modes1,
            self.modes2,
            real_dtype=self.dtype_model,
        )
        self.conv1 = SpectralConv2D(
            self.width,
            self.width,
            self.modes1,
            self.modes2,
            real_dtype=self.dtype_model,
        )
        self.conv2 = SpectralConv2D(
            self.width,
            self.width,
            self.modes1,
            self.modes2,
            real_dtype=self.dtype_model,
        )
        self.conv3 = SpectralConv2D(
            self.width,
            self.width,
            self.modes1,
            self.modes2,
            real_dtype=self.dtype_model,
        )

        # 1x1 conv skips in channels-first format
        self.w0 = tf.keras.layers.Conv2D(
            self.width,
            kernel_size=1,
            data_format="channels_first",
            use_bias=True,
            dtype=self.dtype_model,
        )
        self.w1 = tf.keras.layers.Conv2D(
            self.width,
            kernel_size=1,
            data_format="channels_first",
            use_bias=True,
            dtype=self.dtype_model,
        )
        self.w2 = tf.keras.layers.Conv2D(
            self.width,
            kernel_size=1,
            data_format="channels_first",
            use_bias=True,
            dtype=self.dtype_model,
        )
        self.w3 = tf.keras.layers.Conv2D(
            self.width,
            kernel_size=1,
            data_format="channels_first",
            use_bias=True,
            dtype=self.dtype_model,
        )

        # Projection head
        self.fc1 = tf.keras.layers.Dense(128, dtype=self.dtype_model)
        self.fc2 = tf.keras.layers.Dense(self.output_channels, dtype=self.dtype_model)

        # Dummy forward pass to build variables
        dummy_H = max(16, self.modes1 + 1)
        dummy_W = max(16, 2 * self.modes2 + 2)
        dummy_input = tf.zeros((1, dummy_H, dummy_W, nb_inputs), dtype=self.dtype_model)
        _ = self(dummy_input, training=False)

    def _get_grid(self, x):
        """
        Generate [B, H, W, 2] grid with coordinates in [0,1].
        x: [B, H, W, C]
        """
        shape = tf.shape(x)
        batch_size = shape[0]
        size_x = shape[1]
        size_y = shape[2]

        gridx = tf.cast(tf.linspace(0.0, 1.0, size_x), self.dtype_model)
        gridx = tf.reshape(gridx, [1, size_x, 1, 1])
        gridx = tf.tile(gridx, [batch_size, 1, size_y, 1])

        gridy = tf.cast(tf.linspace(0.0, 1.0, size_y), self.dtype_model)
        gridy = tf.reshape(gridy, [1, 1, size_y, 1])
        gridy = tf.tile(gridy, [batch_size, size_x, 1, 1])

        grid = tf.concat([gridx, gridy], axis=-1)  # [B, H, W, 2]
        return grid

    def call(self, inputs, training=False):
        """
        inputs: [N, H, W, C_in]
        returns: [N, H, W, C_out]
        """
        x = inputs
        if x.dtype != self.dtype_model:
            x = tf.cast(x, self.dtype_model)

        if self.input_normalizer is not None:
            x = self.input_normalizer(x, training=training)

        # Optionally append (x,y) grid
        if self.use_grid:
            grid = self._get_grid(x)
            x = tf.concat([x, grid], axis=-1)  # [B, H, W, C_in (+2)]

        # Lift to hidden width
        x = self.fc0(x)  # [B, H, W, width]

        # Switch to channels-first for the spectral blocks
        x = tf.transpose(x, [0, 3, 1, 2])  # [B, width, H, W]

        # Optional padding (emulate non-periodic boundaries à la Li)
        if self.padding > 0:
            paddings = [[0, 0], [0, 0], [0, self.padding], [0, self.padding]]
            x = tf.pad(x, paddings)

        H_pad = tf.shape(x)[2]
        W_pad = tf.shape(x)[3]

        # 4 Fourier blocks
        # Block 0
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = tf.nn.gelu(x1 + x2)

        # Block 1
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = tf.nn.gelu(x1 + x2)

        # Block 2
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = tf.nn.gelu(x1 + x2)

        # Block 3 (no activation after last)
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # Remove padding
        if self.padding > 0:
            x = x[:, :, : H_pad - self.padding, : W_pad - self.padding]

        # Back to channels-last
        x = tf.transpose(x, [0, 2, 3, 1])  # [B, H, W, width]

        # Per-pixel MLP head
        x = self.fc1(x)
        x = tf.nn.gelu(x)
        x = self.fc2(x)  # [B, H, W, output_channels]

        return x
