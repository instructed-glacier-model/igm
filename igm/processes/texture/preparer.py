import tensorflow as tf

from typing import Tuple, Any
from .emulator import Emulator

class Pix2PixHDImagePreparer:
    def __init__(self, cfg: Any, state: Any, ndvi_emulator: Emulator = None) -> None:
        if ndvi_emulator is None and (not hasattr(state, "ndvi")):
            raise ValueError(
                "Emulator is required for NDVI if no NDVI is provided."
            )  # ? getting a type error with None here... could be a bug to look into
        self.state = state
        self.cfg = cfg
        self.ndvi_emulator = ndvi_emulator

    def get_features(self) -> None:
        topg = self.state.topg
        thk = self.state.thk
        vx = self.state.uvelsurf
        vy = self.state.vvelsurf    
        temp = self.state.meantemp
        prec = self.state.meanprec
        
        prec = self.prec_units(prec)

        if hasattr(self.state, "ndvi"):
            ndvi = self.state.ndvi
        else:
            ndvi = self.compute_ndvi(temp, prec)

        if hasattr(self.state, "water"):
            water = self.state.water
        else:
            water = self.compute_water(topg.shape)

        self.image = tf.stack([topg, water, vx, thk, prec, ndvi, vy, temp], axis=0)

    def prec_units(self, prec: tf.Tensor) -> tf.Tensor:
        return (prec / self.cfg.processes.texture.divide_by_density) * (1 / 12)  # (m / yr. -> mm / month)

    def compute_ndvi(self, temp: tf.Tensor, prec: tf.Tensor) -> tf.Tensor:
        temp_flat = tf.reshape(temp, [-1])
        prec_flat = tf.reshape(prec, [-1])
        climate = tf.stack((temp_flat, prec_flat))

        ndvi = self.ndvi_emulator.apply(climate)
        ndvi = tf.reshape(ndvi, [temp.shape[0], temp.shape[1]])
        ndvi = tf.clip_by_value(ndvi, 0, 255)

        return ndvi

    def compute_water(self, tensor_shape: Tuple[int, int]):
        return tf.zeros(shape=tensor_shape)  # assumes no water for now...

    def prepare_batch(self) -> tf.Tensor:
        image = tf.transpose(self.image, [1, 2, 0])
        image = tf.expand_dims(image, axis=0)

        return image