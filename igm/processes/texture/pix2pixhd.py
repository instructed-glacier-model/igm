from .normalizer import Normalizer
from .image_data import ImageData
from .exporter import Exporter
from typing import Any
import logging


class Pix2PixHDPipeline:
    def __init__(
        self,
        feature_normalizer: Normalizer,
        image_normalizer: Normalizer,
        image: ImageData,
        exporter: Exporter,
        model: Any,
        state: Any,
        cfg: Any,
    ):
        self.feature_normalizer: Normalizer = feature_normalizer
        self.image_normalizer: Normalizer = image_normalizer
        self.image: ImageData = image
        self.exporter: Exporter = exporter
        self.model: Any = model
        self.state: Any = state
        self.cfg: Any = cfg

    def run(self) -> None:
        year = int(round(self.state.t.numpy()))

        self.image.values = self.feature_normalizer.normalize_all(self.image.values)

        image = self.model(self.image.values, training=False)

        logging.info(f"Output Image shape from model: {image.shape}")
        if not self.image.square:
            image = self.exporter.unpad(
                image=image, padding_parameters=self.exporter.padding_parameters
            )

        logging.info(f"Output Image shape after unpadding (if needed): {image.shape}")
        # Converts raw model output [-1,1] to [0,255] for imshow
        image = self.exporter.convert_8bit(
            image=image, image_normalizer=self.image_normalizer
        )
        logging.info(f"Image type after converting to 8bit (if needed): {image.dtype}")
        logging.debug(
            f"First column from Image after converting to 8bit (if needed): {image[:,0]}"
        )
        self.exporter.export(
            image=image, filename=f"texture_{year}.{self.cfg.processes.texture.format}"
        )
