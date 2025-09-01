import tensorflow as tf

from typing import Tuple, Any
from dataclasses import dataclass

from .utils import resize_image
# from igm import State

@dataclass
class ImageData:
    values: tf.Tensor
    height: int
    width: int
    state: Any
    upsampled: bool = False
    square: bool = False

    def upsample(self, height: int, width: int) -> None:
        # self.values = resize_image(self.values, height, width)
        self.values = self.pad_image(self.values, height, width)
        self.upsampled = True

    def compute_shape(self, resolution: int) -> Tuple[int, int]:
        self.upsample_resolution = resolution
        if self.height == self.width:
            self.height = resolution
            self.width = resolution
            self.square = True

            return resolution, resolution
        elif self.height < self.width:
            ratio = resolution / self.width
            new_width = int(tf.math.ceil(ratio * self.width))
            new_height = int(tf.math.ceil(ratio * self.height))

            if new_height % 2 != 0:
                new_height += 1

            self.height = new_height
            self.width = new_width

            return new_width, new_height
        else:
            ratio = resolution / self.height
            new_width = int(tf.math.ceil(ratio * self.width))
            new_height = int(tf.math.ceil(ratio * self.height))

            if new_width % 2 != 0:
                new_width += 1

            self.height = new_height
            self.width = new_width

            return new_width, new_height
    

    def make_square(self, method: str = "reflect"):
        if not self.upsampled:
            self.upsample(self.height, self.width)
            padding_parameters = [
                (0, 0),
                (0, 0),
                (0, 0),
                (0, 0),
            ]  # [bs, h, w, c]
        if self.height > self.width:
            pad_width = (self.height - self.width) // 2
            padding_parameters = [
                (0, 0),
                (0, 0),
                (pad_width, pad_width),
                (0, 0),
            ]  # [bs, h, w + padding, c]
        else:
            pad_width = (self.width - self.height) // 2
            padding_parameters = [
                (0, 0),
                (pad_width, pad_width),
                (0, 0),
                (0, 0),
            ]  # [bs, h + padding, w, c]

        self.values = tf.pad(self.values, padding_parameters, mode=method)

        return padding_parameters
    
    def pad_image(self, resolution_height, resolution_width, method: str = "reflect"):
        import numpy as np

        pad_width = (resolution_width - self.width) // 2
        pad_height = (resolution_height - self.height) // 2
        
        # edge case if the resulting resolution is odd...
        width_offset = (resolution_width - self.width) % 2
        height_offset = (resolution_height - self.height) % 2
        padding_parameters = np.array([
            [0, 0],
            [pad_height, pad_height + height_offset],
            [pad_width, pad_width  + width_offset],
            [0, 0],
        ])  # [bs, h + padding, w + padding, c]

        self.values = tf.pad(self.values, padding_parameters, mode=method)
        self.height += 2 * pad_height + height_offset
        self.width += 2 * pad_width + width_offset
        
        return padding_parameters

    def run(self) -> None:
        new_width, new_height = self.compute_shape(resolution=self.upsample_resolution)
        self.upsample(width=new_width, height=new_height)
        if not self.square:
            padding_parameters = self.make_square()
            self.state.texture_exporter.padding_parameters = padding_parameters