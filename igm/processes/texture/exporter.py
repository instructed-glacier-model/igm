import tensorflow as tf
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Any
from .normalizer import Normalizer

class Exporter(ABC):
    @abstractmethod
    def export(self, image: tf.Tensor, filename: str) -> None:
        pass


class PngExporter(Exporter):

    def convert_8bit(self, image: tf.Tensor, image_normalizer: Normalizer) -> tf.Tensor:
        out = image_normalizer.unnormalize(image)
        out = tf.cast(out, tf.uint8)
        out = tf.squeeze(out).numpy()

        return out

    def unpad(self, image: tf.Tensor, padding_parameters) -> tf.Tensor:
        import numpy as np
        n_dimensions = len(padding_parameters)
        padded_image = image
        for i, dimension in enumerate(padding_parameters):
            if not np.all(np.equal(dimension,0)):
                begin = [0] * n_dimensions
                size = [-1] * n_dimensions
                begin[i] = dimension[0]
                size[i] = image.shape[i] - dimension[0] - dimension[1]
                padded_image = tf.slice(padded_image, begin=begin, size=size)
        unpadded_image = padded_image
        
        return unpadded_image

    def export(self, image: tf.Tensor, filename: str) -> None:
        plt.imsave(fname=filename, arr=image, format='png', dpi=300, origin='lower')


class TiffExporter(Exporter):
    
    def __init__(self, state: Any):
        self.state = state
        
    # this is a copy of the PngExporter, but it should be changed to export tiff files
    def convert_8bit(self, image: tf.Tensor, image_normalizer: Normalizer) -> tf.Tensor:
        out = image_normalizer.unnormalize(image)
        out = tf.cast(out, tf.uint8)
        out = tf.squeeze(out).numpy()

        return out

    # this is a copy of the PngExporter, but it should be changed to export tiff files
    def unpad(self, image: tf.Tensor, padding_parameters) -> tf.Tensor:
        import numpy as np
        n_dimensions = len(padding_parameters)
        padded_image = image
        for i, dimension in enumerate(padding_parameters):
            if not np.all(np.equal(dimension,0)):
                begin = [0] * n_dimensions
                size = [-1] * n_dimensions
                begin[i] = dimension[0]
                size[i] = image.shape[i] - dimension[0] - dimension[1]
                padded_image = tf.slice(padded_image, begin=begin, size=size)
        unpadded_image = padded_image
        
        return unpadded_image    
    
    def export(self, image: tf.Tensor, filename: str) -> None:

        import rasterio
        from rasterio.enums import ColorInterp
        import numpy as np
 
        code = 32632 # UTM 32N (oviously this should be a parameter in the future)
               
        ny=image.shape[0]
        nx=image.shape[1]
        ch=image.shape[2]    
        
        xmin = min(self.state.x)
        xmax = max(self.state.x)
        ymin = min(self.state.y)
        ymax = max(self.state.y)
        
        xres = (xmax - xmin) / float(nx)
        yres = (ymax - ymin) / float(ny)
        
        transform = rasterio.Affine.translation(xmin - xres / 2, ymin - yres / 2)  * rasterio.Affine.scale(xres, yres)

        with rasterio.open(filename, mode="w", driver="GTiff", height=ny, width=nx, 
                           count=3, dtype=np.uint8, transform=transform,) as src:
            src.write(np.rollaxis(image, -1, 0))
            src.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue]