import tensorflow as tf

class TextureModelNotFoundError(Exception):
    pass

def resize_image(
    input_image: tf.Tensor,
    height: int,
    width: int,
    upsampling_method=tf.image.ResizeMethod.BICUBIC,
):
    input_image = tf.image.resize(
        input_image, [height, width], method=upsampling_method
    )

    return input_image