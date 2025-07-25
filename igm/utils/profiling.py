import tensorflow as tf
import warnings

try:
	import nvtx
except ImportError:
    warnings.warn(
		"NVTX is not installed. Profiling will not be available.")

def srange(message, color):
    tf.test.experimental.sync_devices()
    return nvtx.start_range(message, color)

def erange(rng):
    tf.test.experimental.sync_devices()
    nvtx.end_range(rng)