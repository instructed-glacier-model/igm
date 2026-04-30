from abc import ABC, abstractmethod
import tensorflow as tf
from ...emulate.utils.normalizations import StandardizationLayer, FixedAffineLayer


class Normalizer(ABC):

    @abstractmethod
    def set_stats(means: tf.Tensor, variances: tf.Tensor):
        pass

    @abstractmethod
    def compute_stats(inputs: tf.Tensor):
        pass


class IdentityNormalizer(Normalizer):

    def set_stats(self, means, variances):
        pass

    def compute_stats(self, inputs):
        return 0, 0  # use another interface...


class NetworkNormalizer(Normalizer):

    def __init__(
        self, method: tf.keras.layers.Layer
    ):  # make the layer a base class for typing
        self.method = method

    def set_stats(self, means: tf.Tensor, variances: tf.Tensor):
        self.method.set_stats(means, variances)

    def compute_stats(self, inputs: tf.Tensor):
        return self.method.compute_stats(inputs)

# TODO: Look into various data drift metrics
def mahalanobis_distance(
    previous_means, previous_variances, current_means
):
    """Computes the equivalent mahalonbis distance between the previous and current distributions to detect data drift.
    After standardizing the data, this reduces to a simple eulcidean norm and further a single z-score if its a single point.
    """

    mean_drifts = tf.abs(previous_means - current_means) / tf.sqrt(previous_variances)

    return mean_drifts


def cosine_simularity(previous_means: tf.Tensor, current_means: tf.Tensor) -> float:

    A = previous_means
    B = current_means

    cosine_simularity = tf.tensordot(A, B) / (tf.linalg.norm(A) * tf.linalg.norm(B))

    return cosine_simularity

def hotelling_T2(
    previous_means: tf.Tensor,
    previous_variances: tf.Tensor,
    current_means: tf.Tensor,
    n: int,
) -> tf.Tensor:
    """Computes the equivalent Hotelling T squared distance between the previous and current distributions (mean only) to detect data drift."""

    mean_drifts = tf.abs(previous_means - current_means) / (
        tf.sqrt(previous_variances) + 1e-13
    )
    T_squared = float(n) * tf.reduce_sum(mean_drifts**2)

    return T_squared


def is_distribution_shifted(mapping, inputs, init: bool = False, threshold=1.0) -> bool:
    """Checking if the distribution has shifted"""

    # For now, I am choosing a simple metric (mean shift) for testing but we should make this modular as defined below:
    # 1. Choose metric in YAML
    # 2. Define threshold in YAML
    # 3. Determine if metric is triggered
    # 4. Set retrain=True (status DEFAULT)

    means_computed, variances_computed = mapping.input_normalizer.compute_stats(inputs)
    means, variances = mapping.input_normalizer.mean, mapping.input_normalizer.variance

    n = tf.size(inputs) // inputs.shape[-1]
    score = hotelling_T2(means, variances, means_computed, n)
    distribution_shifted = True if score > threshold else False
    # ! Threshold right now is linked to z-score, but we can have it inside mapping for generality

    if distribution_shifted or init:
        mapping.input_normalizer.set_stats(means_computed, variances_computed)
        distribution_shifted = True

    return distribution_shifted