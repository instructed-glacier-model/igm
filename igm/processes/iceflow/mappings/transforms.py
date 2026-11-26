# igm/processes/iceflow/mappings/transforms.py
#!/usr/bin/env python3
from __future__ import annotations
import tensorflow as tf
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Type


class ParameterTransform(ABC):
    name: str

    @abstractmethod
    def to_theta(self, x_phys: tf.Tensor, eps: float = 1e-12) -> tf.Tensor: ...

    @abstractmethod
    def to_physical(self, theta: tf.Tensor) -> tf.Tensor: ...

    @abstractmethod
    def theta_bounds(
        self,
        lower_phys: Optional[float],
        upper_phys: Optional[float],
        dtype: tf.dtypes.DType,
        eps: float = 1e-12,
    ) -> Tuple[tf.Tensor, tf.Tensor]: ...


class IdentityTransform(ParameterTransform):
    name = "identity"

    def to_theta(self, x_phys: tf.Tensor, eps: float = 1e-12) -> tf.Tensor:
        return x_phys

    def to_physical(self, theta: tf.Tensor) -> tf.Tensor:
        return theta

    def theta_bounds(self, lower_phys, upper_phys, dtype, eps: float = 1e-12):
        L = (
            -tf.constant(float("inf"), dtype)
            if lower_phys is None
            else tf.constant(lower_phys, dtype)
        )
        U = (
            tf.constant(float("inf"), dtype)
            if upper_phys is None
            else tf.constant(upper_phys, dtype)
        )
        return L, U


class Log10Transform(ParameterTransform):
    name = "log10"

    def to_theta(self, x_phys: tf.Tensor, eps: float = 1e-12) -> tf.Tensor:
        ln10 = tf.constant(2.302585092994046, x_phys.dtype)
        return tf.math.log(tf.maximum(x_phys, tf.cast(eps, x_phys.dtype))) / ln10

    def to_physical(self, theta: tf.Tensor) -> tf.Tensor:
        ln10 = tf.constant(2.302585092994046, theta.dtype)
        return tf.exp(ln10 * theta)

    def theta_bounds(self, lower_phys, upper_phys, dtype, eps: float = 1e-12):
        ln10 = tf.constant(2.302585092994046, dtype)
        eps_t = tf.cast(eps, dtype)

        # enforce same floor used by to_theta()
        if lower_phys is None:
            L = tf.math.log(eps_t) / ln10
        else:
            L_phys = tf.maximum(tf.cast(lower_phys, dtype), eps_t)
            L = tf.math.log(L_phys) / ln10

        if upper_phys is None:
            U = tf.constant(float("inf"), dtype)
        else:
            U_phys = tf.cast(upper_phys, dtype)
            tf.debugging.assert_greater(U_phys, eps_t)
            U = tf.math.log(U_phys) / ln10

        return L, U


class SoftplusTransform(ParameterTransform):
    """Maps ℝ → (0, ∞) with y = softplus(theta) = log(1 + exp(theta))."""

    name = "softplus"

    @staticmethod
    def _softplus_inverse(y: tf.Tensor) -> tf.Tensor:
        """
        Stable inverse of softplus.
        For small y: log(expm1(y))
        For large y: y + log1p(-exp(-y))
        """
        y = tf.convert_to_tensor(y)
        # threshold chosen so expm1(y) is safe in float32 and float64
        thresh = tf.cast(20.0, y.dtype)
        small = tf.math.log(tf.math.expm1(y))
        large = y + tf.math.log1p(-tf.exp(-y))
        return tf.where(y < thresh, small, large)

    def to_theta(self, x_phys: tf.Tensor, eps: float = 1e-12) -> tf.Tensor:
        x_phys = tf.convert_to_tensor(x_phys)
        y = tf.maximum(x_phys, tf.cast(eps, x_phys.dtype))
        return self._softplus_inverse(y)

    def to_physical(self, theta: tf.Tensor) -> tf.Tensor:
        return tf.nn.softplus(theta)

    def theta_bounds(
        self,
        lower_phys: Optional[float],
        upper_phys: Optional[float],
        dtype: tf.dtypes.DType,
        eps: float = 1e-12,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        if (lower_phys is None) or (lower_phys <= 0.0):
            L = -tf.constant(float("inf"), dtype)
        else:
            L = self._softplus_inverse(eps)

        if upper_phys is None:
            U = tf.constant(float("inf"), dtype)
        else:
            if upper_phys <= 0.0:
                raise ValueError("Upper bound must be > 0 for softplus.")
            U = self._softplus_inverse(tf.constant(upper_phys, dtype))

        return L, U


class MpaToKpaThetaTransform(ParameterTransform):
    """
    Optimize in kPa while the physical parameter remains MPa.

    theta = 1000 * x_phys(MPa)   [kPa]
    x_phys = theta / 1000        [MPa]
    """

    name = "mpa_to_kpa_theta"

    def to_theta(self, x_phys: tf.Tensor, eps: float = 1e-12) -> tf.Tensor:
        return x_phys * tf.cast(1000.0, x_phys.dtype)

    def to_physical(self, theta: tf.Tensor) -> tf.Tensor:
        return theta / tf.cast(1000.0, theta.dtype)

    def theta_bounds(self, lower_phys, upper_phys, dtype, eps: float = 1e-12):
        L = (
            -tf.constant(float("inf"), dtype)
            if lower_phys is None
            else tf.constant(lower_phys * 1000.0, dtype)
        )
        U = (
            tf.constant(float("inf"), dtype)
            if upper_phys is None
            else tf.constant(upper_phys * 1000.0, dtype)
        )
        return L, U


# registry
TRANSFORMS: Dict[str, Type[ParameterTransform]] = {
    "identity": IdentityTransform,
    "log10": Log10Transform,
    "softplus": SoftplusTransform,
    "mpa_to_kpa_theta": MpaToKpaThetaTransform,
}
