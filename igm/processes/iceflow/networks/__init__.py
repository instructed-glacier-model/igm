from .architectures import CNN, Architectures
from .normalizations import (
    AdaptiveAffineLayer,
    FixedAffineLayer,
    IdentityTransformation,
    StandardizationLayer,
)

NormalizationsDict = {
    "fixed": FixedAffineLayer,
    "adaptive": AdaptiveAffineLayer,
    "automatic": StandardizationLayer,
    "none": IdentityTransformation,
}
