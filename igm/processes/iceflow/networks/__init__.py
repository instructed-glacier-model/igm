from .architectures import Architectures, CNN
from .normalizations import (
    FixedAffineLayer,
    AdaptiveAffineLayer,
    StandardizationLayer,
    IdentityTransformation,
)

NormalizationsDict = {
    "fixed": FixedAffineLayer,
    "adaptive": AdaptiveAffineLayer,
    "automatic": StandardizationLayer,
    "none": IdentityTransformation,
}
