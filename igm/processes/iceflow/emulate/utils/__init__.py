from .architectures import Architectures, CNN
from .normalizations import (
    FixedAffineLayer,
    AdaptiveAffineLayer,
    StandardizationLayer,
    IdentityTransformation,
)

from .misc import (
    get_emulator_path,
    get_effective_pressure_precentage,
    save_iceflow_model,
)

NormalizationsDict = {
    "fixed": FixedAffineLayer,
    "adaptive": AdaptiveAffineLayer,
    "automatic": StandardizationLayer,
    "none": IdentityTransformation,
}

from .networks import cnn, unet
