from .data_preprocessing import (
    fieldin_to_X_2d,
    split_field_into_patches,
)

from .velocities import (
    boundvel,
    clip_max_velbar,
    get_velbase,
    get_velbase_1,
    get_velsurf,
    get_velsurf_1,
    get_velbar,
    get_velbar_1,
)

from .vertical_discretization import (
    compute_levels,
    compute_zeta_dzeta,
    define_vertical_weight,
)
from .fields import (
    initialize_iceflow_fields,
)
