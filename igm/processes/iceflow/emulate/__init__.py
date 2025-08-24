from .emulated import get_emulated_bag, EmulatedParams, update_iceflow_emulated
from .emulator import (
    get_emulator_bag,
    EmulatorParams,
    update_iceflow_emulator,
    initialize_iceflow_emulator,
)  # ! this initializer works for both emulated and emulator - maybe make it clearer...


from .utils import (
    save_iceflow_model,
    get_emulator_path,
    get_effective_pressure_precentage,
    cnn,
    unet,
)
