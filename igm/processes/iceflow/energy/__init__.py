from .components import (
    GravityComponent,
    GravityParams,
    get_gravity_params_args,
    ViscosityComponent,
    ViscosityParams,
    get_viscosity_params_args,
    FloatingComponent,
    FloatingParams,
    get_floating_params_args,
    SlidingComponents,
    SlidingParams,
    get_sliding_params_args,
)

EnergyComponents = {
    "gravity": GravityComponent,
    "viscosity": ViscosityComponent,
    "floating": FloatingComponent,
    "sliding": SlidingComponents,
}

EnergyParams = {
    "gravity": GravityParams,
    "viscosity": ViscosityParams,
    "floating": FloatingParams,
    "sliding": SlidingParams,
}

get_energy_params_args = {
    "gravity": get_gravity_params_args,
    "viscosity": get_viscosity_params_args,
    "floating": get_floating_params_args,
    "sliding": get_sliding_params_args,
}
