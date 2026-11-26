from .components import (
    EnergyComponent,
    FloatingComponent,
    FloatingParams,
    GravityComponent,
    GravityParams,
    SlidingComponents,
    SlidingParams,
    ViscosityComponent,
    ViscosityParams,
    get_floating_params_args,
    get_gravity_params_args,
    get_sliding_params_args,
    get_viscosity_params_args,
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
