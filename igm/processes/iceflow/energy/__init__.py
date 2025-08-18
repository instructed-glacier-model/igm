from .components import (
    GravityComponent,
    GravityParams,
    get_gravity_params_args,
    ViscosityComponent,
    ViscosityParams,
    get_viscosity_params_args,
    SlidingWeertmanComponent,
    SlidingWeertmanParams,
    get_sliding_weertman_params_args,
    FloatingComponent,
    FloatingParams,
    get_floating_params_args,
)

EnergyComponents = {
    "gravity": GravityComponent,
    "viscosity": ViscosityComponent,
    "sliding_weertman": SlidingWeertmanComponent,
    "floating": FloatingComponent,
}

EnergyParams = {
    "gravity": GravityParams,
    "viscosity": ViscosityParams,
    "sliding_weertman": SlidingWeertmanParams,
    "floating": FloatingParams,
}

get_energy_params_args = {
    "gravity": get_gravity_params_args,
    "viscosity": get_viscosity_params_args,
    "sliding_weertman": get_sliding_weertman_params_args,
    "floating": get_floating_params_args,
}
