from .components import (
	GravityComponent,
 	GravityParams,
	ViscosityComponent,
	ViscosityParams,
	SlidingWeertmanComponent,
	SlidingWeertmanParams,
	FloatingComponent,
	FloatingParams,
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
    "floating": FloatingParams
}