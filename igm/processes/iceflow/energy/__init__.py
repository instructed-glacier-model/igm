from .components import (
	GravityComponent,
 	GravityParams,
	ViscosityComponent,
	ViscosityParams,
	FloatingComponent,
	FloatingParams,
)

EnergyComponents = {
	"gravity": GravityComponent,
	"viscosity": ViscosityComponent,
	"floating": FloatingComponent,
}

EnergyParams = {
    "gravity": GravityParams,
    "viscosity": ViscosityParams,
    "floating": FloatingParams
}