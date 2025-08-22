from .components import (
	GravityComponent,
 	GravityEnergyParams,
	ViscosityComponent,
	ViscosityEnergyParams,
	SlidingWeertmanComponent,
	SlidingWeertmanEnergyParams,
	FloatingComponent,
	FloatingEnergyParams,
)

EnergyComponents = {
	"gravity": GravityComponent,
	"viscosity": ViscosityComponent,
    "sliding_weertman": SlidingWeertmanComponent,
	"floating": FloatingComponent,
}

EnergyParams = {
    "gravity": GravityEnergyParams,
    "viscosity": ViscosityEnergyParams,
    "sliding_weertman": SlidingWeertmanEnergyParams,
    "floating": FloatingEnergyParams
}