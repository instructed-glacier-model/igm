from .synthetic_costs import (
	quadratic_test_cost_moderate,
	quadratic_test_cost_extreme,
	nonconvex_ackley,
	nonconvex_styblinski_tang,
	nonconvex_rastrigin,
	nonconvex_sine_modulated_quadratic,
)

SyntheticCosts = {
	'quadratic_moderate': quadratic_test_cost_moderate,
	'quadratic_extreme': quadratic_test_cost_extreme,
	'nonconvex_ackley': nonconvex_ackley,
	'nonconvex_styblinski_tang': nonconvex_styblinski_tang,
	'nonconvex_rastrigin': nonconvex_rastrigin,
	'nonconvex_sine_modulated_quadratic': nonconvex_sine_modulated_quadratic,
}