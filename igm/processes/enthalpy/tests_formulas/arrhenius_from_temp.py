import numpy as np
import matplotlib.pyplot as plt

def arrhenius_from_temp_tf(Tpa, omega):
    # Budd Paterson Law adapted for (T,omega), return result in MPa^{-3} y^{-1}
    # (Aschwanden and al, JOG, 2012) & (Paterson 1994)

    A = np.where(Tpa < 263.15, 3.985 * 10 ** (-13), 1.916 * 10**3)  # s^{-1} Pa^{-3}
    Q = np.where(Tpa < 263.15, 60000.0, 139000.0)  # J mol-1

    #chunit = (10**18) * 31556926  # change unit from Pa^{-3} s^{-1} to MPa^{-3} y^{-1}

    return (
        (1.0 + 181.25 * np.minimum(omega, 0.01))
        * A
        * np.exp(-Q / (8.314 * Tpa))
    )

# Define omega values to loop over (hence creating different curves in plot)
omega_values = [0.0001, 0.001, 0.005, 0.01]

# Generate synthetic data for Tpa
Tpa_values = np.linspace(223.15, 273.15, 100)  # Adjust as needed

# Create the plot
plt.figure(figsize=(10, 7))
plt.yscale('log')  # Set y-axis to logarithmic scale
plt.ylim([1.0e-27, 1.0e-23])  # Set y-axis limits
plt.xlim([223.15, 273.15]) # Set x-axis limits
plt.xlabel('Pressure-adjusted temperature (K)')
plt.ylabel('Rate Factor (MPa-3 y-1)')
plt.title('Rate Factor as a function of Tpa')
plt.grid(True)

# Loop over omega values
for omega in omega_values:
    # Calculate rate_factor as function of Tpa and omega
    rate_factor = arrhenius_from_temp_tf(Tpa_values, omega)

    # Add to the plot
    plt.plot(Tpa_values, rate_factor, label=f'omega={omega}')

plt.legend()
plt.show()

