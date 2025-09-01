import numpy as np
import matplotlib.pyplot as plt

def drainageFunc(omega):
    # References: Greve (1997, application), Aschwanden (2012): p450
    # omega is dimensionless, eturn  [y-1]

    A = omega <= 0.01
    B = (omega > 0.01) & (omega <= 0.02)
    C = (omega > 0.02) & (omega <= 0.03)
    D = omega > 0.03

    return np.where(
        A,
        0.0 * omega,
        np.where(B, 0.5 * omega - 0.005, np.where(C, 4.5 * omega - 0.085, 0.05)),
    )

# Generate synthetic data for Omega
Omega_values = np.linspace(0, 0.04, 100)  # Adjust as needed

# Calculate drainage as function of omega
drainage = drainageFunc(Omega_values)

# Create the plot
plt.figure(figsize=(10, 7))
plt.plot(Omega_values,drainage)
plt.ylim([0, 0.06])  # Set y-axis limits
plt.xlim([0, 0.04]) # Set x-axis limits
plt.xlabel('Water content (omega)')
plt.ylabel('Drainage rate (yr-1)')
plt.title('Drainage rate as function of water content')
plt.grid(True)
plt.show()






