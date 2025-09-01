import numpy as np
import matplotlib.pyplot as plt

def compute_slidingco_tf(
    thk,
    tillwat,
    ice_density,
    gravity_cst,
    tillwatmax,
    phi,
    exp_weertman,
    uthreshold, 
    tauc_min,
    tauc_max
):
    # return the sliding coefficient in [m MPa^{-3} y^{-1}]

    e0 = 0.69  # void ratio at reference
    Cc = 0.12  # till compressibility coefficient
    delta = 0.02
    N0 = 1000  # [Pa] reference effective pressure

    s = tillwat / tillwatmax  # []

    P = ice_density * gravity_cst * thk  # [Pa]

    effpress = np.minimum(
        P, N0 * ((delta * P / N0) ** s) * 10 ** (e0 * (1 - s) / Cc)
    )  # [Pa]

    tauc = effpress * np.tan(phi * np.pi / 180)  # [Pa]

    tauc = np.where(thk > 0, tauc, 10**6)  # high value if ice-fre

    tauc = np.clip(tauc, tauc_min, tauc_max)
 
    slidingco = (tauc * 10 ** (-6)) * uthreshold ** (-1.0 / exp_weertman)  # Mpa m^(-1/3) y^(1/3) 

    return tauc,slidingco

# Define specific values for fixed variables

tillwatmax=2.0
phi=30
exp_weertman=4.0
uthreshold=100 
tauc_min=1.0e4
tauc_max=1.0e10
gravity_cst=9.81
ice_density=910


# Generate synthetic data for ice thickness
thk = np.linspace(20, 2000, 200)  # Adjust as needed

# define tillwat values to loop over (hence creating different curves in plot)
tillwat_values = [2]

# Loop over tillwat values
for tillwat in tillwat_values:
    # Calculate sliding_co values
    tauc, sliding_co = compute_slidingco_tf(
        thk,
        tillwat,
        ice_density,
        gravity_cst,
        tillwatmax,
        phi,
        exp_weertman,
        uthreshold, 
        tauc_min,
        tauc_max
    ) 

    # Plot sliding coefficient
    plt.subplot(2, 1, 1)
    plt.plot(thk, sliding_co, label=f'tillwat={tillwat}')
    plt.xlabel('Ice thickness (m)')
    plt.ylabel('Sliding coefficient')
    plt.title('Sliding coefficient as function of ice thickness')
    plt.legend()
    plt.grid(True)

    # Plot tauc
    plt.subplot(2, 1, 2)
    plt.plot(thk, tauc, label=f'tillwat={tillwat}')
    plt.xlabel('Ice thickness (m)')
    plt.ylabel('Tauc (Pa)')
    plt.title('Tauc as function of ice thickness')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()






