import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from matplotlib import colors
from matplotlib.lines import Line2D
from ..utils.emulator_tools import compute_divflux


def to_numpy(tensor):
    """Convert TensorFlow tensor to numpy array."""
    if isinstance(tensor, tf.Tensor):
        return tensor.numpy()
    elif hasattr(tensor, 'numpy'):
        return tensor.numpy()
    else:
        return np.asarray(tensor)



def plot_temp(T_m_lowest, iter, cfg):
    smb_cfg = cfg.assimilations.smb_inference
    outdir = smb_cfg.output.outdir
    os.makedirs(os.path.join(outdir, "Figs"), exist_ok=True)

    n_steps = T_m_lowest.shape[0] if hasattr(T_m_lowest, 'shape') else len(T_m_lowest)
    time_axis = np.linspace(float(smb_cfg.t_start), float(smb_cfg.ttot), n_steps)
    fig, ax = plt.subplots(figsize=(21, 5))
    ax.plot(time_axis, to_numpy(T_m_lowest) - 13.762, label='Reconstructed temperature')
    ax.plot([1880], [-4.5], marker='*', markersize=14, color='red', label='Reference')
    ax.plot([1926], [-4.2], marker='*', markersize=14, color='red')
    ax.plot([1957], [-4.0], marker='*', markersize=14, color='red')
    ax.plot([1999], [-3.0], marker='*', markersize=14, color='red')

    ax.set(title="Temperature Series", xlabel="Time [years]", ylabel="Temperature (°C)")
    ax.grid(True)
    ax.legend()
    ax.set_ylim([-8.0,-2.8])
    fig.tight_layout()
    fig.savefig(os.path.join(outdir,f"Figs/temp_{iter}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# Visualize glacier surface and thickness
def visualize(Z_surf, time, H_ice, Lx, Ly):
    plt.figure(2, figsize=(11, 4), dpi=200)
    # Convert tensors to float32 for NumPy compatibility
    Z_surf_np = to_numpy(tf.cast(Z_surf, tf.float32))
    H_ice_np = to_numpy(tf.cast(H_ice, tf.float32))

    # First subplot: Ice surface
    plt.subplot(1, 2, 1)
    plt.imshow(Z_surf_np, extent=[0, Lx / 1000, 0, Ly / 1000], cmap='terrain', origin='lower')
    plt.colorbar(label='Elevation (m)')
    plt.title('Ice Surface at ' + str(int(time)) + ' y')
    plt.xlabel('Distance, km')
    plt.ylabel('Distance, km')

    # Second subplot: Ice thickness
    plt.subplot(1, 2, 2)
    plt.imshow(np.where(H_ice_np > 0, H_ice_np, np.nan), extent=[0, Lx/1000, 0, Ly/1000], cmap='jet', origin='lower')
    plt.colorbar(label='Ice Thickness (m)')
    plt.title('Ice Thickness at ' + str(int(time)) + ' y')
    plt.xlabel('Distance, km')
    plt.ylabel('Distance, km')
    # Display the plot briefly, then close
    # plt.tight_layout()
    # plt.pause(2)
    # plt.close()
    plt.show()

# Plot loss components
def plot_loss_components(total_loss_history, data_fidelity_history, outdir):
    plt.figure(figsize=(10, 6))
    plt.plot(total_loss_history, label='Total Loss', color='b', linewidth=2)
    plt.plot(data_fidelity_history, label='Data Fidelity', color='g', linestyle='--', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss Value')
    plt.title('Loss Function Components Over Iterations')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "loss_curve.png"))


def plot_loss_and_precipitation(data_fidelity_history, Precip_history, name, true_precip=1.5):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))  # Two subplots side by side

    # First subplot: Loss components
    axs[0].plot(data_fidelity_history, label='Data Fidelity', color='g', linestyle='--', linewidth=2)
    axs[0].set_title('Loss Components Over Iterations')
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('Loss Value')
    axs[0].legend()
    axs[0].grid(True)

    # Second subplot: Precipitation evolution
    axs[1].plot(Precip_history, label='Estimated Precipitation', color='b', linewidth=2)
    axs[1].axhline(y=true_precip, color='k', linestyle='--', label='True Precipitation')
    axs[1].set_title('Precipitation Evolution Over Iterations')
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('Precipitation (m/yr)')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(name)
    plt.show()

def plot_loss_and_temperature(data_fidelity_history, Temp_history, name, true_temp=7.0):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))  # Two subplots side by side

    # First subplot: Loss components
    axs[0].plot(data_fidelity_history, label='Data Fidelity', color='g', linestyle='--', linewidth=2)
    axs[0].set_title('Loss Components Over Iterations')
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('Loss Value')
    axs[0].legend()
    axs[0].grid(True)

    # Second subplot: Temperature evolution
    axs[1].plot(Temp_history, label='Estimated Temperature', color='r', linewidth=2)
    axs[1].axhline(y=true_temp, color='k', linestyle='--', label='True Temperature')
    axs[1].set_title('Temperature Evolution Over Iterations')
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('Temperature (°C)')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(name)
    plt.show()


def plot_gradient_evolution(total_gradients_history,name):
    plt.figure(figsize=(10,6))
# plt.plot(ELA_evolution,label="evolution of ELA",color='b')
    plt.plot(total_gradients_history, label='Evolution of gradients')
    plt.xlabel('Iteration')
    plt.legend()
    plt.savefig(name)

def plot_resulting_ELA(Z_ELA, H_simulated, observed_thk):
    # Convert tensors to NumPy arrays for plotting
    Z_ELA_np = to_numpy(tf.cast(Z_ELA, tf.float32))
    H_ice_np = to_numpy(tf.cast(H_simulated, tf.float32))
    observed_thk_np = to_numpy(tf.cast(observed_thk, tf.float32))

    # Create a figure with three subplots side by side
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))  # Adjust figsize for better layout

    # Plot the ELA field
    im1 = ax[0].imshow(Z_ELA_np, cmap='terrain', origin='lower')
    fig.colorbar(im1, ax=ax[0], orientation='vertical', label='Elevation (m)')
    ax[0].set_title('Reconstructed ELA Field')
    ax[0].set_xlabel('Distance, km')
    ax[0].set_ylabel('Distance, km')

    # Second subplot: Ice thickness (simulated)
    im2 = ax[1].imshow(np.where(H_ice_np > 0, H_ice_np, np.nan), cmap='jet', origin='lower')
    fig.colorbar(im2, ax=ax[1], orientation='vertical', label='Ice Thickness (m)')
    ax[1].set_title('Simulated Ice Thickness')
    ax[1].set_xlabel('Distance, km')

    # Third subplot: Observed ice thickness
    im3 = ax[2].imshow(np.where(observed_thk_np > 0, observed_thk_np, np.nan), cmap='jet', origin='lower')
    fig.colorbar(im3, ax=ax[2], orientation='vertical', label='Ice Thickness (m)')
    ax[2].set_title('Observed Ice Thickness')
    ax[2].set_xlabel('Distance, km')

    # Add a main title
    fig.suptitle('Glacier Evolution Analysis', fontsize=16)

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.95])
def plot_loss_topography_griddata(df, true_P=None, true_T=None, levels=5, savepath=None):
    """
    Plot a topographic map of the loss function using griddata interpolation,
    with contour lines for key minima.
    """
    # Validate input
    if not {'P', 'T', 'loss'}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'P', 'T', and 'loss' columns")

    # Extract raw data
    points = df[['P', 'T']].values
    loss_vals = df['loss'].values

    # Grid definition
    P_vals = df['P']
    T_vals = df['T']
    P_grid = np.linspace(P_vals.min(), P_vals.max(), 200)
    T_grid = np.linspace(T_vals.min(), T_vals.max(), 200)
    P_mesh, T_mesh = np.meshgrid(P_grid, T_grid)

    # Interpolate loss values on the grid
    Z = griddata(points, loss_vals, (P_mesh, T_mesh), method='linear')

    # Plot filled contour
    plt.figure(figsize=(8, 6))
    contourf = plt.contourf(P_mesh, T_mesh, Z, levels=levels, cmap='viridis')
    cbar = plt.colorbar(contourf)
    cbar.set_label("Loss")

    # Overlay fine contour lines at key minima levels
    contour_lines = plt.contour(P_mesh, T_mesh, Z, levels=[0.001, 0.002],
                                colors='white', linewidths=0.5, linestyles='--')
    plt.clabel(contour_lines, fmt='%.3f', inline=True, fontsize=8)

    # Scatter sampled points
    plt.scatter(P_vals, T_vals, s=5, c='black')

    # Optional: overlay true point
    if true_P is not None and true_T is not None:
        plt.plot(true_P, true_T, 'r*', markersize=15, label=f'True T={true_T}, P={true_P}')
        plt.legend()
    #plot stars for 2 examples
    plt.plot(0.842, 4.584, marker='*', color='blue', markersize=12,
         label='low temp low precip')
    plt.legend()

    plt.plot(2.567, 8.931, marker='*', color='green', markersize=12,
         label='high temp high precip')
    plt.legend()


    plt.xlabel("Precipitation (P)")
    plt.ylabel("Temperature (T)")
    plt.title("Topographic Map of Loss Function")
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=300)
    plt.show()


def visualize_velocities(ubar, vbar, H_ice, smb, time, dx=100, dy=100):
    plt.figure(2, figsize=(10, 6), dpi=200)

    # ---- tensors -> numpy ----
    H_ice_np = to_numpy(tf.cast(H_ice, tf.float32))
    u_np = to_numpy(tf.cast(ubar, tf.float32))
    v_np = to_numpy(tf.cast(vbar, tf.float32))
    smb_np = to_numpy(tf.cast(smb, tf.float32))

    # Clamp negatives to -5; positives unchanged
    smb_plot = np.maximum(smb_np, -15.0)

    ny, nx = H_ice_np.shape
    extent = [0, nx*dx, 0, ny*dy]
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    X, Y = np.meshgrid(x, y)

    # --- Subplot 1: Ice Thickness ---
    ax1 = plt.subplot(1, 3, 1)
    im1 = ax1.imshow(np.where(H_ice_np > 0, H_ice_np, np.nan),
                     cmap='jet', origin='lower', extent=extent)
    cax1 = make_axes_locatable(ax1).append_axes("right", size="3%", pad=0.05)
    plt.colorbar(im1, cax=cax1, label="Ice Thickness (m)")
    ax1.set_title(f"Ice Thickness at {int(time)} y")
    ax1.set_xlabel("x (m)"); ax1.set_ylabel("y (m)")

    # --- Subplot 2: SMB (clamped) + zero contour ---
    ax2 = plt.subplot(1, 3, 2)
    im2 = ax2.imshow(smb_plot, cmap='plasma', origin='lower', extent=extent, vmin=-15)
    cax2 = make_axes_locatable(ax2).append_axes("right", size="3%", pad=0.05)
    plt.colorbar(im2, cax=cax2, label="SMB")

    # zero-SMB contour (equilibrium line)
    ax2.contour(X, Y, smb_np, levels=[0.0], colors='blue', linewidths=1.5)
    ax2.set_title(f"SMB (≤-5 clamped) at {int(time)} y")
    ax2.set_xlabel("x (m)"); ax2.set_ylabel("y (m)")

    # --- Subplot 3: Velocity Vectors ---
    ax3 = plt.subplot(1, 3, 3)
    ax3.imshow(np.where(H_ice_np > 0, H_ice_np, np.nan),
               cmap="Greys", origin="lower", alpha=0.6, extent=extent)
    step = max(1, min(ny, nx) // 30)
    x_coords = np.arange(0, nx, step) * dx
    y_coords = np.arange(0, ny, step) * dy
    ax3.quiver(x_coords, y_coords,
               u_np[::step, ::step], v_np[::step, ::step],
               color="red", scale_units="xy", scale=1, width=0.002)
    ax3.set_title(f"Velocity Field at {int(time)} y")
    ax3.set_xlabel("x (m)"); ax3.set_ylabel("y (m)")

    plt.tight_layout()
    plt.show()


def plot_thickness_divflux_velocities(H_ice, ubar, vbar, dx=100, dy=100, time=None, dt=None):
    """
    Create a 3-panel plot showing:
    1. Ice thickness
    2. Flux divergence
    3. Velocity field

    Parameters
    ----------
    H_ice : tf.Tensor
        Ice thickness field (ny, nx)
    ubar : tf.Tensor
        x-component of velocity (ny, nx)
    vbar : tf.Tensor
        y-component of velocity (ny, nx)
    dx : float
        Grid spacing in x direction (m)
    dy : float
        Grid spacing in y direction (m)
    time : float, optional
        Time for display in title
    """

    # Convert tensors to numpy
    H_ice_np = to_numpy(tf.cast(H_ice, tf.float32))
    u_np = to_numpy(tf.cast(ubar, tf.float32))
    v_np = to_numpy(tf.cast(vbar, tf.float32))

    # Compute flux divergence
    divflux = compute_divflux(ubar, vbar, H_ice, dx, dy)
    divflux_np = to_numpy(tf.cast(divflux, tf.float32))

    ny, nx = H_ice_np.shape
    extent = [0, nx*dx/1000, 0, ny*dy/1000]  # Convert to km
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    X, Y = np.meshgrid(x, y)

    # Create figure
    fig = plt.figure(figsize=(15, 5), dpi=150)

    # --- Subplot 1: Ice Thickness ---
    ax1 = plt.subplot(1, 3, 1)
    im1 = ax1.imshow(np.where(H_ice_np > 0, H_ice_np, np.nan),
                     cmap='viridis', origin='lower', extent=extent)
    cax1 = make_axes_locatable(ax1).append_axes("right", size="3%", pad=0.05)
    plt.colorbar(im1, cax=cax1, label="Ice Thickness (m)")
    title_str = "Ice Thickness"
    if time is not None:
        title_str += f" at {int(time)} y"
    ax1.set_title(title_str)
    ax1.set_xlabel("x (km)")
    ax1.set_ylabel("y (km)")

    # --- Subplot 2: Flux Divergence ---
    ax2 = plt.subplot(1, 3, 2)
    # Mask where ice thickness is zero
    divflux_masked = np.where(H_ice_np > 0, divflux_np, np.nan)
    # Use diverging colormap centered at zero
    vmax = 15 #np.nanmax(np.abs(divflux_masked))
    im2 = ax2.imshow(divflux_masked, cmap='RdBu_r', origin='lower',
                     extent=extent, vmin=-vmax, vmax=vmax)
    cax2 = make_axes_locatable(ax2).append_axes("right", size="3%", pad=0.05)
    plt.colorbar(im2, cax=cax2, label="Flux Divergence (m/yr)")
    title_str = "Flux Divergence"
    if time is not None:
        title_str += f" at {int(time)} y"
    ax2.set_title(title_str)
    ax2.set_xlabel("x (km)")
    ax2.set_ylabel("y (km)")

    # --- Subplot 3: Velocity Magnitude ---
    ax3 = plt.subplot(1, 3, 3)

    # Compute velocity magnitude
    vel_mag = np.sqrt(u_np**2 + v_np**2)

    # Mask where ice thickness is zero
    vel_mag_masked = np.where(H_ice_np > 0, vel_mag, np.nan)

    # Plot velocity magnitude
    im3 = ax3.imshow(vel_mag_masked, cmap='plasma', origin='lower', extent=extent)
    cax3 = make_axes_locatable(ax3).append_axes("right", size="3%", pad=0.05)
    plt.colorbar(im3, cax=cax3, label="Velocity Magnitude (m/yr)")

    title_str = "Velocity Magnitude"
    if time is not None:
        title_str += f" at {int(time)} y"
    ax3.set_title(title_str)
    ax3.set_xlabel("x (km)")
    ax3.set_ylabel("y (km)")

    plt.tight_layout()
    plt.savefig(f"velocity_divflux_{int(time)}.png", dpi=150)

    return fig

def plot_sim_obs_extents(
    sim_list,                    # list of 7 arrays (simulations)
    obs_list,                    # list/tuple of 7 arrays (observations)
    cfg,
    iter,
    years=(1880, 1926, 1957, 1980, 1999, 2009, 2017),
    thresh=0.5,
    obs_is_mask=False,
    sim_is_mask=False,
):
    """
    For each year/panel:
      - Draw ONLY the outline (extent) of sim in blue and obs in red.
      - Extent is defined by values > thresh unless *_is_mask=True.
      - 7 panels with titles: 1880, 1926, 1957, 1980, 1999, 2009, 2017.
    """

    # Normalise inputs
    sim_list = [to_numpy(s) for s in sim_list]
    # If observations were provided as 7 separate arrays, accept tuple as well
    if not isinstance(obs_list, (list, tuple)):
        raise ValueError("obs_list must be a list or tuple of 7 observation arrays.")
    obs_list = [to_numpy(o) for o in obs_list]

    if len(sim_list) != 7 or len(obs_list) != 7:
        raise ValueError(f"Expecting 7 simulations and 7 observations. Got {len(sim_list)} and {len(obs_list)}.")

    if len(years) != 7:
        raise ValueError(f"'years' must have length 7. Got {len(years)}.")
    # Prepare color normalization for the diff map (white at 0)
    vmin, vmax = -200.0, 200.0
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap('seismic')  # negative -> blue, positive -> red (works with diff = obs - sim)


    # Create figure: one row with 7 panels
    fig, axes = plt.subplots(1, 7, figsize=(21, 5), constrained_layout=True)
    im_last = None  # will hold the last imshow handle for the colorbar

    for ax, sim, obs, yr in zip(axes, sim_list, obs_list, years):
        # Build binary masks for "extent"
        sim_mask = sim if sim_is_mask else (sim > thresh)
        obs_mask = obs if obs_is_mask else (obs > thresh)
        sim_mask = sim_mask.astype(bool)
        obs_mask = obs_mask.astype(bool)

        # Compute diff as obs - sim so: red => obs thicker, blue => sim thicker
        diff = obs - sim
        std=np.std(diff)
        # Clamp to requested range (for both visualization and values)
        np.clip(diff, vmin, vmax, out=diff)

        im = ax.imshow(
                diff,
                cmap=cmap,
                norm=norm,
                origin='lower',
                interpolation='nearest',
                alpha=0.5,
                zorder=0,
            )
        im_last = im  # keep latest handle; final loop's ax will get the colorbar


        # Plot ONLY outlines via a contour at 0.5
        if sim_mask.any():
            ax.contour(sim_mask.astype(float), levels=[0.5], colors='blue', linewidths=1.5, origin='lower')
        if obs_mask.any():
            ax.contour(obs_mask.astype(float), levels=[0.5], colors='red', linewidths=1.5, origin='lower')

        # Cosmetics
        ax.set_title(f'{str(yr)} ({int(std)} m)')
        ax.set_axis_off()

    # Only add colorbar on the last axis (if we plotted a diff)
    if im_last is not None:
        cbar = fig.colorbar(im_last, ax=axes[-1], fraction=0.06, pad=0.04)
        cbar.set_label("Obs − Sim thickness [m]")

    handles = [
        Line2D([0], [0], color='blue', lw=2, label='Simulation extent'),
        Line2D([0], [0], color='red',  lw=2, label='Observation extent')
    ]
    # Place legend above the panels
    fig.legend(handles=handles, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.05))

    fig.savefig(os.path.join(cfg.assimilations.smb_inference.output.outdir,
                            f"Figs/extent_{iter}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
