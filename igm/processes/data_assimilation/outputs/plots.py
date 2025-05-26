#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import os, copy
import matplotlib.pyplot as plt
import matplotlib 
import tensorflow as tf 
from igm.utils.math.getmag import getmag
 
def plot_cost_functions():

#    costs = np.stack(costs)

    file_path = 'costs.dat'

    # Read the file and process the contents
    with open(file_path, 'r') as file:
        lines = file.readlines()
        label = lines[0].strip().split()
        
#        print(lines)
#        print(lines[1:][0].strip().split())
        
        costs = [np.array(line.strip().split(), dtype=float) for line in lines[1:]]
        # costs = [np.array(line.strip().split(), dtype=float) for line in lines[1:]]

    costs = np.stack(costs)

    for i in range(costs.shape[1]):
        costs[:, i] -= np.min(costs[:, i])
        costs[:, i] /= np.where(np.max(costs[:, i]) == 0, 1.0, np.max(costs[:, i]))

    colors = ["k", "r", "b", "g", "c", "m", "k", "r", "b", "g", "c", "m"]
  
    fig = plt.figure(figsize=(10, 10))
    for i in range(costs.shape[1]):
        plt.plot(costs[:, i], label=label[i], c=colors[i])
    plt.ylim(0, 1)
    plt.legend()

    plt.savefig("convergence.png", pad_inches=0)
    plt.close("all")

def update_plot_inversion(cfg, state, i):
    """
    Plot thickness, velocity, mand slidingco"""

    if hasattr(state, "uvelsurfobs"):
        velsurfobs_mag = getmag(state.uvelsurfobs, state.vvelsurfobs).numpy()
    else:
        velsurfobs_mag = np.zeros_like(state.thk.numpy())

    if hasattr(state, "usurfobs"):
        usurfobs = state.usurfobs
    else:
        usurfobs = np.zeros_like(state.thk.numpy())

    velsurf_mag = getmag(state.uvelsurf, state.vvelsurf).numpy()

    #########################################################

    if i == 0:
        if cfg.processes.data_assimilation.output.editor_plot2d == "vs":
            plt.ion()  # enable interactive mode

        # state.fig = plt.figure()
        state.fig, state.axes = plt.subplots(2, 3,figsize=(10, 8))

        state.extent = [state.x[0], state.x[-1], state.y[0], state.y[-1]]

    #########################################################

    cmap = copy.copy(matplotlib.cm.jet)
    cmap.set_bad(color="white")

    ax1 = state.axes[0, 0]

    im1 = ax1.imshow(
        np.ma.masked_where(state.thk == 0, state.thk),
        origin="lower",
        extent=state.extent,
        vmin=0,
        vmax=np.quantile(state.thk, 0.999),
        cmap=cmap,
    )
    if i == 0:
        plt.colorbar(im1, ax=ax1)
    ax1.set_title(
        "Ice thickness \n (RMS : "
        + str(int(state.rmsthk[-1]))
        + ", STD : "
        + str(int(state.stdthk[-1]))
        + ")",
        size=12,
    )
    ax1.axis("off")

    #########################################################

    ax2 = state.axes[0, 1]

    from matplotlib import colors

    if "arrhenius" in cfg.processes.data_assimilation.control_list:

        im1 = ax2.imshow(
            np.ma.masked_where(state.thk == 0, state.arrhenius),
            origin="lower", 
            vmin=0,
            vmax=300,
            cmap=cmap, 
        )
        if i == 0:
            plt.colorbar(im1, format="%.2f", ax=ax2)
        ax2.set_title("Iteration " + str(i) + " \n Arrhenius coefficient", size=12)
        ax2.axis("off")

    else:

        im1 = ax2.imshow(
            np.ma.masked_where(state.thk == 0, state.slidingco),
            origin="lower",
    #        norm=colors.LogNorm(),
            vmin=0.01,
            vmax=0.10,
            cmap=cmap,
    #        tf.sqrt(state.slidingco/1.0e-6),
    #        vmin=100,
    #        vmax=500,
        )
        if i == 0:
            plt.colorbar(im1, format="%.2f", ax=ax2)
        ax2.set_title("Iteration " + str(i) + " \n Sliding coefficient", size=12)
        ax2.axis("off")

    ########################################################

    ax3 = state.axes[0, 2]

    im1 = ax3.imshow(
        state.usurf - usurfobs,
        origin="lower",
        extent=state.extent,
        vmin=-10,
        vmax=10,
        cmap="RdBu",
    )
    if i == 0:
        plt.colorbar(im1, format="%.2f", ax=ax3)
    ax3.set_title(
        "Top surface adjustement \n (RMS : %5.1f , STD : %5.1f"
        % (state.rmsusurf[-1], state.stdusurf[-1])
        + ")",
        size=12,
    )
    ax3.axis("off")

    #########################################################

    cmap = copy.copy(matplotlib.cm.viridis)
    cmap.set_bad(color="white")

    ax4 = state.axes[1, 0]

    im1 = ax4.imshow(
        np.ma.masked_where(state.thk == 0, velsurf_mag),
        origin="lower",
        extent=state.extent,
        norm=matplotlib.colors.LogNorm(vmin=1, vmax=5000),
        cmap=cmap,
    )
    if i == 0:
        plt.colorbar(im1, format="%.2f", ax=ax4)
    ax4.set_title(
        "Modelled velocities \n (RMS : "
        + str(int(state.rmsvel[-1]))
        + ", STD : "
        + str(int(state.stdvel[-1]))
        + ")",
        size=12,
    )
    ax4.axis("off")

    ########################################################

    ax5 = state.axes[1, 1]
    im1 = ax5.imshow(
        np.ma.masked_where(state.thk == 0, velsurfobs_mag),
        origin="lower",
        extent=state.extent,
        norm=matplotlib.colors.LogNorm(vmin=1, vmax=5000),
        cmap=cmap,
    )
    if i == 0:
        plt.colorbar(im1, format="%.2f", ax=ax5)
    ax5.set_title("Target \n Observed velocities", size=12)
    ax5.axis("off")

    #######################################################

    ax6 = state.axes[1, 2]
    im1 = ax6.imshow(
        state.divflux, # np.where(state.icemaskobs > 0.5, state.divflux,np.nan),
        origin="lower",
        extent=state.extent,
        vmin=-10,
        vmax=10,
        cmap="RdBu",
    )
    if i == 0:
        plt.colorbar(im1, format="%.2f", ax=ax6)
    ax6.set_title(
        "Flux divergence \n (RMS : %5.1f , STD : %5.1f"
        % (state.rmsdiv[-1], state.stddiv[-1])
        + ")",
        size=12,
    )
    ax6.axis("off")

    #########################################################

    if cfg.processes.data_assimilation.output.plot2d_live:
        if cfg.processes.data_assimilation.output.editor_plot2d == "vs":
            state.fig.canvas.draw()  # re-drawing the figure
            state.fig.canvas.flush_events()  # to flush the GUI events
        else:
            from IPython.display import display, clear_output

            clear_output(wait=True)
            display(state.fig)
    else:
        plt.savefig("resu-opti-" + str(i).zfill(4) + ".png", bbox_inches="tight", pad_inches=0.2)

def update_plot_inversion_simple(cfg, state, i):
    """
    Plot thickness, velocity, mand slidingco"""

    if hasattr(state, "uvelsurfobs"):
        velsurfobs_mag = getmag(state.uvelsurfobs, state.vvelsurfobs).numpy()
    else:
        velsurfobs_mag = np.zeros_like(state.thk.numpy())

    if hasattr(state, "usurfobs"):
        usurfobs = state.usurfobs
    else:
        usurfobs = np.zeros_like(state.thk.numpy())

    velsurf_mag = getmag(state.uvelsurf, state.vvelsurf).numpy()

    #########################################################

    if i == 0:
        if cfg.processes.data_assimilation.output.editor_plot2d == "vs":
            plt.ion()  # enable interactive mode

        # state.fig = plt.figure()
        state.fig, state.axes = plt.subplots(1, 2)

        state.extent = [state.x[0], state.x[-1], state.y[0], state.y[-1]]

    #########################################################

    cmap = copy.copy(matplotlib.cm.jet)
    cmap.set_bad(color="white")

    ax1 = state.axes[0]

    im1 = ax1.imshow(
        np.ma.masked_where(state.thk == 0, state.thk),
        origin="lower",
        extent=state.extent,
        vmin=0,
        vmax=np.quantile(state.thk, 0.98),
        cmap=cmap,
    )
    if i == 0:
        plt.colorbar(im1, ax=ax1)
    ax1.set_title(
        "Ice thickness \n (RMS : "
        + str(int(state.rmsthk[-1]))
        + ", STD : "
        + str(int(state.stdthk[-1]))
        + ")",
        size=16,
    )
    ax1.axis("off")

    #########################################################

    cmap = copy.copy(matplotlib.cm.viridis)
    cmap.set_bad(color="white")

    ax4 = state.axes[1]

    im1 = ax4.imshow(
        np.ma.masked_where(state.thk == 0, velsurf_mag),
        origin="lower",
        extent=state.extent,
        vmin=0,
        vmax=np.nanmax(velsurfobs_mag),
        cmap=cmap,
    )
    if i == 0:
        plt.colorbar(im1, format="%.2f", ax=ax4)
    ax4.set_title(
        "Modelled velocities \n (RMS : "
        + str(int(state.rmsvel[-1]))
        + ", STD : "
        + str(int(state.stdvel[-1]))
        + ")",
        size=16,
    )
    ax4.axis("off")

    #########################################################

    if cfg.processes.data_assimilation.output.plot2d_live:
        if cfg.processes.data_assimilation.output.editor_plot2d == "vs":
            state.fig.canvas.draw()  # re-drawing the figure
            state.fig.canvas.flush_events()  # to flush the GUI events
        else:
            from IPython.display import display, clear_output

            clear_output(wait=True)
            display(state.fig)
    else:
        plt.savefig(
            "resu-opti-" + str(i).zfill(4) + ".png",
            pad_inches=0,
        )
        plt.close("all")
















