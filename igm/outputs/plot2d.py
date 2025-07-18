#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib 
import tensorflow as tf

from igm.utils.math.getmag import getmag 

def initialize(cfg, state):
    state.extent = [np.min(state.x), np.max(state.x), np.min(state.y), np.max(state.y)]

    if cfg.outputs.plot2d.editor == "vs":
        plt.ion()  # enable interactive mode

    state.fig = plt.figure(dpi=200)
    state.ax = state.fig.add_subplot(1, 1, 1)
    state.ax.axis("off")
    state.ax.set_aspect("equal")

def run(cfg, state):
    if state.saveresult:

        if cfg.outputs.plot2d.var == "velbar_mag":
            state.velbar_mag = getmag(state.ubar, state.vbar)

        im0 = state.ax.imshow(
            state.topg,
            origin="lower",
            cmap='binary', # matplotlib.cm.terrain,
            extent=state.extent
#            alpha=0.65,
        )
 
        if cfg.outputs.plot2d.var=="velbar_mag":
            im = state.ax.imshow(
                np.where(state.thk > 0, vars(state)[cfg.outputs.plot2d.var], np.nan),
                origin="lower",
                cmap="turbo",
                extent=state.extent, 
                norm=matplotlib.colors.LogNorm(vmin=1, vmax=cfg.outputs.plot2d.var_max)
            )
        else:
            im = state.ax.imshow(
                np.where(state.thk > 0, vars(state)[cfg.outputs.plot2d.var], np.nan),
                origin="lower",
                cmap='jet',
                vmin=0,
                vmax=cfg.outputs.plot2d.var_max,
                extent=state.extent,
            )
        if cfg.outputs.plot2d.particles:
            if hasattr(state, "particle"):
                if hasattr(state, "ip"):
                    state.ip.set_visible(False)
                r = 1
                state.ip = state.ax.scatter(
                    x = state.particle["x"][::r] + state.x[0],
                    y = state.particle["y"][::r] + state.y[0],
                    c = 1 - state.particle["r"][::r].numpy(), #or r ?
                    vmin=0,
                    vmax=1,
                    s=0.5,
                    cmap="RdBu",
                )
        state.ax.set_title("YEAR : " + str(getattr(state, 't', tf.constant(0)).numpy()), size=15)

        if not hasattr(state, "already_set_cbar"):
            state.cbar = plt.colorbar(im, label=cfg.outputs.plot2d.var)
            state.already_set_cbar = True

        if cfg.outputs.plot2d.live:
            if cfg.outputs.plot2d.editor == "vs":
                state.fig.canvas.draw()  # re-drawing the figure
                state.fig.canvas.flush_events()  # to flush the GUI events
            else:
                from IPython.display import display, clear_output

                clear_output(wait=True)
                display(state.fig)
        else:
            plt.savefig(
                cfg.outputs.plot2d.var + "-" + str(getattr(state, 't', tf.constant(0)).numpy()).zfill(4) + ".png",
                bbox_inches="tight",
                pad_inches=0.2,
            )


def finalize(params, state):
    pass
