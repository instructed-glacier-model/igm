#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import matplotlib.pyplot as plt
import numpy as np

from .config import load_test_config


def plot_comparison(
    x_ref: np.ndarray,
    v_ref: np.ndarray,
    x_igm: np.ndarray,
    v_igm: np.ndarray,
    title: str,
    filename: str,
) -> None:
    """Plot comparison between reference and IGM results."""
    config = load_test_config()

    # Only create plot if enabled in config
    if not config["validation"].get("save_plots", True):
        return

    plt.figure()
    plt.plot(x_ref, v_ref, "-", linewidth=1.5, label="Reference")
    plt.plot(x_igm, v_igm, "--", linewidth=1.5, label="IGM")
    plt.title(title)
    plt.xlabel("Scaled position (-)")
    plt.ylabel("Surface velocity (m/a)")
    plt.xlim(0, 1)
    plt.ylim(bottom=0)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
