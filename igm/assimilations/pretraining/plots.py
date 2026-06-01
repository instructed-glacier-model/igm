#!/usr/bin/env python3
# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_loss_plot(
    train_total_hist,
    val_total_hist,
    train_data_hist,
    val_data_hist,
    train_phys_hist,
    val_phys_hist,
    lambda_hist,
    fig_path: Path,
) -> None:
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(train_total_hist)
    epochs = np.arange(1, n + 1)

    train_total = np.asarray(train_total_hist, dtype=float)
    val_total   = np.asarray(val_total_hist, dtype=float)
    train_data  = np.asarray(train_data_hist, dtype=float)
    val_data    = np.asarray(val_data_hist, dtype=float)
    train_phys  = np.asarray(train_phys_hist, dtype=float)
    val_phys    = np.asarray(val_phys_hist, dtype=float)
    lam         = np.asarray(lambda_hist, dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(7, 8), sharex=True)
    ax0, ax1 = axes[0]
    ax2, ax3 = axes[1]

    # ---- subplot 1: total  ----
    ax0.plot(epochs, train_total, label="train_total")
    ax0.plot(epochs, val_total,   label="val_total")
    ax0.set_ylabel("loss (total)")
    min_total = np.min([train_total.min(), val_total.min()])
    if min_total > 0:
        ax0.set_yscale("log")
        ax0.grid(True, which="both", alpha=0.3)
    else:
        ax0.grid(True, which="major", alpha=0.3)
    ax0.legend(ncol=2, fontsize=8)

    # ---- subplot 2: physics ----
    ax1.plot(epochs, train_phys, label="train_phys", linestyle=":")
    ax1.plot(epochs, val_phys,   label="val_phys",   linestyle=":")
    ax1.set_ylabel("physics loss")
    min_phys = np.min([train_phys.min(), val_phys.min()])
    if min_phys > 0:
        ax1.set_yscale("log")
        ax1.grid(True, which="both", alpha=0.3)
    else:
        ax1.grid(True, which="major", alpha=0.3)
    ax1.legend(ncol=2, fontsize=8)

    # ---- subplot 3: data ----
    ax2.plot(epochs, train_data, label="train_data", linestyle="--")
    ax2.plot(epochs, val_data,   label="val_data",   linestyle="--")
    ax2.set_ylabel("data loss")
    ax2.set_yscale("log")
    ax2.grid(True, which="major", alpha=0.3)
    ax2.legend(ncol=2, fontsize=8)

    # ---- subplot 4: lambda_phys ----
    ax3.plot(epochs, lam, label="lambda_phys")
    ax3.set_xlabel("epoch")
    ax3.set_ylabel("lambda_phys")
    if np.all(lam > 0):
        ax3.set_yscale("log")
        ax3.grid(True, which="both", alpha=0.3)
    else:
        ax3.grid(True, which="major", alpha=0.3)
    ax3.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def save_speed_compare(mapping, x_b, y_b, Nz: int, fig_path: Path) -> None:
    """
    Saves a 3-panel image: true surface speed, predicted surface speed, difference.
    This needs to be redone at a later stage, right now it's really just a quick diagnostic for sanity checking the emulator predictions on a batch of training data
    """
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    # pick one random sample from the batch
    b = int(np.random.randint(0, x_b.shape[0]))
    k = Nz - 1  # surface level assumption

    # Forward (BCs applied to predictions via mapping)
    U_pred, V_pred = mapping.get_UV(x_b)
    U_pred = U_pred.numpy()
    V_pred = V_pred.numpy()

    # Targets
    U_true = y_b[..., 0].numpy()
    V_true = y_b[..., 1].numpy()

    sp_true = np.sqrt(U_true[b, k] ** 2 + V_true[b, k] ** 2)
    sp_pred = np.sqrt(U_pred[b, k] ** 2 + V_pred[b, k] ** 2)
    sp_diff = sp_pred - sp_true

    # shared scaling for true/pred (robust upper bound)
    vmax = np.nanpercentile(np.concatenate([sp_true.ravel(), sp_pred.ravel()]), 99)
    vmax = float(vmax) if np.isfinite(vmax) and vmax > 0 else None

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    im0 = axs[0].imshow(sp_true, origin="lower", vmin=0, vmax=vmax)
    axs[0].set_title("true surface speed")
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(sp_pred, origin="lower", vmin=0, vmax=vmax)
    axs[1].set_title("pred surface speed")
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    # difference plot: symmetric bounds
    dmax = np.nanpercentile(np.abs(sp_diff.ravel()), 99)
    dmax = float(dmax) if np.isfinite(dmax) and dmax > 0 else None
    im2 = axs[2].imshow(sp_diff, origin="lower", vmin=-dmax, vmax=dmax)
    axs[2].set_title("pred - true")
    plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
