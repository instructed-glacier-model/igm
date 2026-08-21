"""Temporal homogenization of point SMB (stake) observations.

The inferred SMB profile is a multi-year *mean* — the balance that carries the
geometry from the start year to the end year. Individual stakes are single-year
point balances, sampled in different years and at different elevations, so a raw
scatter mixes the elevation signal we want with the year-to-year (climate)
signal we don't.  This module separates the two with an additive model

    b(i, t) = alpha(z_i) + beta(t) + eps

estimated from the raw stakes alone (alternating least squares).  ``alpha(z)``
is the elevation trend, ``beta(t)`` the glacier-wide annual anomaly.  Subtracting
``beta`` homogenizes every stake to the mean over the *observed* years, which is
directly comparable to the profile.

Caveat — "from raw data only": ``beta`` only exists for sampled years, and is
centered so its mean over those years is zero.  The homogenized balances
therefore represent the mean over *observed* years, which is not necessarily the
profile's full window (e.g. Rhone stakes start in 2006, not 2000).  To homogenize
to the full window you must bring in an external annual series (e.g. GLAMOS) for
the missing years.  The reference period is reported in the plot label so the
comparison is not silently mislabeled.
"""
import csv

import numpy as np


def load_stakes(path, alt_col="Alt", smb_col="Annual_SMB (m w.e.)",
                year_col="Year", x_col="X", y_col="Y"):
    """Read a raw-stakes CSV into arrays (x, y, z, smb [m w.e./yr], year).

    ``Year`` is optional: CSVs without it (single-snapshot point balances such
    as Argentiere's) get a sentinel year 0, which collapses the temporal
    homogenization to a no-op and yields no repeat-cell clusters — callers then
    fall back to a raw-stake overlay (see :func:`draw_raw_stake_overlay`).
    """
    x, y, z, b, yr = [], [], [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            # Parse the required columns first, so a row is appended to every
            # array or to none (a mid-row failure used to leave z/b longer than
            # x/y, producing ragged, misaligned arrays).
            try:
                z_i = float(row[alt_col])
                b_i = float(row[smb_col])
                x_i = float(row[x_col])
                y_i = float(row[y_col])
            except (KeyError, ValueError):
                continue
            try:
                yr_i = int(float(row[year_col]))
            except (KeyError, ValueError):
                yr_i = 0
            z.append(z_i); b.append(b_i); x.append(x_i); y.append(y_i); yr.append(yr_i)
    return (np.asarray(x), np.asarray(y), np.asarray(z, float),
            np.asarray(b, float), np.asarray(yr, int))


def fit_additive_model(z, b, years, deg=2, n_iter=5):
    """Alternating least squares for ``b = alpha(z) + beta(t)``.

    Returns ``(coeffs, beta)`` where ``coeffs`` is the degree-``deg`` polynomial
    for the elevation trend and ``beta`` is a ``{year: anomaly}`` dict centered
    to zero mean over the observed years.  Iterating decouples the year effect
    from the elevation sampling of each year, so a year staked only at the
    terminus does not masquerade as a low-balance year.
    """
    uniq = np.unique(years)
    beta = {int(t): 0.0 for t in uniq}
    coeffs = np.polyfit(z, b, deg)
    for _ in range(n_iter):
        b_detr = b - np.array([beta[int(t)] for t in years])
        coeffs = np.polyfit(z, b_detr, deg)
        resid = b - np.polyval(coeffs, z)
        beta = {int(t): float(resid[years == t].mean()) for t in uniq}
        mean_beta = float(np.mean(list(beta.values())))
        beta = {t: v - mean_beta for t, v in beta.items()}
    return coeffs, beta


def homogenize(z, b, years, deg=2, n_iter=5):
    """Return ``(b_homogenized, coeffs, beta)``; b_homog = b - beta(t)."""
    coeffs, beta = fit_additive_model(z, b, years, deg=deg, n_iter=n_iter)
    b_hom = b - np.array([beta[int(t)] for t in years])
    return b_hom, coeffs, beta


def bin_by_elevation(z, b, z_edges):
    """Mean/std/count of ``b`` per elevation bin -> (centers, mean, std, n)."""
    idx = np.digitize(z, z_edges) - 1
    nb = len(z_edges) - 1
    centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    mean = np.full(nb, np.nan)
    std = np.full(nb, np.nan)
    n = np.zeros(nb, int)
    for k in range(nb):
        sel = idx == k
        n[k] = int(sel.sum())
        if n[k]:
            mean[k] = b[sel].mean()
            std[k] = b[sel].std()
    return centers, mean, std, n


def repeat_clusters(x, y, z, b, years, xy_km=1.0, dz_m=50.0, min_years=10):
    """Assumption-free 'continuous record' anchor.

    Group stakes by (``xy_km`` horizontal, ``dz_m`` vertical) cells and keep only
    cells observed in >= ``min_years`` distinct years.  Returns sorted-by-altitude
    arrays ``(alt, mean, std, n_years)`` computed from the RAW balances — these
    cells already average out the year effect by construction, so they need no
    homogenization and serve as a check on it.
    """
    cells = {}
    for i in range(len(z)):
        k = (round(x[i] / (xy_km * 1000.0)),
             round(y[i] / (xy_km * 1000.0)),
             round(z[i] / dz_m))
        cells.setdefault(k, []).append(i)
    alt, mean, std, nyr = [], [], [], []
    for idxs in cells.values():
        idxs = np.asarray(idxs)
        if np.unique(years[idxs]).size < min_years:
            continue
        alt.append(float(z[idxs].mean()))
        mean.append(float(b[idxs].mean()))
        std.append(float(b[idxs].std()))
        nyr.append(int(np.unique(years[idxs]).size))
    order = np.argsort(alt) if alt else np.array([], int)
    a = lambda v: np.asarray(v)[order]
    return a(alt), a(mean), a(std), a(nyr)


def prepare_stake_overlay(csv_path, bin_m=100.0, cluster_min_years=10,
                          cluster_xy_km=1.0, cluster_dz_m=50.0, deg=2):
    """Do the homogenization + binning + clustering once for a stakes CSV.

    Returns a dict consumed by :func:`draw_stake_overlay`, or ``None`` if the
    file has no usable rows.  All balances are in m w.e./yr.
    """
    x, y, z, b, yr = load_stakes(csv_path)
    if z.size == 0:
        return None
    b_hom, coeffs, beta = homogenize(z, b, yr, deg=deg)

    lo = np.floor(z.min() / bin_m) * bin_m
    hi = np.ceil(z.max() / bin_m) * bin_m
    z_edges = np.arange(lo, hi + bin_m, bin_m)
    centers, mean, std, n = bin_by_elevation(z, b_hom, z_edges)

    a_alt, a_mean, a_std, a_ny = repeat_clusters(
        x, y, z, b, yr, xy_km=cluster_xy_km, dz_m=cluster_dz_m,
        min_years=cluster_min_years)

    obs_years = sorted(beta)
    return {
        "raw_z": z, "raw_b": b,
        "bin_centers": centers, "bin_mean": mean, "bin_std": std, "bin_n": n,
        "clu_alt": a_alt, "clu_mean": a_mean, "clu_std": a_std, "clu_ny": a_ny,
        "beta": beta, "coeffs": coeffs,
        "period": f"{obs_years[0]}–{obs_years[-1]}",
        "cluster_min_years": cluster_min_years,
    }


def draw_stake_overlay(ax, prep, z_profile=None, smb_profile_we=None):
    """Faded raw stakes (the per-researcher point measurements GLAMOS ingests)
    plus the repeat-cell multi-year anchors. The blue GLAMOS *product*
    (elevation-band balance) dots are drawn separately by compare_smb."""
    ax.scatter(prep["raw_b"], prep["raw_z"], s=10, c="0.7", alpha=0.5,
               zorder=2, label="raw stakes (single year)")

    if prep["clu_alt"].size:
        ax.errorbar(prep["clu_mean"], prep["clu_alt"], xerr=prep["clu_std"],
                    fmt="D", color="tab:green", ecolor="tab:green",
                    elinewidth=1.2, capsize=3, markersize=6, zorder=6,
                    label=f"repeat cells (≥{prep['cluster_min_years']} yr)")


def draw_anchor_overlay(ax, prep, z_profile, smb_profile_we):
    """Minimal overlay (Kneib 2024 style): only the repeat-cell averages.

    Each anchor is a stake position observed in many years; its mean is the
    per-stake multi-year average and the (thin) bar is its interannual std.
    Both directly comparable to the inferred multi-year-mean profile.  Returns
    RMSE of the profile against the anchors.
    """
    if not prep["clu_alt"].size:
        return float("nan")
    ax.errorbar(prep["clu_mean"], prep["clu_alt"], xerr=prep["clu_std"],
                fmt="o", color="tab:green", ecolor="tab:green",
                elinewidth=0.8, capsize=2, markersize=5, zorder=6,
                label=f"stake bins (≥{prep['cluster_min_years']} yr, mean ± σ)")
    prof_at = np.interp(prep["clu_alt"], z_profile, smb_profile_we)
    rmse = float(np.sqrt(np.mean((prep["clu_mean"] - prof_at) ** 2)))
    ax.plot([], [], " ", label=f"RMSE={rmse:.3f} m w.e./yr ({prep['period']})")
    return rmse


def draw_raw_stake_overlay(ax, prep, z_profile, smb_profile_we):
    """Overlay each raw stake as a point + the profile-vs-stakes RMSE.

    For stake CSVs without a Year column (single-snapshot point balances, e.g.
    Argentiere), temporal homogenization and repeat-cell anchors don't apply, so
    the inferred profile is compared directly against the individual stakes —
    matching the original per-iter overlay (outputs/2026-04-29 reference).
    Returns the RMSE (m w.e./yr).
    """
    z, b = prep["raw_z"], prep["raw_b"]
    ax.scatter(b, z, s=18, c="tab:red", alpha=0.8, zorder=5,
               label="stakes (m w.e./yr)")
    prof_at = np.interp(z, z_profile, smb_profile_we)
    rmse = float(np.sqrt(np.mean((b - prof_at) ** 2)))
    ax.plot([], [], " ", label=f"profile vs stakes RMSE={rmse:.3f}")
    return rmse
