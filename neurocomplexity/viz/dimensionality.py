"""Covariance eigenspectrum (scree) + cumulative variance with PR annotation."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from neurocomplexity.viz._palettes import DEFAULT_PALETTE, get_palette
from neurocomplexity.viz._style import _apply_panel_label, stats_box


def _sorted_eig(result):
    eig = np.asarray(result.eigenvalues, dtype=float)
    eig = np.sort(eig)[::-1]
    return eig[eig > 0]


def _draw_cumulative(ax, eig, pr, n_units, *, p):
    cum = np.cumsum(eig) / eig.sum()
    idx = np.arange(1, eig.size + 1)
    ax.plot(idx, cum, "-", lw=1.3, color=p["signal"])
    ax.axhline(0.9, ls="--", lw=0.7, color=p["muted"], label="90% var")
    ax.axvline(pr, ls="--", lw=1.1, color=p["accent"], label=f"PR = {pr:.1f}")
    ax.set_xlabel("Component index")
    ax.set_ylabel("Cumulative variance")
    ax.set_ylim(0, 1.04)
    ax.legend(loc="lower right", handlelength=1.6, borderpad=0.3)
    ratio = pr / max(n_units, 1)
    # Cumulative-variance panel rises from lower-left to upper-right →
    # top-LEFT corner is empty for the parameters box.
    annot = f"N = {n_units}\nPR = {pr:.1f}\nPR/N = {ratio:.2f}"
    if ratio > 0.85:
        annot += "\nsaturated"
    stats_box(ax, annot, corner="tl")


def figure_dimensionality(
    result,
    *,
    palette: str = DEFAULT_PALETTE,
    panel_label: str | None = None,
    figsize: tuple[float, float] | None = None,
    ax=None,
):
    """Plot the covariance eigenspectrum and participation ratio.

    Standalone call draws two panels — the eigenvalue scree (log-y) and the
    cumulative explained-variance curve with the participation ratio ``PR``
    marked. A "saturated" flag is shown when ``PR / n_units > 0.85``. When an
    ``ax`` is supplied only the cumulative-variance panel is drawn.
    """
    p = get_palette(palette)
    eig = _sorted_eig(result)
    composite = ax is not None

    if composite:
        fig = ax.figure
        _draw_cumulative(ax, eig, result.participation_ratio,
                         result.n_units, p=p)
        _apply_panel_label(ax, panel_label)
        return fig

    size = figsize if figsize is not None else (5.4, 2.6)
    fig, (ax_scree, ax_cum) = plt.subplots(1, 2, figsize=size)

    idx = np.arange(1, eig.size + 1)
    # No marker edge — for N~1000 eigenvalues the small dots in the bulk
    # would be erased by a white edge stroke.
    ax_scree.semilogy(idx, eig, "o-", ms=2.8, lw=0.9,
                      color=p["signal"], mec="none")
    # Noise-floor reference at the median eigenvalue: under a flat noise
    # spectrum the bulk sits here, so it shows how many components stand out.
    floor = float(np.median(eig))
    if floor > 0:
        ax_scree.axhline(floor, ls=":", lw=0.7, color=p["muted"],
                         label="median (noise floor)")
        # Legend bottom-left: scree decays from upper-left → lower-right,
        # so the TR corner is reserved for the parameters box.
        ax_scree.legend(loc="lower left", handlelength=1.6)
    ax_scree.set_xlabel("Component index")
    ax_scree.set_ylabel("Eigenvalue")
    # Scree decays from upper-left → lower-right; top-RIGHT empty.
    stats_box(ax_scree,
              f"N = {result.n_units}\n"
              f"$\\lambda_1$ = {eig[0]:.2g}\n"
              f"median = {float(np.median(eig)):.2g}",
              corner="tr")

    _draw_cumulative(ax_cum, eig, result.participation_ratio,
                     result.n_units, p=p)

    _apply_panel_label(ax_scree, panel_label)
    return fig
