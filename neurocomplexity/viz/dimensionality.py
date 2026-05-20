"""Dimensionality: eigenvalue scree + cumulative variance with PR annotation."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from neurocomplexity.viz._palettes import get_palette, DEFAULT_PALETTE
from neurocomplexity.viz._style import _apply_panel_label


def _sorted_eig(result):
    eig = np.asarray(result.eigenvalues, dtype=float)
    eig = np.sort(eig)[::-1]
    return eig[eig > 0]


def _draw_cumulative(ax, eig, pr, n_units, *, p):
    cum = np.cumsum(eig) / eig.sum()
    idx = np.arange(1, eig.size + 1)
    ax.plot(idx, cum, "-", lw=1.2, color=p["signal"])
    ax.axhline(0.9, ls="--", lw=0.7, color=p["muted"])
    ax.axvline(pr, ls="--", lw=1.0, color=p["accent"], label=f"PR={pr:.1f}")
    ax.set_xlabel("Component index")
    ax.set_ylabel("Cumulative variance")
    ax.set_ylim(0, 1.04)
    ax.legend(loc="lower right", handlelength=1.6, borderpad=0.3)
    ax.text(0.02, 0.97,
            f"N units = {n_units}\nPR/N = {pr / n_units:.2f}",
            transform=ax.transAxes, va="top", fontsize=6, color=p["text"])


def figure_dimensionality(
    result,
    *,
    palette: str = DEFAULT_PALETTE,
    panel_label: str | None = None,
    figsize: tuple[float, float] | None = None,
    ax=None,
):
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
    ax_scree.semilogy(idx, eig, "o-", ms=3, lw=1.0,
                      color=p["signal"], mec="none")
    ax_scree.set_xlabel("Component index")
    ax_scree.set_ylabel("Eigenvalue")

    _draw_cumulative(ax_cum, eig, result.participation_ratio,
                     result.n_units, p=p)

    _apply_panel_label(ax_scree, panel_label)
    return fig
