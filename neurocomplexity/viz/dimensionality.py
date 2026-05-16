"""Dimensionality figure: eigenvalue scree + participation ratio annotation."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from neurocomplexity.viz._style import PALETTE, panel_label


def figure_dimensionality(result, *, fig=None):
    if fig is None:
        fig, axes = plt.subplots(1, 2, figsize=(4.4, 2.1))
    else:
        axes = fig.subplots(1, 2)
    ax_scree, ax_cum = axes

    eig = np.asarray(result.eigenvalues, dtype=float)
    eig = np.sort(eig)[::-1]
    eig = eig[eig > 0]
    idx = np.arange(1, eig.size + 1)

    ax_scree.semilogy(idx, eig, "o-", ms=2.5, lw=0.8,
                      color=PALETTE["signal"], mec="none")
    ax_scree.set_xlabel("Component index")
    ax_scree.set_ylabel("Eigenvalue")

    cum = np.cumsum(eig) / eig.sum()
    ax_cum.plot(idx, cum, "-", lw=1.0, color=PALETTE["signal"])
    ax_cum.axhline(0.9, ls="--", lw=0.6, color=PALETTE["muted"])
    pr = result.participation_ratio
    ax_cum.axvline(pr, ls="--", lw=0.8, color=PALETTE["accent"],
                   label=f"PR={pr:.1f}")
    ax_cum.set_xlabel("Component index")
    ax_cum.set_ylabel("Cumulative variance")
    ax_cum.set_ylim(0, 1.02)
    ax_cum.legend(loc="lower right")
    ax_cum.text(0.02, 0.97,
                f"N units = {result.n_units}\nPR/N = {pr/result.n_units:.2f}",
                transform=ax_cum.transAxes, va="top", fontsize=6,
                color=PALETTE["muted"])

    panel_label(ax_scree, "a")
    panel_label(ax_cum, "b")
    fig.tight_layout()
    return fig
