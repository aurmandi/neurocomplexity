"""Avalanche size & lifetime distributions with power-law fits.

Two-panel figure: log-binned P(s) and P(t) on log-log axes with the fitted
exponents (alpha_s, alpha_t) overlaid as dashed reference lines.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from neurocomplexity.viz._style import PALETTE, panel_label


def _log_pdf(values, nbins=30):
    v = np.asarray(values, dtype=float)
    v = v[v > 0]
    if v.size == 0:
        return np.array([]), np.array([])
    edges = np.logspace(np.log10(v.min()), np.log10(v.max()), nbins + 1)
    h, _ = np.histogram(v, bins=edges)
    widths = np.diff(edges)
    centers = 0.5 * (edges[1:] + edges[:-1])
    pdf = h / (h.sum() * widths)
    mask = pdf > 0
    return centers[mask], pdf[mask]


def figure_criticality(result, *, fig=None):
    """Render P(s), P(t) with fitted power laws.

    Parameters
    ----------
    result : CriticalityResult
    fig    : optional pre-existing Figure (for composite layouts)
    """
    if fig is None:
        fig, axes = plt.subplots(1, 2, figsize=(4.4, 2.1))
    else:
        axes = fig.subplots(1, 2)
    ax_s, ax_t = axes

    # Sizes
    xs, ps = _log_pdf(result.sizes)
    ax_s.loglog(xs, ps, "o", ms=3, color=PALETTE["signal"],
                mec="none", alpha=0.85, label="data")
    if xs.size:
        # Anchor reference line to the first decade
        x0 = xs[0]; y0 = ps[0]
        xx = np.array([xs.min(), xs.max()])
        yy = y0 * (xx / x0) ** (-result.alpha_s)
        ax_s.loglog(xx, yy, "--", lw=0.9, color=PALETTE["accent"],
                    label=fr"$\alpha_s={result.alpha_s:.2f}$")
    ax_s.set_xlabel("Avalanche size $s$")
    ax_s.set_ylabel("$P(s)$")
    ax_s.legend(loc="lower left")

    # Lifetimes
    xt, pt = _log_pdf(result.lifetimes)
    ax_t.loglog(xt, pt, "o", ms=3, color=PALETTE["ok"],
                mec="none", alpha=0.85, label="data")
    if xt.size:
        x0 = xt[0]; y0 = pt[0]
        xx = np.array([xt.min(), xt.max()])
        yy = y0 * (xx / x0) ** (-result.alpha_t)
        ax_t.loglog(xx, yy, "--", lw=0.9, color=PALETTE["accent"],
                    label=fr"$\alpha_t={result.alpha_t:.2f}$")
    ax_t.set_xlabel("Lifetime $T$ (s)")
    ax_t.set_ylabel("$P(T)$")
    ax_t.legend(loc="lower left")

    # Annotate R^2 in top-right corner of the size panel (compactness > captions)
    ax_s.text(0.98, 0.97, f"$R^2={result.r_squared:.2f}$\nbin={result.optimal_bin_seconds*1e3:.1f} ms",
              transform=ax_s.transAxes, ha="right", va="top", fontsize=6,
              color=PALETTE["muted"])

    panel_label(ax_s, "a")
    panel_label(ax_t, "b")
    fig.tight_layout()
    return fig
