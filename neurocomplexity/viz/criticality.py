"""Avalanche size & lifetime distributions with power-law fits."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from neurocomplexity.viz._palettes import get_palette, DEFAULT_PALETTE
from neurocomplexity.viz._style import _apply_panel_label


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


def figure_criticality(
    result,
    *,
    palette: str = DEFAULT_PALETTE,
    panel_label: str | None = None,
    figsize: tuple[float, float] | None = None,
    ax=None,
):
    """Render P(s), P(T) with fitted power laws."""
    p = get_palette(palette)
    if ax is None:
        size = figsize if figsize is not None else (4.4, 2.1)
        fig, axes = plt.subplots(1, 2, figsize=size)
    else:
        fig = ax.figure
        ax.set_axis_off()
        gs = ax.get_subplotspec().subgridspec(1, 2)
        axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]
    ax_s, ax_t = axes

    xs, ps = _log_pdf(result.sizes)
    ax_s.loglog(xs, ps, "o", ms=3, color=p["signal"], mec="none", alpha=0.85,
                label="data")
    if xs.size:
        x0, y0 = xs[0], ps[0]
        xx = np.array([xs.min(), xs.max()])
        yy = y0 * (xx / x0) ** (-result.alpha_s)
        ax_s.loglog(xx, yy, "--", lw=0.9, color=p["accent"],
                    label=fr"$\alpha_s={result.alpha_s:.2f}$")
    ax_s.set_xlabel("Avalanche size $s$")
    ax_s.set_ylabel("$P(s)$")
    ax_s.legend(loc="lower left")

    xt, pt = _log_pdf(result.lifetimes)
    ax_t.loglog(xt, pt, "o", ms=3, color=p["signal"], mec="none", alpha=0.85,
                label="data")
    if xt.size:
        x0, y0 = xt[0], pt[0]
        xx = np.array([xt.min(), xt.max()])
        yy = y0 * (xx / x0) ** (-result.alpha_t)
        ax_t.loglog(xx, yy, "--", lw=0.9, color=p["accent"],
                    label=fr"$\alpha_t={result.alpha_t:.2f}$")
    ax_t.set_xlabel("Lifetime $T$ (s)")
    ax_t.set_ylabel("$P(T)$")
    ax_t.legend(loc="lower left")

    ax_s.text(0.98, 0.97,
              f"$R^2={result.r_squared:.2f}$\n"
              f"bin={result.optimal_bin_seconds * 1e3:.1f} ms",
              transform=ax_s.transAxes, ha="right", va="top",
              fontsize=6, color=p["muted"])

    _apply_panel_label(ax_s, panel_label)
    fig.tight_layout()
    return fig
