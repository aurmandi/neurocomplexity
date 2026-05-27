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


def _draw_powerlaw_panel(ax, xs, ps, alpha, *, p, x_label, y_label, alpha_label):
    """Single P(x) panel with median-anchored power-law fit overlay."""
    ax.loglog(xs, ps, "o", ms=3.5, color=p["signal"], mec="none", alpha=0.9,
              label="data")
    if xs.size and np.isfinite(alpha):
        # Anchor by median offset in log space — visually centred line
        log_offset = float(np.median(np.log(ps) + alpha * np.log(xs)))
        xx = np.array([xs.min(), xs.max()])
        yy = np.exp(log_offset) * xx ** (-alpha)
        ax.loglog(xx, yy, "--", lw=1.1, color=p["accent"],
                  label=fr"${alpha_label}={alpha:.2f}$")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(loc="lower left", handlelength=1.6, borderpad=0.3)


def figure_criticality(
    result,
    *,
    palette: str = DEFAULT_PALETTE,
    panel_label: str | None = None,
    figsize: tuple[float, float] | None = None,
    ax=None,
):
    """Render P(s) + P(T) standalone, or P(s)-only when ``ax=`` is given."""
    p = get_palette(palette)
    composite = ax is not None

    if composite:
        # Compact single-panel mode when ax= is provided: P(s) only
        fig = ax.figure
        xs, ps = _log_pdf(result.sizes)
        _draw_powerlaw_panel(ax, xs, ps, result.alpha_s, p=p,
                             x_label="Avalanche size $s$",
                             y_label="$P(s)$",
                             alpha_label=r"\alpha_s")
        ax.text(0.98, 0.97,
                f"$R^2={result.r_squared:.2f}$  bin$={result.optimal_bin_seconds * 1e3:.1f}$ ms",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=6, color=p["text"])
        _apply_panel_label(ax, panel_label)
        return fig

    # Standalone two-panel figure
    size = figsize if figsize is not None else (5.4, 2.6)
    fig, (ax_s, ax_t) = plt.subplots(1, 2, figsize=size)

    xs, ps = _log_pdf(result.sizes)
    _draw_powerlaw_panel(ax_s, xs, ps, result.alpha_s, p=p,
                         x_label="Avalanche size $s$", y_label="$P(s)$",
                         alpha_label=r"\alpha_s")
    xt, pt = _log_pdf(result.lifetimes)
    _draw_powerlaw_panel(ax_t, xt, pt, result.alpha_t, p=p,
                         x_label="Lifetime $T$ (s)", y_label="$P(T)$",
                         alpha_label=r"\alpha_t")

    annot = (f"$R^2={result.r_squared:.2f}$\n"
             f"bin$={result.optimal_bin_seconds * 1e3:.1f}$ ms")
    ax_s.text(0.98, 0.97, annot, transform=ax_s.transAxes,
              ha="right", va="top", fontsize=6, color=p["text"])
    # Same annotation on the lifetime panel so each plot is self-describing.
    ax_t.text(0.98, 0.97, annot, transform=ax_t.transAxes,
              ha="right", va="top", fontsize=6, color=p["text"])

    _apply_panel_label(ax_s, panel_label)
    return fig
