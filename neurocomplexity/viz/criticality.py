"""Avalanche size P(s) & lifetime P(T) distributions with power-law fits."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from neurocomplexity.viz._palettes import DEFAULT_PALETTE, get_palette
from neurocomplexity.viz._style import _apply_panel_label, stats_box


def _log_pdf(values, nbins=30):
    """Logarithmically-binned probability density of positive ``values``."""
    v = np.asarray(values, dtype=float)
    v = v[v > 0]
    if v.size == 0:
        return np.array([]), np.array([])
    edges = np.logspace(np.log10(v.min()), np.log10(v.max()), nbins + 1)
    h, _ = np.histogram(v, bins=edges)
    widths = np.diff(edges)
    centers = np.sqrt(edges[1:] * edges[:-1])  # geometric centre (log binning)
    pdf = h / (h.sum() * widths)
    mask = pdf > 0
    return centers[mask], pdf[mask]


def _draw_powerlaw_panel(ax, xs, ps, alpha, *, p, x_label, y_label, exponent_sym):
    """One P(x) panel: empirical density + power-law guide line."""
    ax.loglog(xs, ps, "o", ms=4, color=p["signal"], mec="white", mew=0.4,
              alpha=0.95, zorder=3, label="data")
    if xs.size and np.isfinite(alpha):
        # Guide line anchored at the median log-offset so it threads the data
        # cloud rather than being pinned to a single extreme bin.
        log_offset = float(np.median(np.log(ps) + alpha * np.log(xs)))
        xx = np.array([xs.min(), xs.max()])
        yy = np.exp(log_offset) * xx ** (-alpha)
        ax.loglog(xx, yy, "--", lw=1.3, color=p["accent"], zorder=4,
                  label=fr"${exponent_sym} = {alpha:.2f}$")
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
    """Plot avalanche size and lifetime distributions with power-law fits.

    Standalone call draws two panels — size ``P(s)`` and lifetime ``P(T)``,
    each with its fitted exponent. When an ``ax`` is supplied the compact
    single-panel ``P(s)`` form is drawn instead (for composite figures).
    """
    p = get_palette(palette)
    composite = ax is not None

    if composite:
        fig = ax.figure
        xs, ps = _log_pdf(result.sizes)
        _draw_powerlaw_panel(ax, xs, ps, result.alpha_s, p=p,
                             x_label="Avalanche size $s$", y_label="$P(s)$",
                             exponent_sym=r"\tau")
        stats_box(ax,
                  f"$R^2$ = {result.r_squared:.2f}\n"
                  f"bin = {result.optimal_bin:.1f} ms",
                  corner="tr")
        _apply_panel_label(ax, panel_label)
        return fig

    size = figsize if figsize is not None else (5.4, 2.6)
    fig, (ax_s, ax_t) = plt.subplots(1, 2, figsize=size)

    xs, ps = _log_pdf(result.sizes)
    _draw_powerlaw_panel(ax_s, xs, ps, result.alpha_s, p=p,
                         x_label="Avalanche size $s$", y_label="$P(s)$",
                         exponent_sym=r"\tau")
    xt, pt = _log_pdf(result.lifetimes)
    _draw_powerlaw_panel(ax_t, xt, pt, result.alpha_t, p=p,
                         x_label="Lifetime $T$ (s)", y_label="$P(T)$",
                         exponent_sym=r"\alpha")

    box = (f"$R^2$ = {result.r_squared:.2f}\n"
           f"bin = {result.optimal_bin:.1f} ms")
    stats_box(ax_s, box, corner="tr")
    stats_box(ax_t, box, corner="tr")
    _apply_panel_label(ax_s, panel_label)
    # Detach from pyplot's figure manager so Jupyter's inline backend
    # does not auto-render the figure once on cell-exit AND again when
    # the user displays the returned `fig` (a single line such as
    # `fig = figure_criticality(crit); fig` otherwise renders twice).
    plt.close(fig)
    return fig
