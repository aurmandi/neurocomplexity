"""Avalanche size P(s) & lifetime P(T) distributions with power-law fits."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from neurocomplexity.viz._palettes import DEFAULT_PALETTE, get_palette
from neurocomplexity.viz._style import _apply_panel_label

# Critical branching-process / mean-field directed-percolation exponents
# (Sethna 2001; Beggs & Plenz 2003; Friedman et al. 2012): size P(s) ~ s^-1.5,
# duration P(T) ~ T^-2.0. Drawn as a black dashed reference so deviation of the
# empirical exponent from criticality is read directly off the panel.
_MF_TAU = 1.5
_MF_ALPHA = 2.0

# Number of log-spaced bins per display mode (the empirical-density modes).
_NBINS = {"pdf": 30, "pdf_fine": 60}


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


def _ccdf(values):
    """Empirical complementary CDF ``P(X >= x)``.

    Every observation becomes one point on the survival curve, so all N
    avalanches appear with no binning choice (Clauset, Shalizi & Newman 2009,
    the recommended display for heavy-tailed data). On log-log axes a
    ``P(x) ~ x^-a`` density appears as a line of slope ``-(a-1)``.
    """
    v = np.sort(np.asarray(values, dtype=float))
    v = v[v > 0]
    if v.size == 0:
        return np.array([]), np.array([])
    n = v.size
    # P(X >= v_i) for the sorted sample: (n - i) / n at the i-th smallest.
    ccdf = (n - np.arange(n)) / n
    return v, ccdf


def _empirical_xy(values, display):
    if display == "ccdf":
        return _ccdf(values)
    return _log_pdf(values, nbins=_NBINS.get(display, 30))


def _draw_powerlaw_panel(ax, xs, ps, alpha, *, p, x_label, y_label,
                         exponent_sym, ref_exponent=None, ref_label=None,
                         title=None, cumulative=False, markers=True):
    """One distribution panel: empirical curve + fitted + critical guides.

    ``cumulative=True`` means ``ps`` is a CCDF, so reference lines are drawn
    with exponent ``a-1`` while the legend still reports the underlying
    density exponent ``a`` (the physically meaningful quantity).
    """
    def _slope(a):
        return (a - 1.0) if cumulative else a

    if markers:
        ax.loglog(xs, ps, "-", color=p["signal"], lw=1.2, marker="o", ms=2.5,
                  mec="white", mew=0.3, alpha=0.95, zorder=3, label="data")
    else:
        # CCDF / fine PDF: hundreds of points → draw as a bare line.
        ax.loglog(xs, ps, "-", color=p["signal"], lw=1.0, alpha=0.95,
                  zorder=3, label="data")

    if xs.size and np.isfinite(alpha):
        # Fitted guide: MLE density exponent, anchored at the median log-offset
        # so it threads the data cloud rather than pinning to one bin.
        se = _slope(alpha)
        log_offset = float(np.median(np.log(ps) + se * np.log(xs)))
        xx = np.array([xs.min(), xs.max()])
        yy = np.exp(log_offset) * xx ** (-se)
        ax.loglog(xx, yy, "--", lw=1.3, color=p["accent"], zorder=4,
                  label=fr"${exponent_sym} = {alpha:.2f}$")
    if ref_exponent is not None and xs.size:
        sr = _slope(ref_exponent)
        log_off_ref = float(np.median(np.log(ps) + sr * np.log(xs)))
        xx = np.array([xs.min(), xs.max()])
        yy = np.exp(log_off_ref) * xx ** (-sr)
        ax.loglog(xx, yy, "--", lw=1.0, color=p["text"], alpha=0.65, zorder=2,
                  label=ref_label or fr"critical (${exponent_sym}={ref_exponent:.1f}$)")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title, loc="center")
    ax.legend(loc="lower left", handlelength=1.6, borderpad=0.3)


def _ylabels(display):
    """(size y-label, duration y-label) for the chosen display mode."""
    if display == "ccdf":
        return r"$P(S \geq s)$", r"$P(T \geq t)$"
    return "$P(S)$", "$P(T)$"


def figure_criticality(
    result,
    *,
    palette: str = DEFAULT_PALETTE,
    panel_label: str | None = None,
    figsize: tuple[float, float] | None = None,
    display: str = "ccdf",
    title: str | None = None,
    ax=None,
):
    """Plot avalanche size and lifetime distributions with power-law fits.

    Standalone call draws two panels — size and duration — each titled, with
    its fitted exponent (orange dashed) and the critical reference (black
    dashed, ``τ=3/2`` / ``α=2``). When an ``ax`` is supplied the compact
    single-panel size form is drawn instead (for composite figures).

    Parameters
    ----------
    display
        How the empirical distribution is shown:

        * ``"ccdf"`` (default) — complementary CDF ``P(X≥x)``; **every**
          avalanche appears as one point with no binning, smooth at any N.
          Reference-line slopes shift to ``a-1`` while the legend still
          reports the density exponent ``a``.
        * ``"pdf"`` — log-binned probability density (~30 bins) with markers.
        * ``"pdf_fine"`` — finely log-binned density (~60 bins); shows the raw
          empirical distribution including the noisy tail.
    """
    if display not in {"pdf", "pdf_fine", "ccdf"}:
        raise ValueError(
            f"unknown display={display!r}; choose 'pdf', 'pdf_fine', or 'ccdf'")
    p = get_palette(palette)
    cumulative = display == "ccdf"
    markers = display == "pdf"
    yl_s, yl_t = _ylabels(display)
    composite = ax is not None

    if composite:
        fig = ax.figure
        xs, ps = _empirical_xy(result.sizes, display)
        _draw_powerlaw_panel(ax, xs, ps, result.alpha_s, p=p,
                             x_label="Avalanche Size", y_label=yl_s,
                             exponent_sym=r"\tau", ref_exponent=_MF_TAU,
                             ref_label=fr"critical ($\tau=3/2$)",
                             cumulative=cumulative, markers=markers)
        _apply_panel_label(ax, panel_label)
        return fig

    size = figsize if figsize is not None else (5.4, 2.6)
    fig, (ax_s, ax_t) = plt.subplots(1, 2, figsize=size)

    xs, ps = _empirical_xy(result.sizes, display)
    _draw_powerlaw_panel(ax_s, xs, ps, result.alpha_s, p=p,
                         x_label="Avalanche Size", y_label=yl_s,
                         exponent_sym=r"\tau", ref_exponent=_MF_TAU,
                         ref_label=fr"critical ($\tau=3/2$)",
                         title="Size Distribution",
                         cumulative=cumulative, markers=markers)
    xt, pt = _empirical_xy(result.lifetimes, display)
    _draw_powerlaw_panel(ax_t, xt, pt, result.alpha_t, p=p,
                         x_label="Avalanche Duration (s)", y_label=yl_t,
                         exponent_sym=r"\alpha", ref_exponent=_MF_ALPHA,
                         ref_label=fr"critical ($\alpha=2$)",
                         title="Duration Distribution",
                         cumulative=cumulative, markers=markers)

    if title:
        fig.suptitle(title, fontweight="bold", fontsize=9)
    _apply_panel_label(ax_s, panel_label)
    # Detach from pyplot's figure manager so Jupyter's inline backend
    # does not auto-render the figure once on cell-exit AND again when
    # the user displays the returned `fig` (a single line such as
    # `fig = figure_criticality(crit); fig` otherwise renders twice).
    plt.close(fig)
    return fig
