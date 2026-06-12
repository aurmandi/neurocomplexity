"""Branching ratio: log(r_k) vs time lag with semilog fit + optional CI band."""
from __future__ import annotations

import numpy as np

from neurocomplexity.viz._palettes import DEFAULT_PALETTE
from neurocomplexity.viz._style import _apply_panel_label, _resolve_palette_and_axes


def figure_branching(
    result,
    *,
    palette: str = DEFAULT_PALETTE,
    panel_label: str | None = None,
    figsize: tuple[float, float] | None = None,
    ax=None,
):
    """Render a Wilting-Priesemann branching-ratio panel.

    Plots ``r_k`` versus the time lag on a semilog y-axis and overlays the
    log-linear fit. The estimated ``m̂`` and ``R²`` are reported in the
    legend (top-left). The lag axis is shown in milliseconds
    (``k × bin_size``) rather than abstract bin counts.

    Parameters
    ----------
    result
        :class:`~neurocomplexity.analysis.BranchingResult`.
    palette
        Name of a palette registered in
        :mod:`~neurocomplexity.viz._palettes` (default ``"paper"``).
    panel_label
        Optional one-letter panel label drawn in the corner (e.g. ``"A"``).
    figsize
        Figure size in inches; ignored when ``ax`` is provided.
    ax
        Existing matplotlib ``Axes`` to draw into (for composite figures).
        ``None`` creates a new figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    p, fig, ax = _resolve_palette_and_axes(
        palette=palette, ax=ax, figsize=figsize, default_size=(3.6, 2.7),
    )

    ks = np.asarray(result.k_lags)
    rks = np.asarray(result.r_values)
    # Convert lag (bins) → milliseconds so the decay timescale is physical.
    bs_ms = float(getattr(result, "bin_size_seconds", 0.0)) * 1e3
    to_ms = (lambda k: k * bs_ms) if bs_ms > 0 else (lambda k: k)
    nz = rks > 0
    ax.semilogy(to_ms(ks[nz]), rks[nz], "o", ms=3, mec="none",
                color=p["signal"], label="data")

    if nz.any() and np.isfinite(result.m):
        k_min = int(getattr(result, "k_min", 1))
        k_max = int(getattr(result, "k_max", int(ks.max())))
        # Anchor on median log-residual within the fit window so the line
        # visually splits the data points used to estimate m, rather than
        # being pinned to the (often saturated) first lag.
        fit_mask = (ks >= k_min) & (ks <= k_max) & nz
        fit_label = (fr"$m={result.m:.3f}$, $R^2={result.r_squared:.2f}$")
        if fit_mask.any():
            log_offset = float(np.median(np.log(rks[fit_mask])
                                          - ks[fit_mask] * np.log(result.m)))
            ks_line = ks[fit_mask]
            yy = np.exp(log_offset) * (result.m ** ks_line)
            ax.semilogy(to_ms(ks_line), yy, "--", lw=1.1, color=p["accent"],
                        label=fit_label)
        else:
            k0 = ks[nz][0]
            r0 = rks[nz][0]
            b = r0 / (result.m ** k0)
            yy = b * (result.m ** ks)
            ax.semilogy(to_ms(ks), yy, "--", lw=0.9, color=p["accent"],
                        label=fit_label)

    lo = getattr(result, "r_values_ci_lower", None)
    hi = getattr(result, "r_values_ci_upper", None)
    if lo is not None and hi is not None:
        ax.fill_between(to_ms(ks), np.asarray(lo), np.asarray(hi),
                        color=p["fill"], alpha=0.5, linewidth=0,
                        label="bootstrap CI")

    ax.set_xlabel("Time lag (ms)" if bs_ms > 0 else "Time lag $k$ (bins)")
    ax.set_ylabel(r"$r_k$")
    # Head-room above the data so the top-left legend clears the high-r_k
    # points at short lag (r_k decays with lag, so the curve is highest at
    # the left edge).
    y0, y1 = ax.get_ylim()
    if y1 > y0:
        ax.set_ylim(y0, y1 * 1.8)
    ax.legend(loc="upper left", handlelength=1.6)

    _apply_panel_label(ax, panel_label)
    return fig
