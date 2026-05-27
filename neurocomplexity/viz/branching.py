"""Branching ratio: log(r_k) vs k with semilog fit + optional CI band."""
from __future__ import annotations

import numpy as np

from neurocomplexity.viz._palettes import get_palette, DEFAULT_PALETTE
from neurocomplexity.viz._style import _resolve_palette_and_axes, _apply_panel_label


def figure_branching(
    result,
    *,
    palette: str = DEFAULT_PALETTE,
    panel_label: str | None = None,
    figsize: tuple[float, float] | None = None,
    ax=None,
):
    """Render a Wilting-Priesemann branching-ratio panel.

    Plots ``r_k`` versus lag ``k`` on a semilog y-axis and overlays the
    log-linear fit. The estimated ``m̂`` and ``R²`` are annotated.

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
    nz = rks > 0
    ax.semilogy(ks[nz], rks[nz], "o", ms=3, mec="none",
                color=p["signal"], label="data")

    if nz.any() and np.isfinite(result.m):
        k_min = int(getattr(result, "k_min", 1))
        k_max = int(getattr(result, "k_max", int(ks.max())))
        # Anchor on median log-residual within the fit window so the line
        # visually splits the data points used to estimate m, rather than
        # being pinned to the (often saturated) first lag.
        fit_mask = (ks >= k_min) & (ks <= k_max) & nz
        if fit_mask.any():
            log_offset = float(np.median(np.log(rks[fit_mask])
                                          - ks[fit_mask] * np.log(result.m)))
            ks_line = ks[fit_mask]
            yy = np.exp(log_offset) * (result.m ** ks_line)
            ax.semilogy(ks_line, yy, "--", lw=1.1, color=p["accent"],
                        label=fr"$m={result.m:.3f}$ "
                              fr"(fit $k\in[{k_min},{k_max}]$)")
        else:
            k0 = ks[nz][0]
            r0 = rks[nz][0]
            b = r0 / (result.m ** k0)
            yy = b * (result.m ** ks)
            ax.semilogy(ks, yy, "--", lw=0.9, color=p["accent"],
                        label=fr"$m={result.m:.3f}$")

    lo = getattr(result, "r_values_ci_lower", None)
    hi = getattr(result, "r_values_ci_upper", None)
    if lo is not None and hi is not None:
        ax.fill_between(ks, np.asarray(lo), np.asarray(hi),
                        color=p["fill"], alpha=0.5, linewidth=0,
                        label="bootstrap CI")

    ax.set_xlabel("Lag $k$ (bins)")
    ax.set_ylabel(r"$r_k$")
    ax.legend(loc="upper right", fontsize=6.5, handlelength=1.6)
    # R² annotation moved to top-left to avoid colliding with the data points
    # that sit at low k.
    ax.text(0.02, 0.97, f"$R^2={result.r_squared:.2f}$",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=6, color=p["text"])

    _apply_panel_label(ax, panel_label)
    return fig
