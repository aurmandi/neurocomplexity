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
    p, fig, ax = _resolve_palette_and_axes(
        palette=palette, ax=ax, figsize=figsize, default_size=(3.0, 2.2),
    )

    ks = np.asarray(result.k_lags)
    rks = np.asarray(result.r_values)
    nz = rks > 0
    ax.semilogy(ks[nz], rks[nz], "o", ms=3, mec="none",
                color=p["signal"], label="data")

    if nz.any() and np.isfinite(result.m):
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
    ax.legend(loc="upper right")
    ax.text(0.02, 0.04, f"$R^2={result.r_squared:.2f}$",
            transform=ax.transAxes, fontsize=6, color=p["muted"])

    _apply_panel_label(ax, panel_label)
    fig.tight_layout()
    return fig
