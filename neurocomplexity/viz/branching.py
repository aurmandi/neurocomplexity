"""Wilting MR estimator visualisation: log(r_k) vs k with exponential fit."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from neurocomplexity.viz._style import PALETTE, panel_label


def figure_branching(result, *, fig=None):
    if fig is None:
        fig, ax = plt.subplots(figsize=(3.0, 2.2))
    else:
        ax = fig.subplots(1, 1)
    ks = result.k_lags
    rks = result.r_values
    nz = rks > 0
    ax.semilogy(ks[nz], rks[nz], "o", ms=3, mec="none",
                color=PALETTE["signal"], label="data")
    if nz.any() and np.isfinite(result.m):
        # plot fit r = b * m^k anchored at first valid k
        k0 = ks[nz][0]; r0 = rks[nz][0]
        b = r0 / (result.m ** k0)
        yy = b * (result.m ** ks)
        ax.semilogy(ks, yy, "--", lw=0.9, color=PALETTE["accent"],
                    label=fr"$m={result.m:.3f}$")
    ax.set_xlabel("Lag $k$ (bins)")
    ax.set_ylabel(r"$r_k = \mathrm{Cov}(A_t,A_{t+k})/\mathrm{Var}(A_t)$")
    ax.legend(loc="upper right")
    # Tag near-critical region
    ax.text(0.02, 0.04, f"$R^2={result.r_squared:.2f}$",
            transform=ax.transAxes, fontsize=6, color=PALETTE["muted"])
    panel_label(ax, "a", x=-0.22)
    fig.tight_layout()
    return fig
