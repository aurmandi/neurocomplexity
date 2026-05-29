"""Multi-scale entropy profile figure."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from neurocomplexity.analysis.mse import MSEResult
from neurocomplexity.viz._palettes import DEFAULT_PALETTE, get_palette


def _pop_colors(palette_name: str, n: int) -> list[str]:
    p = get_palette(palette_name)
    cat = p["categorical"]
    return [cat[i % len(cat)] for i in range(n)]


def figure_mse(result: MSEResult, *,
               null_result=None,
               ax=None,
               palette: str = DEFAULT_PALETTE,
               show_envelope: bool = True,
               figsize: tuple[float, float] | None = None):
    """Plot SampEn vs scale, one line per population.

    With ``null_result`` provided and ``show_envelope=True``, draw a grey
    [mean +/- 2 SD] band per population from surrogate sampen matrices.
    """
    colors = _pop_colors(palette, len(result.populations))
    p = get_palette(palette)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (5.0, 4.0))
    else:
        fig = ax.figure

    scales = result.scales
    sampen = result.sampen  # (P, S)

    envelope_drawn = False
    if null_result is not None and show_envelope:
        null_arr = np.asarray(null_result.null_distribution)
        if null_arr.ndim == 3 and null_arr.shape[1:] == sampen.shape:
            mean = np.nanmean(null_arr, axis=0)
            sd = np.nanstd(null_arr, axis=0)
            for pi in range(sampen.shape[0]):
                ax.fill_between(scales, mean[pi] - 2 * sd[pi], mean[pi] + 2 * sd[pi],
                                color=p["muted"], alpha=0.3, linewidth=0)
            envelope_drawn = True

    for pi, name in enumerate(result.populations):
        y = sampen[pi]
        # Solid markers where finite, hollow markers at NaN coarse-grained
        # scales (SampEn unstable at high tau with short series).
        ax.plot(scales, y, color=colors[pi], marker="o",
                label=name, lw=1.2, markersize=4)
        nan_mask = ~np.isfinite(y)
        if nan_mask.any():
            ax.scatter(np.asarray(scales)[nan_mask],
                       np.zeros(int(nan_mask.sum())),
                       marker="x", s=18, color=colors[pi],
                       label=f"{name} (NaN)" if pi == 0 else None)
    ax.set_xlabel(r"Scale $\tau$", color=p["text"])
    ax.set_ylabel("SampEn", color=p["text"])
    ax.set_title(
        f"MSE   m={result.m}  r={result.r_factor:g}·SD  "
        f"bin={result.bin_size_seconds*1e3:.0f} ms"
        + ("  (with surrogate envelope)" if envelope_drawn else ""),
        loc="left", fontsize=8, color=p["text"], pad=8,
    )
    if not envelope_drawn and null_result is None:
        ax.text(0.98, 0.02,
                "pass null_result= for surrogate envelope",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=5.5, color=p["muted"])
    if len(result.populations) > 1:
        ax.legend(frameon=False, loc="upper left",
                  bbox_to_anchor=(0.0, 1.18),
                  ncol=min(4, len(result.populations)))
    fig.tight_layout()
    return fig
