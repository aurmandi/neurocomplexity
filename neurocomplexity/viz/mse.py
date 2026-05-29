"""Multi-scale entropy profile figure."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from neurocomplexity.analysis.mse import MSEResult
from neurocomplexity.viz._palettes import DEFAULT_PALETTE, get_palette, series_styles


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
    styles = series_styles(len(result.populations), palette)
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

    scales_arr = np.asarray(scales)
    nan_scales: set = set()
    for pi, name in enumerate(result.populations):
        y = np.asarray(sampen[pi], dtype=float)
        st = styles[pi]
        # NaN at high tau (SampEn unstable at coarse scales with short
        # series). Plot only finite points so the line breaks at the gap;
        # NaNs are NOT drawn at y=0 (that would misrepresent an undefined
        # value as zero). Undefined scales are reported textually instead.
        finite = np.isfinite(y)
        ax.plot(scales_arr, np.where(finite, y, np.nan),
                color=st["color"], marker=st["marker"],
                linestyle=st["linestyle"], label=name, lw=1.2, markersize=4)
        nan_scales.update(int(s) for s in scales_arr[~finite])
    ax.set_xlabel(r"Scale $\tau$", color=p["text"])
    ax.set_ylabel("SampEn", color=p["text"])
    ax.set_title(
        f"MSE   m={result.m}  r={result.r_factor:g}·SD  "
        f"bin={result.bin_size_seconds*1e3:.0f} ms"
        + ("  (with surrogate envelope)" if envelope_drawn else ""),
        loc="left", fontsize=8, color=p["text"], pad=8,
    )
    if nan_scales:
        scales_txt = ", ".join(str(s) for s in sorted(nan_scales))
        ax.text(0.98, 0.02,
                f"SampEn undefined at scale {scales_txt} (omitted)",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=5.5, color=p["muted"])
    elif not envelope_drawn and null_result is None:
        ax.text(0.98, 0.02,
                "pass null_result= for surrogate envelope",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=5.5, color=p["muted"])
    if len(result.populations) > 1:
        ax.legend(frameon=False, loc="upper left",
                  bbox_to_anchor=(0.0, 1.18),
                  ncol=min(4, len(result.populations)))
    return fig
