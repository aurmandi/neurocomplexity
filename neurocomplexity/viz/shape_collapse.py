"""Avalanche shape collapse: raw shape family + collapsed onto F(t/T)."""
from __future__ import annotations

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

from neurocomplexity.viz._palettes import DEFAULT_PALETTE, get_palette
from neurocomplexity.viz._style import _apply_panel_label

# Viridis is the matplotlib default perceptually-uniform sequential colormap
# (Smith & van der Walt 2015) and the de-facto standard in the avalanche shape-
# collapse literature (Fontenele et al. 2019; Ponce-Alvarez et al. 2018). It is
# colour-blind safe, monotonic in luminance, and prints to greyscale faithfully.
# Decoupled from the figure palette by design. The duration gradient is what
# makes the collapse legible — short→long avalanches in cool→warm hues fanning
# out in the raw panel and converging onto one curve in the collapsed panel.
SHAPE_COLLAPSE_CMAP = "viridis"

# Degree of the polynomial fitted to the pooled rescaled points as a smooth
# estimate of the universal scaling function F(t/T). A quartic captures the
# single-humped inverted-parabola shape without over-fitting the tails.
_POLY_DEG = 4

# Default cap on how many duration classes are *drawn* (evenly subsampled).
# Mean + polynomial fit still use every class; this only de-clutters the family.
_MAX_CLASSES = 10

# White outline drawn under the hero curves so they stay legible over the
# shape family regardless of the family colour (Wilke 2019, halo trick).
_HALO = [pe.withStroke(linewidth=2.8, foreground="white")]


def _sequential_cmap(p):  # kept for backward compat with callers / tests
    return plt.get_cmap(SHAPE_COLLAPSE_CMAP)


def _family_colors(n, color_by_duration, p):
    """Per-curve colours for the shape family.

    ``color_by_duration=True`` (default) → viridis gradient (one hue per
    duration class), which is what makes the fan-out / collapse legible.
    ``False`` → a single muted grey for a minimal, monochrome look.
    """
    if color_by_duration:
        cmap = plt.get_cmap(SHAPE_COLLAPSE_CMAP)
        return [cmap(i / max(n - 1, 1)) for i in range(n)]
    return [p["muted"]] * n


def _smooth(y, on=True):
    """Savitzky-Golay smoothing with a length-safe odd window (no-op if short)."""
    y = np.asarray(y, dtype=float)
    n = y.size
    if not on or n < 7:
        return y
    win = min(n if n % 2 == 1 else n - 1, 11)
    if win < 5:
        return y
    try:
        return savgol_filter(y, win, 3)
    except Exception:
        return y


def _subsample_idx(n, k):
    """Indices of ``k`` evenly-spaced classes out of ``n`` (all if ``n<=k``)."""
    if not k or n <= k:
        return list(range(n))
    return sorted(set(np.linspace(0, n - 1, k).round().astype(int).tolist()))


def _dense_head(durations):
    """Count of leading duration classes that are consecutive (no gaps).

    Duration classes are kept by the analysis only when they have >= 5
    avalanches, so a gap in the sorted durations (e.g. ``...26, 27, 30, 31``)
    marks the onset of the under-sampled tail whose mean shapes are noisy.
    Drawing only the dense consecutive head removes those jagged curves
    without touching the upstream gamma/residual fit.
    """
    d = np.asarray(durations)
    if d.size <= 1:
        return int(d.size)
    cut = d.size
    for i in range(1, d.size):
        if d[i] - d[i - 1] > 1:
            cut = i
            break
    return max(cut, 2)  # always keep at least two classes to draw


def _raw_curve(T, shape, bs, smooth):
    """Densify + smooth one mean shape so short polylines read as clean arcs."""
    t = np.arange(T) * bs * 1e3
    sh = np.asarray(shape, dtype=float)
    if T >= 4:
        tt = np.linspace(t[0], t[-1], 50)
        return tt, _smooth(np.interp(tt, t, sh), smooth)
    return t, sh


def _expand_top(ax, lo, hi, frac=0.22):
    """Set y-limits with head-room above the data for a top-left annotation."""
    if hi > lo:
        ax.set_ylim(lo, hi + frac * (hi - lo))


def _draw_collapsed(ax, result, *, p, colors, color_by_duration=True,
                    smooth=True, max_classes=_MAX_CLASSES, show_annot=True):
    u = result.rescaled_x
    ys_all = np.asarray(result.rescaled_y)  # shape (n_durations, n_interp)
    cut = _dense_head(result.durations_used)
    ys = ys_all[:cut]  # dense, well-sampled classes only (display + hero)
    idx = _subsample_idx(len(ys), max_classes)
    fam_alpha = 0.45 if color_by_duration else 0.30
    for i in idx:
        ax.plot(u, _smooth(ys[i], smooth), lw=0.9, color=colors[i],
                alpha=fam_alpha, zorder=2)

    # Mean scaling function + polynomial fit — the universal shape F(t/T) that
    # readers judge collapse quality against (Friedman et al. 2012; Fontenele
    # et al. 2019). Mean is the literature-standard summary; the polynomial is
    # the smooth presentation curve. Both use *every* duration class.
    if ys.size:
        mean = _smooth(np.nanmean(ys, axis=0), smooth)
        ax.plot(u, mean, "-", color=p["signal"], lw=1.0, alpha=0.65,
                zorder=4, label="mean")

        # Quartic fit to the pooled rescaled points (robust to per-curve jitter).
        pts_u = np.tile(u, ys.shape[0])
        pts_y = ys.ravel()
        finite = np.isfinite(pts_y)
        if finite.sum() > _POLY_DEG:
            coef = np.polyfit(pts_u[finite], pts_y[finite], _POLY_DEG)
            ax.plot(u, np.polyval(coef, u), "-", color=p["accent"], lw=2.2,
                    zorder=5, path_effects=_HALO, label="scaling function")

        ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.86),
                  handlelength=1.6, frameon=False)

        # Robust y-limits: percentile clip so outlier shapes don't dominate,
        # then add top head-room for the corner annotation.
        finite_y = ys[np.isfinite(ys)]
        if finite_y.size:
            lo = float(np.percentile(finite_y, 1))
            hi = float(np.percentile(finite_y, 99))
            _expand_top(ax, lo, hi)

    ax.set_xlabel("Scaled duration ($t/T$)")
    ax.set_ylabel(r"Scaled size ($\langle a\rangle / T^{\gamma-1}$)")
    if show_annot:
        # Collapsed curves arch and peak centre → top-LEFT corner is empty.
        ax.text(0.03, 0.97,
                fr"$\gamma$ = {result.gamma:.3f}" + "\n"
                f"resid = {result.residual:.2g}",
                transform=ax.transAxes, ha="left", va="top",
                multialignment="left", linespacing=1.35,
                color=p["text"])


def _add_duration_colorbar(fig, ax, durations):
    cmap = plt.get_cmap(SHAPE_COLLAPSE_CMAP)
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=durations.min(), vmax=durations.max()),
    )
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.04)
    cb.set_label("Duration $T$ (bins)", fontsize=6)
    cb.ax.tick_params(labelsize=6)


def figure_shape_collapse(
    result,
    *,
    palette: str = DEFAULT_PALETTE,
    panel_label: str | None = None,
    figsize: tuple[float, float] | None = None,
    color_by_duration: bool = True,
    smooth: bool = True,
    max_classes: int | None = _MAX_CLASSES,
    title: str | None = None,
    ax=None,
):
    """Render mean avalanche shapes before / after rescaling by ``gamma``.

    Two sub-panels: raw mean shapes per duration class (the fan-out), and the
    collapsed curves at the optimal ``gamma`` with the mean scaling function
    and a polynomial fit overlaid (the convergence). The exponent and residual
    are annotated in the empty top-left corner.

    Parameters
    ----------
    color_by_duration
        When ``True`` (default) each duration class is coloured by a viridis
        gradient with a colourbar, so the raw panel fans out short→long and
        the collapsed panel visibly converges onto one curve. ``False`` draws
        the family in a single muted grey (minimal monochrome look).
    smooth
        Savitzky-Golay smoothing of each drawn curve (default ``True``) so
        short avalanche shapes read as clean arcs rather than jagged polylines.
        The fitted exponent and residual are unaffected (analysis is upstream).
    max_classes
        Cap on the number of duration classes *drawn* (evenly subsampled;
        default 10) to de-clutter dense recordings. The mean and polynomial
        fit still use every class. ``None`` draws all classes.

    See :func:`~neurocomplexity.viz.branching.figure_branching` for the other
    shared keyword arguments.
    """
    p = get_palette(palette)
    durations = np.asarray(result.durations_used)
    n = len(durations)
    colors = _family_colors(n, color_by_duration, p)
    composite = ax is not None

    if composite:
        fig = ax.figure
        _draw_collapsed(ax, result, p=p, colors=colors,
                        color_by_duration=color_by_duration,
                        smooth=smooth, max_classes=max_classes)
        _apply_panel_label(ax, panel_label)
        return fig

    size = figsize if figsize is not None else (6.8, 3.2)
    fig, (ax_raw, ax_col) = plt.subplots(1, 2, figsize=size)

    bs = result.bin_size_seconds
    cut = _dense_head(durations)
    idx = _subsample_idx(cut, max_classes)
    raw_alpha = 0.85 if color_by_duration else 0.6
    drawn_vals: list[float] = []
    for i in idx:
        T = int(durations[i])
        t, y = _raw_curve(T, result.mean_shapes[i], bs, smooth)
        ax_raw.plot(t, y, lw=1.0, color=colors[i], alpha=raw_alpha)
        drawn_vals.extend(np.asarray(y, dtype=float).ravel().tolist())
    ax_raw.set_xlabel("Time within avalanche (ms)")
    ax_raw.set_ylabel("Mean activity (spikes/bin)")
    # Robust y-limits suppress single-shape spikes that otherwise compress the
    # bulk of curves.
    if drawn_vals:
        arr = np.array(drawn_vals)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            lo, hi = float(np.percentile(arr, 1)), float(np.percentile(arr, 99))
            if hi > lo:
                ax_raw.set_ylim(lo, hi)

    _draw_collapsed(ax_col, result, p=p, colors=colors,
                    color_by_duration=color_by_duration,
                    smooth=smooth, max_classes=max_classes)

    if color_by_duration:
        _add_duration_colorbar(fig, ax_col, durations[:cut])

    if title:
        fig.suptitle(title, fontweight="bold", fontsize=9)
    _apply_panel_label(ax_raw, panel_label)
    return fig
