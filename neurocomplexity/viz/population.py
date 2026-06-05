"""Population activity heatmap — Rastermap-style binned raster.

Reproduces the canonical Neuropixels population panel popularised by the
Pachitariu / Stringer / Steinmetz group (Stringer & Pachitariu 2019 *Nature*;
Steinmetz et al. 2019 *Nature*; Stringer & Pachitariu 2023 *Nature Neurosci.*
— "Rastermap"). The recipe is deliberately conservative and palette-neutral:

  1. Bin spikes finely (default 50 ms).
  2. Gaussian-smooth along the time axis (σ ≈ 1.5 bins).
  3. Per-unit max-normalisation so heterogeneous rates share a scale.
  4. Sort rows by peak time (or firing rate / unit id).
  5. ``imshow`` with ``gray_r`` — white = silence, black = peak.
  6. ``vmin=0``, ``vmax = 99th percentile`` so a few bursty bins do not
     wash the rest out.
  7. Standard ``Time (s)`` / ``Neuron`` axes; tiny colorbar "Norm. rate".
"""
from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

from neurocomplexity.core.recording import SpikeRecording
from neurocomplexity.viz._palettes import DEFAULT_PALETTE, get_palette
from neurocomplexity.viz._scale_bar import add_scale_bar
from neurocomplexity.viz._style import _apply_panel_label

# Palette-independent sequential map: white → black, the Rastermap default.
POPULATION_CMAP = "gray_r"


def _bin_units(rec: SpikeRecording, bin_size: float) -> tuple[np.ndarray, np.ndarray]:
    n_bins = int(np.floor(rec.duration / bin_size))
    edges = np.linspace(0.0, n_bins * bin_size, n_bins + 1)
    unit_ids = rec.units["id"].to_numpy(dtype=np.int64)
    counts = np.zeros((len(unit_ids), n_bins), dtype=np.float64)
    for i, uid in enumerate(unit_ids):
        mask = rec.unit_ids == uid
        if mask.any():
            h, _ = np.histogram(rec.spike_times[mask], bins=edges)
            counts[i] = h
    return counts, unit_ids


def _zscore_rows(M: np.ndarray) -> np.ndarray:
    """Per-unit z-score (alternative normalisation; kept for callers)."""
    mu = M.mean(axis=1, keepdims=True)
    sd = M.std(axis=1, keepdims=True)
    sd = np.where(sd > 0, sd, 1.0)
    return (M - mu) / sd


def _max_normalise_rows(M: np.ndarray) -> np.ndarray:
    """Per-unit max-normalisation: each row's peak rate becomes 1, silence 0."""
    peak = M.max(axis=1, keepdims=True)
    peak = np.where(peak > 0, peak, 1.0)
    return M / peak


def figure_population_heatmap(
    rec: SpikeRecording,
    *,
    bin_size: float = 0.05,
    smooth_sigma_bins: float = 1.5,
    sort_by: Literal["peak_time", "firing_rate", "unit_id"] = "peak_time",
    population: str | None = None,
    palette: str = DEFAULT_PALETTE,
    panel_label: str | None = None,
    figsize: tuple[float, float] | None = None,
    show_scale_bars: bool = False,
):
    """Render a sorted peak-normalised population activity heatmap.

    Bins every unit's spikes at ``bin_size`` (s), Gaussian-smooths each
    row with ``smooth_sigma_bins`` bins, peak-normalises each row, and
    stacks rows in the order chosen by ``sort_by``.

    Parameters
    ----------
    rec
        Spike recording.
    bin_size
        Bin size in seconds (default 50 ms).
    smooth_sigma_bins
        Gaussian smoothing std in *bins* (default 1.5).
    sort_by
        Row order: ``"peak_time"`` (default — produces a diagonal pattern
        when the population has clear sequential activity), ``"firing_rate"``,
        or ``"unit_id"``.
    population
        Restrict to one population by name. ``None`` → all units in ``rec``.
    palette
        Palette name.
    panel_label
        Optional one-letter panel label.
    figsize
        Figure size in inches.
    show_scale_bars
        If ``True``, replace the y-axis tick labels with a unit-count scale
        bar inset.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if sort_by not in ("peak_time", "firing_rate", "unit_id"):
        raise ValueError(
            f"sort_by must be 'peak_time'|'firing_rate'|'unit_id', got {sort_by!r}"
        )

    p = get_palette(palette)

    target = rec
    if population is not None:
        from dataclasses import replace
        keep_mask = rec.populations[population]
        keep_ids = rec.units.loc[keep_mask, "id"].to_numpy()
        unit_mask = np.isin(rec.unit_ids, keep_ids)
        target = replace(
            rec,
            spike_times=rec.spike_times[unit_mask],
            unit_ids=rec.unit_ids[unit_mask],
            units=rec.units[keep_mask].reset_index(drop=True),
            populations={"all": np.ones(int(np.sum(keep_mask)), dtype=bool)},
        )

    M, _ = _bin_units(target, bin_size)

    # Gaussian smooth along time (Rastermap convention) before normalisation.
    if smooth_sigma_bins and smooth_sigma_bins > 0 and M.shape[1] > 1:
        M = gaussian_filter1d(M, sigma=float(smooth_sigma_bins), axis=1, mode="nearest")

    N = _max_normalise_rows(M)

    if sort_by == "peak_time":
        order = np.argsort(np.argmax(N, axis=1))
    elif sort_by == "firing_rate":
        order = np.argsort(-M.sum(axis=1))
    else:
        order = np.arange(N.shape[0])
    Ns = N[order]

    size = figsize if figsize is not None else (6.4, 3.2)
    fig, ax = plt.subplots(figsize=size)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    cmap = plt.get_cmap(POPULATION_CMAP)
    vmax = float(np.percentile(Ns, 99)) if Ns.size else 1.0
    if vmax <= 0:
        vmax = 1.0
    im = ax.imshow(
        Ns, aspect="auto", cmap=cmap, vmin=0.0, vmax=vmax,
        interpolation="nearest",
        extent=[0, Ns.shape[1] * bin_size, Ns.shape[0], 0],
    )

    if show_scale_bars:
        xlen = max(5.0, Ns.shape[1] * bin_size * 0.1)
        ylen = max(1, Ns.shape[0] // 5)
        add_scale_bar(
            ax,
            x_length=xlen, x_label=f"{xlen:.0f} s",
            y_length=ylen, y_label=f"{ylen:d} units",
            location="lower right",
        )
    else:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Neuron (sorted)")

    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015, shrink=0.85)
    cb.set_label("Norm. rate", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_linewidth(0.6)

    _apply_panel_label(ax, panel_label)
    return fig
