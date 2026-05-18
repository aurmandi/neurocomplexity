"""Population activity heatmap: binned, z-scored, sorted matrix."""
from __future__ import annotations

from typing import Literal

import numpy as np
import matplotlib.pyplot as plt

from neurocomplexity.core.recording import SpikeRecording
from neurocomplexity.viz._palettes import (
    get_palette, DEFAULT_PALETTE, diverging_cmap,
)
from neurocomplexity.viz._style import _apply_panel_label
from neurocomplexity.viz._scale_bar import add_scale_bar


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
    mu = M.mean(axis=1, keepdims=True)
    sd = M.std(axis=1, keepdims=True)
    sd = np.where(sd > 0, sd, 1.0)
    return (M - mu) / sd


def figure_population_heatmap(
    rec: SpikeRecording,
    *,
    bin_size: float = 0.05,
    sort_by: Literal["peak_time", "firing_rate", "unit_id"] = "peak_time",
    population: str | None = None,
    palette: str = DEFAULT_PALETTE,
    panel_label: str | None = None,
    figsize: tuple[float, float] | None = None,
    show_scale_bars: bool = True,
):
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
    Z = _zscore_rows(M)

    if sort_by == "peak_time":
        order = np.argsort(np.argmax(Z, axis=1))
    elif sort_by == "firing_rate":
        order = np.argsort(-M.sum(axis=1))
    else:
        order = np.arange(Z.shape[0])
    Zs = Z[order]

    size = figsize if figsize is not None else (3.6, 2.4)
    fig, ax = plt.subplots(figsize=size)
    vmax = float(np.percentile(np.abs(Zs), 99)) if Zs.size else 1.0
    cmap = diverging_cmap(palette)
    ax.imshow(
        Zs, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax,
        interpolation="nearest",
        extent=[0, Zs.shape[1] * bin_size, Zs.shape[0], 0],
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if show_scale_bars:
        xlen = max(5.0, Zs.shape[1] * bin_size * 0.1)
        ylen = max(1, Zs.shape[0] // 5)
        add_scale_bar(
            ax,
            x_length=xlen, x_label=f"{xlen:.0f} s",
            y_length=ylen, y_label=f"{ylen:d} units",
            location="lower right",
        )
    else:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Unit (sorted)")

    _apply_panel_label(ax, panel_label)
    fig.tight_layout()
    return fig
