"""Attractor manifold trajectory figure."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from neurocomplexity.analysis.manifold import ManifoldResult
from neurocomplexity.viz._palettes import DEFAULT_PALETTE, get_palette
from neurocomplexity.viz._style import stats_box


def _make_time_cmap(palette: str):
    from matplotlib.colors import LinearSegmentedColormap
    p = get_palette(palette)
    return LinearSegmentedColormap.from_list(
        f"nc_time_{palette}", [p["muted"], p["signal"]],
    )


def _interval_colors(palette: str, n_intervals: int) -> list[str]:
    p = get_palette(palette)
    cat = p["categorical"]
    return [cat[i % len(cat)] for i in range(n_intervals)]


def figure_manifold(result: ManifoldResult, *,
                    color_by: str = "time",
                    rec=None,
                    palette: str = DEFAULT_PALETTE,
                    ax=None,
                    figsize: tuple[float, float] | None = None):
    """Plot 2-D or 3-D manifold trajectory.

    Parameters
    ----------
    color_by
        ``"time"`` (default) colors points by bin index using a palette
        gradient. Otherwise must match a key in ``rec.intervals``.
    rec
        Required if ``color_by`` is an interval name.
    palette
        Palette name (forest / wine / sage).
    ax
        Optional Axes for ``dims=2`` only. Ignored for ``dims=3``.
    """
    p = get_palette(palette)
    coords = result.coords
    T = coords.shape[0]
    dims = result.dims

    if color_by != "time":
        if rec is None:
            raise ValueError(
                f"color_by={color_by!r} requires passing rec= so intervals can be looked up"
            )
        if color_by not in rec.intervals:
            raise ValueError(
                f"color_by={color_by!r} not in rec.intervals; available: {list(rec.intervals)}"
            )

    if dims == 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure(figsize=figsize or (5.5, 5.0))
        ax3 = fig.add_subplot(111, projection="3d")
        target_ax = ax3
    else:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize or (5.0, 4.5))
        else:
            fig = ax.figure
        target_ax = ax

    if color_by == "time":
        cmap = _make_time_cmap(palette)
        t_norm = np.linspace(0.0, 1.0, T)
        if dims == 2:
            target_ax.scatter(coords[:, 0], coords[:, 1], c=t_norm, cmap=cmap,
                              s=10, edgecolor="none", zorder=2)
            # Trajectory line uses a colour-graded line collection so the
            # orbit is readable on top of dense scatter, rather than the
            # near-invisible single grey line.
            from matplotlib.collections import LineCollection
            segs = np.stack([coords[:-1], coords[1:]], axis=1)
            lc = LineCollection(segs, cmap=cmap,
                                array=t_norm[:-1], linewidths=0.9,
                                alpha=0.55, zorder=1)
            target_ax.add_collection(lc)
        else:
            target_ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                              c=t_norm, cmap=cmap, s=10, edgecolor="none")
            target_ax.plot(coords[:, 0], coords[:, 1], coords[:, 2],
                           color=p["accent"], alpha=0.5, lw=0.7)
        # Time colourbar (2-D only — 3-D mplot3d makes it awkward).
        if dims == 2:
            import matplotlib.cm as mcm
            from matplotlib.colors import Normalize
            sm = mcm.ScalarMappable(norm=Normalize(0, T), cmap=cmap)
            sm.set_array([])
            cb = target_ax.figure.colorbar(sm, ax=target_ax,
                                            fraction=0.04, pad=0.02)
            cb.set_label("bin (time)", fontsize=6)
            cb.ax.tick_params(labelsize=6)
            cb.outline.set_linewidth(0.6)
        # Robust axis limits: 1-99 percentile clip prevents a few outlier
        # bins (e.g. stimulus-onset transients) from compressing the bulk.
        for axis_idx, axis_set in zip(
            range(dims),
            ((target_ax.set_xlim, target_ax.set_ylim)
             if dims == 2 else
             (target_ax.set_xlim, target_ax.set_ylim, target_ax.set_zlim))
        ):
            col = coords[:, axis_idx]
            lo = float(np.percentile(col, 1))
            hi = float(np.percentile(col, 99))
            if hi > lo:
                pad = 0.05 * (hi - lo)
                axis_set(lo - pad, hi + pad)
    else:
        # Interval coloring: assign each bin to first interval row it falls within.
        df = rec.intervals[color_by]
        bin_size = result.bin_size_seconds
        bin_centers = (np.arange(T) + 0.5) * bin_size
        n_rows = len(df)
        color_idx = np.full(T, -1, dtype=np.int64)
        for ri, row in enumerate(df.itertuples(index=False)):
            s = float(row.start_time)
            e = float(row.stop_time)
            mask = (bin_centers >= s) & (bin_centers < e)
            color_idx[mask] = ri
        colors = _interval_colors(palette, n_rows)
        # plot uncategorized first in muted
        un = color_idx < 0
        if dims == 2:
            if un.any():
                target_ax.scatter(coords[un, 0], coords[un, 1],
                                  c=p["muted"], s=10, alpha=0.4, edgecolor="none")
            for ri in range(n_rows):
                m = color_idx == ri
                if m.any():
                    target_ax.scatter(coords[m, 0], coords[m, 1],
                                      c=colors[ri], s=14, edgecolor="none",
                                      label=f"{color_by}[{ri}]")
        else:
            if un.any():
                target_ax.scatter(coords[un, 0], coords[un, 1], coords[un, 2],
                                  c=p["muted"], s=10, alpha=0.4, edgecolor="none")
            for ri in range(n_rows):
                m = color_idx == ri
                if m.any():
                    target_ax.scatter(coords[m, 0], coords[m, 1], coords[m, 2],
                                      c=colors[ri], s=14, edgecolor="none",
                                      label=f"{color_by}[{ri}]")
        if n_rows <= 10:
            target_ax.legend(frameon=False, loc="upper left",
                             bbox_to_anchor=(0.0, 1.18),
                             ncol=min(4, n_rows))

    target_ax.set_xlabel(f"{result.method.upper()}-1", color=p["text"])
    target_ax.set_ylabel(f"{result.method.upper()}-2", color=p["text"])
    if dims == 3:
        target_ax.set_zlabel(f"{result.method.upper()}-3", color=p["text"])

    if dims == 2:
        if result.method == "pca" and result.explained_variance_ratio is not None:
            ev = result.explained_variance_ratio
            pct = " / ".join(f"{100*v:.1f}%" for v in ev)
            box = f"N = {result.n_units}\nvar = {pct}"
        else:
            box = f"N = {result.n_units}"
        # Manifold trajectories typically fill the cloud body → top-right
        # corner is empty unless the projection happens to drift up-right;
        # the auto-axis percentile-clip above prevents that for spike data.
        stats_box(target_ax, box, corner="tr")

    return fig
