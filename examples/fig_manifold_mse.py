"""Manifold and MSE figures for Allen session 715093703, spontaneous block.

Generates panels saved to examples/figures/:
  manifold_pca.png          — PCA 2-D state-space trajectory, colored by time
  manifold_umap.png         — UMAP 2-D, colored by time
  mse.png                   — SampEn vs scale, one curve per area
  manifold_pca_umap_mse.png — composite 3-panel (PCA | UMAP | MSE)

Populations: LGd, VISp, CA1 — same three areas as the TE analysis.
MSE params: bin_size_s=0.025 (25 ms), scale_max=40.
"""
from __future__ import annotations

import warnings
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as mcm
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd

import neurocomplexity as nc
from neurocomplexity.core.provenance import ProvenanceRecord
from neurocomplexity.core.recording import SpikeRecording
from neurocomplexity.viz._palettes import DEFAULT_PALETTE, series_styles
from neurocomplexity.viz._style import (
    apply_style, current_palette, panel_label, top_strip,
)

# ── paths ─────────────────────────────────────────────────────────────────────
NWB_PATH = Path(
    r"C:\Users\Sazgar\OneDrive\Desktop\Arman_Dinarvand code sample"
    r"\neuropixel\NeuropixelVisCodingData_cache"
    r"\session_715093703\session_715093703.nwb"
)
FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── window + populations ───────────────────────────────────────────────────────
WINDOW_START = 3766.14
WINDOW_END   = 4066.39
AREAS        = ("LGd", "VISp", "CA1")
DROP_AREAS   = {"grey", "", "nan", "none"}

# ── analysis params ────────────────────────────────────────────────────────────
MSE_BIN_S    = 0.025   # 25 ms
MSE_SCALE    = 40
MANIFOLD_BIN = 0.050   # 50 ms
SIGMA_MS     = 50.0    # Gaussian smoothing (Churchland 2007)

DPI_STANDALONE = 200
DPI_COMPOSITE  = 250


# ── data loading ──────────────────────────────────────────────────────────────

def load_window(path: Path, start: float, end: float) -> SpikeRecording:
    import pynwb
    with pynwb.NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb  = io.read()
        u    = nwb.units
        uids = np.asarray(u.id[:], dtype=np.int64)
        cols = list(u.colnames)
        meta: dict = {"id": uids}
        for c in cols:
            if c in ("spike_times", "spike_amplitudes", "waveform_mean"):
                continue
            try:
                vals = u[c][:]
                if (isinstance(vals, np.ndarray) and vals.ndim == 1
                        and len(vals) == len(uids)):
                    meta[c] = vals
            except Exception:
                pass
        units_df = pd.DataFrame(meta)

        mask     = ((units_df["quality"] == "good").to_numpy()
                    if "quality" in units_df.columns
                    else np.ones(len(units_df), dtype=bool))
        keep_idx = np.where(mask)[0]

        all_times, all_owners, kept_ids = [], [], []
        for orig_i in keep_idx:
            st  = np.asarray(u["spike_times"][int(orig_i)], dtype=np.float64)
            sel = (st >= start) & (st < end)
            if not sel.any():
                continue
            tt      = st[sel] - start
            uid_val = int(uids[orig_i])
            all_times.append(tt)
            all_owners.append(np.full(tt.size, uid_val, dtype=np.int64))
            kept_ids.append(uid_val)

        kept_df        = units_df.iloc[keep_idx].reset_index(drop=True)
        active         = kept_df["id"].isin(kept_ids).to_numpy()
        units_df_final = kept_df[active].reset_index(drop=True)

        if ("peak_channel_id" in units_df_final.columns
                and nwb.electrodes is not None):
            try:
                elec = nwb.electrodes.to_dataframe()
                if "location" in elec.columns:
                    loc_map = elec["location"].to_dict()
                    units_df_final["brain_area"] = (
                        units_df_final["peak_channel_id"].map(loc_map))
            except Exception:
                pass

        st_arr  = np.concatenate(all_times)  if all_times  else np.empty(0)
        uid_arr = (np.concatenate(all_owners) if all_owners
                   else np.empty(0, dtype=np.int64))
        order   = np.argsort(st_arr, kind="stable")
        prov    = ProvenanceRecord.for_file(path, source_format="nwb-window")

    return SpikeRecording(
        spike_times=st_arr[order], unit_ids=uid_arr[order],
        units=units_df_final,
        populations={"all": np.ones(len(units_df_final), dtype=bool)},
        duration=float(end - start), sampling_rate=None, source=prov,
    )


def build_area_pops(rec: SpikeRecording, areas: tuple[str, ...]) -> SpikeRecording:
    if "brain_area" not in rec.units.columns:
        raise RuntimeError("brain_area column missing")
    rec_all = rec.with_populations(by="brain_area")
    pops = {k: v for k, v in rec_all.populations.items()
            if k in areas and k.lower() not in DROP_AREAS}
    missing = set(areas) - set(pops)
    if missing:
        print(f"  [warn] areas not found: {missing}")
    return replace(rec_all, populations=pops)


# ── manifold helpers ──────────────────────────────────────────────────────────

def _time_cmap() -> LinearSegmentedColormap:
    p = current_palette()
    return LinearSegmentedColormap.from_list("nc_time", [p["muted"], p["signal"]])


def _draw_manifold_ax(ax, coords: np.ndarray, method: str,
                      n_units: int, ev_ratio=None) -> None:
    """Draw a 2-D manifold scatter+trajectory on ax, colored by time."""
    p      = current_palette()
    cmap   = _time_cmap()
    T      = coords.shape[0]
    t_norm = np.linspace(0.0, 1.0, T)

    # trajectory line (behind scatter)
    segs = np.stack([coords[:-1], coords[1:]], axis=1)
    lc   = LineCollection(segs, cmap=cmap, array=t_norm[:-1],
                          linewidths=0.6, alpha=0.35, zorder=1)
    ax.add_collection(lc)

    # scatter
    ax.scatter(coords[:, 0], coords[:, 1], c=t_norm, cmap=cmap,
               s=5, edgecolor="none", zorder=2, rasterized=True)

    # percentile-clipped axis limits
    for get_lim, set_lim, col_i in (
        (ax.get_xlim, ax.set_xlim, 0),
        (ax.get_ylim, ax.set_ylim, 1),
    ):
        col  = coords[:, col_i]
        lo   = float(np.percentile(col, 1))
        hi   = float(np.percentile(col, 99))
        pad  = 0.06 * (hi - lo) if hi > lo else 0.1
        set_lim(lo - pad, hi + pad)

    ax.set_xlabel(f"{method.upper()}-1")
    ax.set_ylabel(f"{method.upper()}-2")

    if method == "pca" and ev_ratio is not None:
        pct = " / ".join(f"{100*v:.1f}%" for v in ev_ratio)
        strip = f"{method.upper()}   n={n_units}   {pct} var"
    else:
        strip = f"{method.upper()}   n={n_units}"
    top_strip(ax, strip)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    apply_style()
    p = current_palette()

    # 1. load
    print("loading NWB window ...")
    rec = load_window(NWB_PATH, WINDOW_START, WINDOW_END)
    print(f"  units={len(rec.units)}  spikes={rec.spike_times.size:,}")

    rec_areas = build_area_pops(rec, AREAS)
    pops      = list(rec_areas.populations.keys())
    n_per     = {pop: int(rec_areas.populations[pop].sum()) for pop in pops}
    print(f"  populations: {n_per}")

    # 2. PCA manifold
    print("PCA manifold ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mf_pca = nc.analysis.manifold(
            rec_areas, populations=pops, method="pca", dims=2,
            bin_size_s=MANIFOLD_BIN, sigma_ms=SIGMA_MS,
        )
    fig_pca = nc.viz.figure_manifold(mf_pca, color_by="time")
    fig_pca.savefig(FIG_DIR / "manifold_pca.png",
                    dpi=DPI_STANDALONE, bbox_inches="tight")
    plt.close(fig_pca)
    print("  -> manifold_pca.png")

    # 3. UMAP manifold
    print("UMAP manifold ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mf_umap = nc.analysis.manifold(
            rec_areas, populations=pops, method="umap", dims=2,
            bin_size_s=MANIFOLD_BIN, sigma_ms=SIGMA_MS, random_state=0,
        )
    fig_umap = nc.viz.figure_manifold(mf_umap, color_by="time")
    fig_umap.savefig(FIG_DIR / "manifold_umap.png",
                     dpi=DPI_STANDALONE, bbox_inches="tight")
    plt.close(fig_umap)
    print("  -> manifold_umap.png")

    # 4. MSE
    print("MSE (scale_max=40, bin=25 ms) …")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mse_res = nc.analysis.multiscale_entropy(
            rec_areas, populations=pops,
            bin_size_s=MSE_BIN_S, scale_max=MSE_SCALE,
        )
    fig_mse = nc.viz.figure_mse(mse_res)
    fig_mse.savefig(FIG_DIR / "mse.png",
                    dpi=DPI_STANDALONE, bbox_inches="tight")
    plt.close(fig_mse)
    print("  -> mse.png")

    # 5. composite 3-panel
    print("composite figure ...")

    # Nature double-column width = 7.09 in; height chosen so panels are
    # roughly square. Extra right margin reserved for colorbar.
    fig, axes = plt.subplots(
        1, 3,
        figsize=(7.09, 2.55),
        gridspec_kw={"width_ratios": [1, 1, 1], "wspace": 0.38},
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.07, right=0.88, top=0.88, bottom=0.16,
                        wspace=0.42)

    ax_pca, ax_umap, ax_mse = axes

    # panel (a) PCA
    _draw_manifold_ax(ax_pca, mf_pca.coords, "pca",
                      mf_pca.n_units, mf_pca.explained_variance_ratio)
    panel_label(ax_pca, "a")

    # panel (b) UMAP
    _draw_manifold_ax(ax_umap, mf_umap.coords, "umap", mf_umap.n_units)
    panel_label(ax_umap, "b")

    # shared time colorbar for (a) and (b) — right of panel (b)
    cmap   = _time_cmap()
    T      = mf_pca.coords.shape[0]
    sm     = mcm.ScalarMappable(norm=Normalize(0, T * MANIFOLD_BIN), cmap=cmap)
    sm.set_array([])
    # position: right edge of ax_umap, same height
    pos_b  = ax_umap.get_position()
    cax    = fig.add_axes([pos_b.x1 + 0.012, pos_b.y0,
                           0.012, pos_b.height])
    cbar   = fig.colorbar(sm, cax=cax)
    cbar.set_label("time (s)", fontsize=6, labelpad=4)
    cbar.ax.tick_params(labelsize=6, length=2, width=0.5)
    cbar.outline.set_linewidth(0.5)
    # only 3 ticks
    cbar.set_ticks(np.linspace(0, T * MANIFOLD_BIN, 3))
    cbar.set_ticklabels([f"{v:.0f}" for v in np.linspace(0, T * MANIFOLD_BIN, 3)])

    # panel (c) MSE
    styles     = series_styles(len(mse_res.populations))
    scales_arr = np.asarray(mse_res.scales)
    nan_scales: set = set()
    for pi, name in enumerate(mse_res.populations):
        y      = np.asarray(mse_res.sampen[pi], dtype=float)
        st_    = styles[pi]
        finite = np.isfinite(y)
        ax_mse.plot(scales_arr, np.where(finite, y, np.nan),
                    color=st_["color"], marker=st_["marker"],
                    linestyle=st_["linestyle"], label=name,
                    lw=1.1, markersize=3.5)
        nan_scales.update(int(s) for s in scales_arr[~finite])

    ax_mse.set_xlabel(r"Scale $\tau$")
    ax_mse.set_ylabel("SampEn")
    top_strip(ax_mse,
              f"MSE   bin={MSE_BIN_S*1e3:.0f} ms   "
              fr"$\tau_{{max}}$={MSE_SCALE}")
    panel_label(ax_mse, "c")
    ax_mse.legend(frameon=False, fontsize=6,
                  loc="upper right", borderpad=0.2,
                  handlelength=1.2, labelspacing=0.3)
    if nan_scales:
        ax_mse.text(0.97, 0.97,
                    f"SampEn undefined at $\\tau$={','.join(str(s) for s in sorted(nan_scales))}",
                    transform=ax_mse.transAxes,
                    ha="right", va="top", fontsize=5, color=p["muted"])

    fig.savefig(FIG_DIR / "manifold_pca_umap_mse.png",
                dpi=DPI_COMPOSITE, bbox_inches="tight")
    plt.close(fig)
    print("  -> manifold_pca_umap_mse.png")

    print(f"\nAll figures -> {FIG_DIR}")


if __name__ == "__main__":
    main()
