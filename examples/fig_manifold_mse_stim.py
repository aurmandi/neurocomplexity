"""Manifold + MSE on the drifting-gratings stimulus block of Allen 715093703.

Outputs to examples/figures/:
  stim_manifold_visp_pca.png       VISp PCA, colored by stimulus orientation
  stim_manifold_visp_umap.png      VISp UMAP, colored by orientation
  stim_manifold_ca1.png            CA1 PCA, colored by population rate
  stim_mse.png                     MSE across LGd / VISp / CA1
  stim_composite.png               4-panel composite
  spont_vs_stim_mse.png            MSE: spontaneous vs drifting-gratings overlay

Stimulus block: drifting_gratings_presentations, contiguous span.

Two manifold narratives (the canonical pair from the population-geometry lit):
  VISp = sensory-driven low-D ring shaped by orientation (Stringer/Carandini)
  CA1  = internally-organized rate manifold (Pastalkova/Buzsáki)
"""
from __future__ import annotations

import importlib.util
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
from neurocomplexity.analysis.manifold import bin_units
from neurocomplexity.core.provenance import ProvenanceRecord
from neurocomplexity.core.recording import SpikeRecording
from neurocomplexity.viz._palettes import series_styles
from neurocomplexity.viz._style import (
    apply_style, current_palette, panel_label, top_strip,
)

# reuse spont loader helpers
_spec = importlib.util.spec_from_file_location(
    "_sw1", str(Path(__file__).parent / "_manifold_sweep.py"))
_sw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sw)

NWB_PATH = _sw.NWB_PATH
FIG_DIR  = _sw.FIG_DIR
AREAS    = ("LGd", "VISp", "CA1")
DROP_AREAS = _sw.DROP_AREAS

# analysis params (matched to spont run)
MSE_BIN_S    = 0.025
MSE_SCALE    = 40
MANIFOLD_BIN = 0.20
SIGMA_MS     = 150.0
SOFT_C       = 5.0

DPI = 220


# ── stimulus discovery ────────────────────────────────────────────────────────

def load_drifting_gratings_window(path: Path):
    """Return (start_s, end_s, presentations_df) for the drifting-gratings
    block. We pick the longest contiguous run of presentations (gap < 5 s)."""
    import pynwb
    with pynwb.NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()
        if "drifting_gratings_presentations" not in nwb.intervals:
            raise RuntimeError("no drifting_gratings_presentations table in NWB")
        df = nwb.intervals["drifting_gratings_presentations"].to_dataframe()
    df = df.sort_values("start_time").reset_index(drop=True)
    # contiguous block: split on inter-trial gap > 5 s
    gaps = np.diff(df["start_time"].to_numpy())
    splits = np.where(gaps > 5.0)[0] + 1
    blocks = np.split(df.index.to_numpy(), splits)
    longest = max(blocks, key=len)
    sub = df.loc[longest].reset_index(drop=True)
    return float(sub["start_time"].iloc[0]), float(sub["stop_time"].iloc[-1]), sub


# ── loader (windowed, same shape as spont) ────────────────────────────────────

def load_window(path, start, end):
    import pynwb
    with pynwb.NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb  = io.read()
        u    = nwb.units
        uids = np.asarray(u.id[:], dtype=np.int64)
        meta = {"id": uids}
        for c in list(u.colnames):
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
        mask = ((units_df["quality"] == "good").to_numpy()
                if "quality" in units_df.columns
                else np.ones(len(units_df), dtype=bool))
        keep = np.where(mask)[0]
        times_l, owners_l, kept = [], [], []
        for oi in keep:
            st  = np.asarray(u["spike_times"][int(oi)], dtype=np.float64)
            sel = (st >= start) & (st < end)
            if not sel.any():
                continue
            tt = st[sel] - start
            uv = int(uids[oi])
            times_l.append(tt); owners_l.append(np.full(tt.size, uv, np.int64))
            kept.append(uv)
        kdf = units_df.iloc[keep].reset_index(drop=True)
        act = kdf["id"].isin(kept).to_numpy()
        kdf = kdf[act].reset_index(drop=True)
        if "peak_channel_id" in kdf.columns and nwb.electrodes is not None:
            try:
                elec = nwb.electrodes.to_dataframe()
                if "location" in elec.columns:
                    kdf["brain_area"] = kdf["peak_channel_id"].map(
                        elec["location"].to_dict())
            except Exception:
                pass
        sa  = np.concatenate(times_l)  if times_l  else np.empty(0)
        ua  = np.concatenate(owners_l) if owners_l else np.empty(0, np.int64)
        o   = np.argsort(sa, kind="stable")
        prov = ProvenanceRecord.for_file(path, source_format="nwb-window")
    return SpikeRecording(
        spike_times=sa[o], unit_ids=ua[o], units=kdf,
        populations={"all": np.ones(len(kdf), dtype=bool)},
        duration=float(end - start), sampling_rate=None, source=prov,
    )


def area_subrec(rec, areas):
    rec_all = rec.with_populations(by="brain_area")
    pops = {k: v for k, v in rec_all.populations.items()
            if k in areas and k.lower() not in DROP_AREAS}
    return replace(rec_all, populations=pops)


def per_bin_orientation(pres_df, t0, bin_s, T):
    """Vector of length T with orientation (deg) per bin; NaN for blanks."""
    out = np.full(T, np.nan, dtype=np.float64)
    if "orientation" not in pres_df.columns:
        return out
    for _, row in pres_df.iterrows():
        try:
            ori = float(row["orientation"])
        except Exception:
            continue
        if not np.isfinite(ori):
            continue
        s = (float(row["start_time"]) - t0) / bin_s
        e = (float(row["stop_time"])  - t0) / bin_s
        b0 = int(np.floor(max(s, 0)))
        b1 = int(np.ceil(min(e, T)))
        if b1 > b0:
            out[b0:b1] = ori
    return out


def pop_rate(rec, pops, bin_s):
    uid_all = []
    for pn in pops:
        m = rec.populations[pn]
        uid_all.append(rec.units["id"].to_numpy()[m])
    uid_all = np.concatenate(uid_all)
    counts = bin_units(rec, bin_s, uid_all)
    return counts.sum(axis=1)


# ── manifold draw helpers ─────────────────────────────────────────────────────

def _percentile_clip(ax, coords):
    for col_i, set_lim in ((0, ax.set_xlim), (1, ax.set_ylim)):
        lo, hi = np.percentile(coords[:, col_i], [1, 99])
        pad = 0.06 * (hi - lo) if hi > lo else 0.1
        set_lim(lo - pad, hi + pad)


def draw_ori_manifold(ax, coords, ori_per_bin, method_label, n_units, ev=None):
    """Color by stimulus orientation using a cyclic colormap (HSV)."""
    p = current_palette()
    finite = np.isfinite(ori_per_bin)
    # blanks first (grey, behind)
    if (~finite).any():
        ax.scatter(coords[~finite, 0], coords[~finite, 1],
                   c=p["muted"], s=4, edgecolor="none",
                   alpha=0.35, rasterized=True, zorder=1)
    sc = None
    if finite.any():
        sc = ax.scatter(coords[finite, 0], coords[finite, 1],
                        c=ori_per_bin[finite] % 360.0, cmap="hsv",
                        vmin=0, vmax=360, s=8, edgecolor="none",
                        rasterized=True, zorder=2)
    _percentile_clip(ax, coords)
    ax.set_xlabel(f"{method_label}-1")
    ax.set_ylabel(f"{method_label}-2")
    if ev is not None:
        pct = " / ".join(f"{100*v:.1f}%" for v in ev)
        top_strip(ax, f"VISp  {method_label}  n={n_units}  {pct} var")
    else:
        top_strip(ax, f"VISp  {method_label}  n={n_units}")
    return sc


def draw_rate_manifold(ax, coords, rate, method_label, n_units, ev=None,
                       label="CA1"):
    p = current_palette()
    cmap = LinearSegmentedColormap.from_list(
        "nc_rate", [p["muted"], p["signal"], p["accent"]])
    vlo, vhi = np.percentile(rate, [2, 98])
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=rate, cmap=cmap,
                    s=7, edgecolor="none", vmin=vlo, vmax=vhi,
                    rasterized=True)
    _percentile_clip(ax, coords)
    ax.set_xlabel(f"{method_label}-1")
    ax.set_ylabel(f"{method_label}-2")
    if ev is not None:
        pct = " / ".join(f"{100*v:.1f}%" for v in ev)
        top_strip(ax, f"{label}  {method_label}  n={n_units}  {pct} var")
    else:
        top_strip(ax, f"{label}  {method_label}  n={n_units}")
    return sc


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    apply_style()
    p = current_palette()

    print("locating drifting_gratings block ...")
    start, end, pres = load_drifting_gratings_window(NWB_PATH)
    print(f"  block: [{start:.1f}, {end:.1f}] s  duration={end-start:.1f}s  "
          f"trials={len(pres)}")
    print(f"  orientations: {sorted(pres['orientation'].dropna().unique().tolist())}")

    print("loading spikes in window ...")
    rec = load_window(NWB_PATH, start, end)
    print(f"  units={len(rec.units)}  spikes={rec.spike_times.size:,}")

    rec_a = area_subrec(rec, AREAS)
    n_per = {k: int(v.sum()) for k, v in rec_a.populations.items()}
    print(f"  populations: {n_per}")

    # ── VISp PCA + UMAP, colored by orientation ──────────────────────────────
    print("VISp PCA ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mf_visp_pca = nc.analysis.manifold(
            rec_a, populations=["VISp"], method="pca", dims=2,
            bin_size_s=MANIFOLD_BIN, sigma_ms=SIGMA_MS,
            normalize="soft", soft_norm_constant=SOFT_C,
        )
    print("VISp UMAP ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mf_visp_umap = nc.analysis.manifold(
            rec_a, populations=["VISp"], method="umap", dims=2,
            bin_size_s=MANIFOLD_BIN, sigma_ms=SIGMA_MS,
            normalize="soft", soft_norm_constant=SOFT_C,
            random_state=0,
        )

    T_v = mf_visp_pca.coords.shape[0]
    ori_v = per_bin_orientation(pres, start, MANIFOLD_BIN, T_v)

    # ── CA1 PCA, colored by population rate ──────────────────────────────────
    print("CA1 PCA ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mf_ca1_pca = nc.analysis.manifold(
            rec_a, populations=["CA1"], method="pca", dims=2,
            bin_size_s=MANIFOLD_BIN, sigma_ms=SIGMA_MS,
            normalize="soft", soft_norm_constant=SOFT_C,
        )
    rate_ca1 = pop_rate(rec_a, ["CA1"], MANIFOLD_BIN)
    rate_ca1 = rate_ca1[:mf_ca1_pca.coords.shape[0]].astype(float)

    # ── MSE all areas ────────────────────────────────────────────────────────
    print(f"MSE (bin={MSE_BIN_S*1e3:.0f}ms scale_max={MSE_SCALE}) ...")
    pops = list(rec_a.populations.keys())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mse_res = nc.analysis.multiscale_entropy(
            rec_a, populations=pops,
            bin_size_s=MSE_BIN_S, scale_max=MSE_SCALE,
        )

    # ── standalone panels ────────────────────────────────────────────────────
    def _save_standalone(fname, draw):
        fig, ax = plt.subplots(figsize=(3.6, 3.3), constrained_layout=True)
        draw(ax)
        fig.savefig(FIG_DIR / fname, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {fname}")

    def _visp_pca(ax):
        sc = draw_ori_manifold(ax, mf_visp_pca.coords, ori_v, "PCA",
                               mf_visp_pca.n_units,
                               mf_visp_pca.explained_variance_ratio)
        if sc is not None:
            cb = ax.figure.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
            cb.set_label("orientation (deg)", fontsize=6)
            cb.set_ticks([0, 90, 180, 270, 360])
            cb.ax.tick_params(labelsize=6, length=2, width=0.5)
            cb.outline.set_linewidth(0.5)
    _save_standalone("stim_manifold_visp_pca.png", _visp_pca)

    def _visp_umap(ax):
        sc = draw_ori_manifold(ax, mf_visp_umap.coords, ori_v, "UMAP",
                               mf_visp_umap.n_units)
        if sc is not None:
            cb = ax.figure.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
            cb.set_label("orientation (deg)", fontsize=6)
            cb.set_ticks([0, 90, 180, 270, 360])
            cb.ax.tick_params(labelsize=6, length=2, width=0.5)
            cb.outline.set_linewidth(0.5)
    _save_standalone("stim_manifold_visp_umap.png", _visp_umap)

    def _ca1(ax):
        sc = draw_rate_manifold(ax, mf_ca1_pca.coords, rate_ca1, "PCA",
                                mf_ca1_pca.n_units,
                                mf_ca1_pca.explained_variance_ratio, "CA1")
        cb = ax.figure.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
        cb.set_label("pop. spikes / bin", fontsize=6)
        cb.ax.tick_params(labelsize=6, length=2, width=0.5)
        cb.outline.set_linewidth(0.5)
    _save_standalone("stim_manifold_ca1.png", _ca1)

    def _mse(ax):
        styles = series_styles(len(mse_res.populations))
        scales_arr = np.asarray(mse_res.scales)
        nan_scales: set = set()
        for pi, name in enumerate(mse_res.populations):
            y = np.asarray(mse_res.sampen[pi], dtype=float)
            st_ = styles[pi]
            finite = np.isfinite(y)
            ax.plot(scales_arr, np.where(finite, y, np.nan),
                    color=st_["color"], marker=st_["marker"],
                    linestyle=st_["linestyle"], label=name,
                    lw=1.1, markersize=3.5)
            nan_scales.update(int(s) for s in scales_arr[~finite])
        ax.set_xlabel(r"Scale $\tau$"); ax.set_ylabel("SampEn")
        top_strip(ax, f"MSE  bin={MSE_BIN_S*1e3:.0f} ms  "
                       fr"$\tau_{{max}}$={MSE_SCALE}")
        ax.legend(frameon=False, fontsize=6, loc="upper right",
                  borderpad=0.2, handlelength=1.2, labelspacing=0.3)
    _save_standalone("stim_mse.png", _mse)

    # ── composite 4-panel ────────────────────────────────────────────────────
    print("composite ...")
    fig, axes = plt.subplots(2, 2, figsize=(7.09, 6.0))
    fig.subplots_adjust(left=0.08, right=0.90, top=0.94, bottom=0.08,
                        wspace=0.40, hspace=0.42)
    (ax_a, ax_b), (ax_c, ax_d) = axes

    sc_a = draw_ori_manifold(ax_a, mf_visp_pca.coords, ori_v, "PCA",
                             mf_visp_pca.n_units,
                             mf_visp_pca.explained_variance_ratio)
    panel_label(ax_a, "a")
    if sc_a is not None:
        pos = ax_a.get_position()
        cax = fig.add_axes([pos.x1 + 0.012, pos.y0, 0.010, pos.height])
        cb = fig.colorbar(sc_a, cax=cax)
        cb.set_label("ori. (°)", fontsize=6, labelpad=3)
        cb.set_ticks([0, 90, 180, 270, 360])
        cb.ax.tick_params(labelsize=6, length=2, width=0.5)
        cb.outline.set_linewidth(0.5)

    sc_b = draw_ori_manifold(ax_b, mf_visp_umap.coords, ori_v, "UMAP",
                             mf_visp_umap.n_units)
    panel_label(ax_b, "b")
    if sc_b is not None:
        pos = ax_b.get_position()
        cax = fig.add_axes([pos.x1 + 0.012, pos.y0, 0.010, pos.height])
        cb = fig.colorbar(sc_b, cax=cax)
        cb.set_label("ori. (°)", fontsize=6, labelpad=3)
        cb.set_ticks([0, 90, 180, 270, 360])
        cb.ax.tick_params(labelsize=6, length=2, width=0.5)
        cb.outline.set_linewidth(0.5)

    sc_c = draw_rate_manifold(ax_c, mf_ca1_pca.coords, rate_ca1, "PCA",
                              mf_ca1_pca.n_units,
                              mf_ca1_pca.explained_variance_ratio, "CA1")
    panel_label(ax_c, "c")
    pos = ax_c.get_position()
    cax = fig.add_axes([pos.x1 + 0.012, pos.y0, 0.010, pos.height])
    cb = fig.colorbar(sc_c, cax=cax)
    cb.set_label("spikes/bin", fontsize=6, labelpad=3)
    cb.ax.tick_params(labelsize=6, length=2, width=0.5)
    cb.outline.set_linewidth(0.5)

    # panel d — MSE
    styles = series_styles(len(mse_res.populations))
    scales_arr = np.asarray(mse_res.scales)
    for pi, name in enumerate(mse_res.populations):
        y = np.asarray(mse_res.sampen[pi], dtype=float)
        st_ = styles[pi]
        finite = np.isfinite(y)
        ax_d.plot(scales_arr, np.where(finite, y, np.nan),
                  color=st_["color"], marker=st_["marker"],
                  linestyle=st_["linestyle"], label=name,
                  lw=1.1, markersize=3.5)
    ax_d.set_xlabel(r"Scale $\tau$"); ax_d.set_ylabel("SampEn")
    top_strip(ax_d, f"MSE  bin={MSE_BIN_S*1e3:.0f} ms")
    ax_d.legend(frameon=False, fontsize=6, loc="upper right",
                borderpad=0.2, handlelength=1.2, labelspacing=0.3)
    panel_label(ax_d, "d")

    fig.suptitle(
        f"Allen 715093703 — drifting gratings  "
        f"[{start:.0f}, {end:.0f}] s  ({end-start:.0f} s, {len(pres)} trials)",
        fontsize=8, y=0.99,
    )
    fig.savefig(FIG_DIR / "stim_composite.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  -> stim_composite.png")

    # ── spontaneous-vs-stimulus MSE overlay ─────────────────────────────────
    print("loading spontaneous window for MSE compare ...")
    spont_rec = load_window(NWB_PATH, _sw.WINDOW_START, _sw.WINDOW_END)
    spont_a   = area_subrec(spont_rec, AREAS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mse_spont = nc.analysis.multiscale_entropy(
            spont_a, populations=list(spont_a.populations.keys()),
            bin_size_s=MSE_BIN_S, scale_max=MSE_SCALE,
        )

    fig2, ax = plt.subplots(figsize=(5.0, 3.6), constrained_layout=True)
    styles = series_styles(len(mse_res.populations))
    for pi, name in enumerate(mse_res.populations):
        st_ = styles[pi]
        ax.plot(mse_res.scales, mse_res.sampen[pi],
                color=st_["color"], marker=st_["marker"], linestyle="-",
                lw=1.2, markersize=3.5, label=f"{name} stim")
    # spontaneous overlay (dashed, same colors)
    sp_map = {n: i for i, n in enumerate(mse_spont.populations)}
    for pi, name in enumerate(mse_res.populations):
        if name not in sp_map:
            continue
        spi = sp_map[name]
        st_ = styles[pi]
        ax.plot(mse_spont.scales, mse_spont.sampen[spi],
                color=st_["color"], marker=st_["marker"], linestyle="--",
                lw=1.0, markersize=3.0, alpha=0.85, label=f"{name} spont")
    ax.set_xlabel(r"Scale $\tau$"); ax.set_ylabel("SampEn")
    top_strip(ax, f"MSE  stim (solid) vs spontaneous (dashed)  "
                  f"bin={MSE_BIN_S*1e3:.0f} ms")
    ax.legend(frameon=False, fontsize=5.5, loc="upper right", ncol=2,
              handlelength=1.4, labelspacing=0.3)
    fig2.savefig(FIG_DIR / "spont_vs_stim_mse.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig2)
    print("  -> spont_vs_stim_mse.png")

    print(f"\nAll figures -> {FIG_DIR}")


if __name__ == "__main__":
    main()
