"""Neurocomplexity integration on a clean SPONTANEOUS window of Allen 715093703.

Window choice (scientific rationale):
  Allen Brain Observatory session 715093703 has three ~300 s gray-screen
  spontaneous blocks. Two are contaminated by `invalid_times` artifact
  intervals; one is clean:

      spontaneous block id6 : [3765.64, 4066.89] s  (301.25 s, clean)

  Spontaneous (stimulus-free) activity is the canonical substrate for
  criticality / avalanche analysis (Beggs & Plenz 2003; Fontenele et al.
  2019), so this window is the right epoch for the size/lifetime/branching/
  shape-collapse statistics. All good ('quality == good') units are kept;
  the unassigned 'grey' anatomical label is dropped.

Memory-capped: reads ONLY this window via a windowed pynwb loader, so RSS
stays small even though the full session holds 1.1e8 spikes.
"""
from __future__ import annotations

import gc
import time
import warnings
from dataclasses import replace
from pathlib import Path
from traceback import format_exc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import neurocomplexity as nc
from neurocomplexity.core.provenance import ProvenanceRecord
from neurocomplexity.core.recording import SpikeRecording

try:
    import psutil
    _PROC = psutil.Process()
except ImportError:
    _PROC = None

NWB_PATH = Path(r"C:\Users\Sazgar\OneDrive\Desktop\Arman_Dinarvand code sample"
                r"\neuropixel\NeuropixelVisCodingData_cache"
                r"\session_715093703\session_715093703.nwb")
OUT_DIR = Path(__file__).parent.parent / "outputs" / "integration_session_715093703_spont"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Clean spontaneous block id6, trimmed 0.5 s at each edge.
WINDOW_START = 3766.14
WINDOW_END = 4066.39
MAX_UNITS = 2200          # keep all good units
DROP_AREAS = {"grey", "", "nan", "None"}
SURROGATE_N = 5000   # n=5000 -> p-floor 2e-4; clears BH-FDR over ~150 tests
BOOTSTRAP_N = 50

results: list[tuple[str, str, float, str]] = []


def _mem_mb() -> float:
    return _PROC.memory_info().rss / 1e6 if _PROC else float("nan")


def step(name):
    def decorator(fn):
        def wrap():
            mem0 = _mem_mb()
            print(f"[..] {name}  (mem={mem0:.0f}MB)", flush=True)
            t0 = time.perf_counter()
            try:
                info = fn() or ""
                dt = time.perf_counter() - t0
                print(f"[OK] {name}  ({dt:.1f}s) mem={_mem_mb():.0f}MB  {info}",
                      flush=True)
                results.append((name, "PASS", dt, info))
            except Exception as exc:
                dt = time.perf_counter() - t0
                print(f"[FAIL] {name}  ({dt:.1f}s)", flush=True)
                print(format_exc(limit=3), flush=True)
                results.append((name, "FAIL", dt, type(exc).__name__))
        return wrap
    return decorator


def load_window(path, start, end, max_units, *, seed=0):
    import pynwb
    rng = np.random.default_rng(seed)
    with pynwb.NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()
        u = nwb.units
        unit_ids = np.asarray(u.id[:], dtype=np.int64)
        cols = list(u.colnames)
        meta = {"id": unit_ids}
        for c in cols:
            if c in ("spike_times", "spike_amplitudes", "waveform_mean"):
                continue
            try:
                vals = u[c][:]
                if isinstance(vals, np.ndarray) and vals.ndim == 1 and len(vals) == len(unit_ids):
                    meta[c] = vals
            except Exception:
                pass
        units_df = pd.DataFrame(meta)

        if "quality" in units_df.columns:
            mask = (units_df["quality"] == "good").to_numpy()
        else:
            mask = np.ones(len(units_df), dtype=bool)
        keep_indices = np.where(mask)[0]
        if keep_indices.size > max_units:
            choose = rng.choice(keep_indices.size, size=max_units, replace=False)
            choose.sort()
            keep_indices = keep_indices[choose]

        all_times, all_owners, kept_ids = [], [], []
        for orig_i in keep_indices:
            st = np.asarray(u["spike_times"][int(orig_i)], dtype=np.float64)
            sel = (st >= start) & (st < end)
            if not sel.any():
                continue
            tt = st[sel] - start
            uid_val = int(unit_ids[orig_i])
            all_times.append(tt)
            all_owners.append(np.full(tt.size, uid_val, dtype=np.int64))
            kept_ids.append(uid_val)

        kept_df = units_df.iloc[keep_indices].reset_index(drop=True)
        active = kept_df["id"].isin(kept_ids).to_numpy()
        units_df_final = kept_df[active].reset_index(drop=True)

        if "peak_channel_id" in units_df_final.columns and nwb.electrodes is not None:
            try:
                elec = nwb.electrodes.to_dataframe()
                if "location" in elec.columns:
                    loc_map = elec["location"].to_dict()
                    units_df_final["brain_area"] = units_df_final["peak_channel_id"].map(loc_map)
            except Exception:
                pass

        st_arr = np.concatenate(all_times) if all_times else np.empty(0)
        uid_arr = np.concatenate(all_owners) if all_owners else np.empty(0, dtype=np.int64)
        order = np.argsort(st_arr, kind="stable")
        prov = ProvenanceRecord.for_file(path, source_format="nwb-window")

    return SpikeRecording(
        spike_times=st_arr[order],
        unit_ids=uid_arr[order],
        units=units_df_final,
        populations={"all": np.ones(len(units_df_final), dtype=bool)},
        duration=float(end - start),
        sampling_rate=None,
        source=prov,
    )


class State:
    rec = None


@step(f"load spontaneous window [{WINDOW_START:.0f},{WINDOW_END:.0f}]s")
def load_step():
    State.rec = load_window(NWB_PATH, WINDOW_START, WINDOW_END, MAX_UNITS)
    gc.collect()
    r = State.rec
    rate = r.spike_times.size / max(len(r.units), 1) / r.duration
    return (f"units={len(r.units)} spikes={r.spike_times.size:,} "
            f"mean_rate={rate:.2f}Hz/unit")


@step("build brain-area populations (drop 'grey')")
def build_pops():
    r = State.rec
    if "brain_area" in r.units.columns:
        ba = r.units["brain_area"].astype(str)
        good_area = ~ba.str.lower().isin({a.lower() for a in DROP_AREAS}) & ba.notna()
        counts = ba[good_area].value_counts()
        top = list(counts.head(4).index)
        State.rec = r.with_populations(by="brain_area")
        new_pops = {k: v for k, v in State.rec.populations.items() if k in top}
        if new_pops:
            State.rec = replace(State.rec, populations=new_pops)
        print(f"     area counts: {counts.head(8).to_dict()}")
    return f"populations={list(State.rec.populations.keys())}"


@step("stationarity")
def stat_step():
    s = nc.analysis.stationarity(State.rec, window_s=30.0)
    return f"is_stationary={s.is_stationary} flags={s.flags}"


@step("criticality")
def crit_step():
    pops = list(State.rec.populations.keys())[:1]
    State.crit = nc.analysis.criticality(State.rec, populations=pops,
                                          bin_size=(2, 4, 8, 16))
    return (f"pop={pops[0]} alpha_s={State.crit.alpha_s:.3f} "
            f"alpha_t={State.crit.alpha_t:.3f} R2={State.crit.r_squared:.3f}")


@step("branching (Wilting MR)")
def br_step():
    pops = list(State.rec.populations.keys())[:1]
    State.br = nc.analysis.wilting_mr(State.rec, populations=pops,
                                       bin_size_ms=4.0, k_max=20)
    return f"m={State.br.m:.4f} R2={State.br.r_squared:.3f}"


@step("transfer_entropy (per-unit, top-K x 3 areas)")
def te_step():
    # Per-unit TE on top-K firing units pulled from THREE anatomically
    # connected areas (LGd -> VISp visual stream + CA1 hippocampus).
    # Cross-area pairs carry the canonical directed flow in spontaneous
    # Allen Neuropixels recordings (LGd -> VISp; Siegle 2021 *Nature*),
    # so a multi-area unit selection both (a) preserves the field-standard
    # per-unit granularity (Shimono 2015; Timme 2016) and (b) populates
    # the TE matrix with biologically expected edges rather than relying
    # on within-CA1 monosynaptic detection alone. Active-rate floor (>0.5
    # Hz) avoids units whose 5-ms bins are nearly all zeros.
    AREAS = ("LGd", "VISp", "CA1")
    K_PER_AREA = 5
    MIN_RATE_HZ = 1.0
    MAX_RATE_HZ = 25.0  # avoid 5-ms-bin saturation on fast LGd relays
    PREFIX = {"LGd": "L", "VISp": "V", "CA1": "C"}

    selected: list[tuple[str, int]] = []
    rate_log: list[float] = []
    for area in AREAS:
        try:
            a_rec = State.rec.filter_units(query=f"brain_area == '{area}'")
        except Exception:
            continue
        if len(a_rec.units) == 0:
            continue
        counts = np.bincount(a_rec.unit_ids,
                             minlength=int(a_rec.unit_ids.max()) + 1)
        uid_arr = a_rec.units["id"].to_numpy(dtype=np.int64)
        rates = counts[uid_arr] / a_rec.duration
        order = np.argsort(rates)[::-1]
        kept = 0
        for idx in order:
            r = float(rates[idx])
            if r < MIN_RATE_HZ:
                break
            if r > MAX_RATE_HZ:
                continue
            selected.append((area, int(uid_arr[idx])))
            rate_log.append(r)
            kept += 1
            if kept >= K_PER_AREA:
                break

    if len(selected) < 4:
        raise RuntimeError(f"too few qualifying units ({len(selected)})")

    all_uids = [uid for _, uid in selected]
    sub = State.rec.filter_units(query=f"id in {all_uids}")
    counter = {a: 0 for a in AREAS}
    defn: dict[str, np.ndarray] = {}
    uid_col = sub.units["id"].to_numpy()
    for area, uid in selected:
        name = f"{PREFIX[area]}{counter[area]}"
        counter[area] += 1
        defn[name] = (uid_col == uid)
    sub = sub.with_populations(definition=defn, on_unassigned="drop")
    State.rec_te = sub
    State.te_areas = {n: a for n, (a, _) in zip(defn.keys(), selected)}

    # TE at 20 ms lag (monosynaptic + di-synaptic window, intra- + inter-area).
    State.te = nc.analysis.transfer_entropy(
        sub, populations=list(defn.keys()),
        bin_size_ms=5.0, delay_bins=4,
    )
    by_area = {a: counter[a] for a in AREAS}
    return (f"shape={State.te.matrix.shape} max={State.te.matrix.max():.5f} "
            f"units_per_area={by_area} mean_rate={np.mean(rate_log):.2f}Hz")


@step("dimensionality")
def dim_step():
    pops = list(State.rec.populations.keys())
    State.dim = nc.analysis.dimensionality(State.rec, populations=pops,
                                            bin_size_ms=20.0)
    return f"PR={State.dim.participation_ratio:.3f} N={State.dim.n_units}"


@step("shape_collapse")
def shape_step():
    pops = list(State.rec.populations.keys())[:1]
    State.shape = nc.analysis.shape_collapse(State.rec, populations=pops,
                                              bin_size_ms=4.0)
    return f"gamma={State.shape.gamma:.3f} resid={State.shape.residual:.3g}"


@step("partial_information (PID)")
def pid_step():
    pops = list(State.rec.populations.keys())
    if len(pops) < 3:
        return "skip: <3 pops"
    State.pid = nc.analysis.partial_information(
        State.rec, target_pop=pops[0], sources=(pops[1], pops[2]),
        bin_size_ms=10.0, n_levels=2)
    return (f"R={State.pid.redundancy:.4f} U1={State.pid.unique_1:.4f} "
            f"U2={State.pid.unique_2:.4f} S={State.pid.synergy:.4f}")


@step(f"null_test TE (n={SURROGATE_N})")
def null_step():
    # Use the tiny CA1-only recording (State.rec_te) for surrogate
    # generation — spike_dither scales with total spike count, so this is
    # ~100x faster than dithering the full 3.1M-spike recording.
    # isi_shuffle is the package-recommended TE null (see docs/quickstart.md):
    # destroys cross-unit timing while preserving each unit's ISI distribution
    # exactly, so a positive TE cannot be explained by per-unit rate or
    # burstiness alone (Shimono & Beggs 2015).
    State.null = nc.inference.test(State.te, State.rec_te,
                                    surrogate="isi_shuffle",
                                    n=SURROGATE_N, seed=0, fdr=True)
    p_fdr = State.null.p_value_fdr
    n_sig = int(np.sum(np.asarray(p_fdr) < 0.05)) if p_fdr is not None else 0
    n_tot = np.asarray(p_fdr).size if p_fdr is not None else 0
    return f"sig edges (p_fdr<0.05): {n_sig}/{n_tot}"


@step(f"bootstrap branching (n={BOOTSTRAP_N})")
def boot_step():
    State.boot = nc.inference.bootstrap(State.br, State.rec,
                                         n=BOOTSTRAP_N, seed=0)
    return f"CI=({State.boot.ci_lower:.3f},{State.boot.ci_upper:.3f})"


def _save(fig, name):
    p = OUT_DIR / f"{name}.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p.name


@step("figures: Fig 3 + Fig 4 panels")
def figs():
    # Uniform figsize so panels tile evenly inside the LaTeX subfigure grid.
    FIG3_SIZE = (5.4, 2.6)
    FIG4_SIZE = (5.4, 3.6)
    _save(nc.viz.figure_branching(State.br), "branching")
    _save(nc.viz.figure_bootstrap(State.boot, figsize=FIG3_SIZE),
          "branching_bootstrap")
    _save(nc.viz.figure_criticality(State.crit, figsize=FIG3_SIZE),
          "criticality")
    _save(nc.viz.figure_shape_collapse(State.shape, figsize=FIG3_SIZE),
          "shape_collapse")
    _save(nc.viz.figure_dimensionality(State.dim, figsize=FIG3_SIZE),
          "dimensionality")
    pops = list(State.te.populations)
    _save(nc.viz.figure_significance_matrix(State.null, row_labels=pops,
                                            col_labels=pops,
                                            alpha=0.10,
                                            figsize=FIG4_SIZE),
          "te_significance_matrix")
    _save(nc.viz.figure_volcano(State.null, alpha=0.10,
                                figsize=FIG4_SIZE), "te_volcano")
    _save(nc.viz.figure_te_network(State.te, State.null, alpha=0.10,
                                    figsize=FIG4_SIZE), "te_network")
    if hasattr(State, "pid"):
        _save(nc.viz.figure_pid(State.pid, figsize=FIG4_SIZE), "pid")
    # Pickle state so figures can be regenerated cheaply without re-running
    # the slow null-test / bootstrap (which dominate runtime).
    import pickle
    state_dict = {k: getattr(State, k) for k in
                  ("crit", "br", "boot", "shape", "dim", "te", "null", "pid")
                  if hasattr(State, k)}
    state_dict["pops"] = pops
    with open(OUT_DIR / "_state.pkl", "wb") as fh:
        pickle.dump(state_dict, fh)
    return f"wrote PNGs + _state.pkl -> {OUT_DIR.name}"


def main():
    warnings.simplefilter("default")
    print(f"Loading spontaneous window from {NWB_PATH.name}")
    print(f"Window [{WINDOW_START}, {WINDOW_END}] s  ({WINDOW_END-WINDOW_START:.1f} s)")
    print(f"Outputs -> {OUT_DIR}")
    if _PROC is not None:
        vm = psutil.virtual_memory()
        print(f"System RAM: total={vm.total/1e9:.1f}GB available={vm.available/1e9:.1f}GB")
    print(flush=True)

    load_step(); build_pops(); stat_step()
    crit_step(); br_step(); te_step(); dim_step(); shape_step(); pid_step()
    null_step(); boot_step(); figs()

    n_pass = sum(1 for r in results if r[1] == "PASS")
    n_fail = sum(1 for r in results if r[1] == "FAIL")
    print("\n" + "=" * 70)
    print(f"SUMMARY: {n_pass} PASS / {n_fail} FAIL / {len(results)} total")
    for name, status, dt, info in results:
        if status == "FAIL":
            print(f"  FAIL {name}: {info}")
    print("=" * 70)


if __name__ == "__main__":
    main()
