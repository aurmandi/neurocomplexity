"""Light-weight end-to-end neurocomplexity test on Allen NWB session 715093703.

Memory-capped variant: uses a custom pynwb loader that reads ONLY the
requested time window for ONLY a 250-unit random subset. Total RSS stays
under ~700 MB on an 8 GB laptop.
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
OUT_DIR = Path(__file__).parent.parent / "outputs" / "integration_session_715093703_light"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuration: keep RSS under ~700 MB.
WINDOW_START = 1000.0    # seconds into session (after settling)
WINDOW_DUR = 900.0       # 15 minutes
MAX_UNITS = 250
SUBSAMPLE_SEED = 0
SURROGATE_N = 30         # smaller null/bootstrap pools to stay light
BOOTSTRAP_N = 20

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
                mem1 = _mem_mb()
                print(f"[OK] {name}  ({dt:.1f}s) mem={mem1:.0f}MB (d{mem1-mem0:+.0f})  {info}",
                      flush=True)
                results.append((name, "PASS", dt, info))
            except Exception as exc:
                dt = time.perf_counter() - t0
                print(f"[FAIL] {name}  ({dt:.1f}s) mem={_mem_mb():.0f}MB", flush=True)
                print(format_exc(limit=3), flush=True)
                results.append((name, "FAIL", dt, type(exc).__name__))
        return wrap
    return decorator


# ---------------------------------------------------------------------------
# Light-weight NWB window loader (bypasses full from_nwb materialization).
# ---------------------------------------------------------------------------

def load_window(path, start, end, max_units, *, quality_filter=True, seed=0):
    """Read only [start, end] window + max_units subset directly via pynwb.

    Avoids the full-session memory blowup of `nc.io.from_nwb`.
    """
    import pynwb
    rng = np.random.default_rng(seed)
    with pynwb.NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()
        u = nwb.units
        unit_ids = np.asarray(u.id[:], dtype=np.int64)

        # Read scalar columns
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

        # Quality filter
        if quality_filter and "quality" in units_df.columns:
            mask = (units_df["quality"] == "good").to_numpy()
        else:
            mask = np.ones(len(units_df), dtype=bool)
        keep_indices = np.where(mask)[0]

        # Subsample
        if keep_indices.size > max_units:
            choose = rng.choice(keep_indices.size, size=max_units, replace=False)
            choose.sort()
            keep_indices = keep_indices[choose]

        # Read only the spike times within window per kept unit
        all_times: list[np.ndarray] = []
        all_owners: list[np.ndarray] = []
        kept_ids: list[int] = []
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

        # Slim units_df to actually-spiking kept units (preserve original order
        # in keep_indices but drop empties)
        kept_df = units_df.iloc[keep_indices].reset_index(drop=True)
        active_mask = kept_df["id"].isin(kept_ids).to_numpy()
        units_df_final = kept_df[active_mask].reset_index(drop=True)

        # brain_area join
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


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

class State:
    rec = None


@step(f"load window [{WINDOW_START:.0f},{WINDOW_START+WINDOW_DUR:.0f}]s + subset {MAX_UNITS} units")
def load_step():
    State.rec = load_window(NWB_PATH, WINDOW_START, WINDOW_START + WINDOW_DUR,
                            max_units=MAX_UNITS, quality_filter=True,
                            seed=SUBSAMPLE_SEED)
    gc.collect()
    r = State.rec
    return f"units={len(r.units)} spikes={r.spike_times.size:,}"


@step("build brain-area populations")
def build_pops():
    r = State.rec
    if "brain_area" in r.units.columns:
        State.rec = r.with_populations(by="brain_area")
        # Keep only top-4 most-populated areas to bound TE matrix size
        counts = r.units["brain_area"].value_counts().head(4)
        top = list(counts.index)
        new_pops = {k: v for k, v in State.rec.populations.items() if k in top}
        if new_pops:
            State.rec = replace(State.rec, populations=new_pops)
    return f"populations={list(State.rec.populations.keys())}"


@step("estimate_bin_spikes_bytes")
def estimate_step():
    b = nc.estimate_bin_spikes_bytes(State.rec, list(State.rec.populations.keys()),
                                     bin_size_ms=5.0)
    return f"~{b/1e6:.1f} MB"


@step("stationarity")
def stat_step():
    s = nc.analysis.stationarity(State.rec, window_s=60.0)
    return f"is_stationary={s.is_stationary} flags={s.flags}"


@step("criticality")
def crit_step():
    pops = list(State.rec.populations.keys())[:1]
    State.crit = nc.analysis.criticality(State.rec, populations=pops,
                                          bin_size_ms=(2, 4, 8, 16))
    return f"alpha_s={State.crit.alpha_s:.3f} alpha_t={State.crit.alpha_t:.3f}"


@step("branching (Wilting MR)")
def br_step():
    pops = list(State.rec.populations.keys())[:1]
    State.br = nc.analysis.wilting_mr(State.rec, populations=pops,
                                       bin_size_ms=4.0, k_max=20)
    return f"m={State.br.m:.4f}"


@step("transfer_entropy")
def te_step():
    pops = list(State.rec.populations.keys())
    State.te = nc.analysis.transfer_entropy(State.rec, populations=pops,
                                             bin_size_ms=5.0, delay_bins=1)
    return f"shape={State.te.matrix.shape} max={State.te.matrix.max():.4f}"


@step("autonomy")
def auto_step():
    pops = list(State.rec.populations.keys())
    if len(pops) < 2:
        return "skip: <2 pops"
    State.auto = nc.analysis.autonomy(State.rec, populations=pops,
                                       bin_size_ms=5.0, max_lag=3)
    return f"keys={list(State.auto.values.keys())}"


@step("dimensionality")
def dim_step():
    pops = list(State.rec.populations.keys())
    State.dim = nc.analysis.dimensionality(State.rec, populations=pops,
                                            bin_size_ms=20.0)
    return f"PR={State.dim.participation_ratio:.3f}"


@step("shape_collapse")
def shape_step():
    pops = list(State.rec.populations.keys())[:1]
    State.shape = nc.analysis.shape_collapse(State.rec, populations=pops,
                                              bin_size_ms=4.0)
    return f"gamma={State.shape.gamma:.3f}"


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


@step("LMC complexity (both)")
def lmc_step():
    pops = list(State.rec.populations.keys())
    State.lmc = nc.analysis.lmc_complexity(
        State.rec, populations=pops, bin_size_s=0.05,
        kind="both", window_seconds=30.0, step_seconds=15.0)
    return f"H={State.lmc.H_per_pop.round(3).tolist()} C={State.lmc.C_per_pop.round(4).tolist()}"


@step("multiscale_entropy")
def mse_step():
    pops = list(State.rec.populations.keys())
    State.mse = nc.analysis.multiscale_entropy(State.rec, populations=pops,
                                                bin_size_s=0.05, scale_max=8)
    return f"sampen.shape={State.mse.sampen.shape}"


@step("manifold PCA")
def man_step():
    State.man = nc.analysis.manifold(State.rec, method="pca", dims=2,
                                      bin_size_s=0.05, sigma_ms=50.0)
    return f"coords={State.man.coords.shape} var={State.man.explained_variance_ratio.round(3).tolist()}"


@step(f"null_test TE (n={SURROGATE_N})")
def null_step():
    State.null = nc.inference.test(State.te, State.rec,
                                    surrogate="spike_dither",
                                    n=SURROGATE_N, seed=0, fdr=True)
    p_fdr = State.null.p_value_fdr
    if p_fdr is not None:
        sig = bool(np.any(np.asarray(p_fdr) < 0.05))
    else:
        sig = False
    return f"any sig (p_fdr<0.05): {sig}"


@step(f"bootstrap branching (n={BOOTSTRAP_N})")
def boot_step():
    State.boot = nc.inference.bootstrap(State.br, State.rec,
                                         n=BOOTSTRAP_N, seed=0)
    return f"CI=({State.boot.ci_lower:.3f},{State.boot.ci_upper:.3f})"


def _save(fig, name):
    p = OUT_DIR / f"{name}.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return p.name


@step("figure_criticality")
def f_crit():
    return _save(nc.viz.figure_criticality(State.crit), "criticality")


@step("figure_branching")
def f_br():
    return _save(nc.viz.figure_branching(State.br), "branching")


@step("figure_dimensionality")
def f_dim():
    return _save(nc.viz.figure_dimensionality(State.dim), "dimensionality")


@step("figure_shape_collapse")
def f_shape():
    return _save(nc.viz.figure_shape_collapse(State.shape), "shape_collapse")


@step("figure_pid")
def f_pid():
    return _save(nc.viz.figure_pid(State.pid), "pid")


@step("figure_lmc_complexity")
def f_lmc():
    return _save(nc.viz.figure_lmc_complexity(State.lmc), "lmc_complexity")


@step("figure_mse")
def f_mse():
    return _save(nc.viz.figure_mse(State.mse), "mse")


@step("figure_manifold")
def f_man():
    return _save(nc.viz.figure_manifold(State.man), "manifold")


@step("figure_significance_matrix")
def f_sig():
    pops = list(State.te.populations)
    return _save(
        nc.viz.figure_significance_matrix(
            State.null, row_labels=pops, col_labels=pops),
        "te_significance_matrix",
    )


@step("figure_volcano")
def f_volc():
    return _save(nc.viz.figure_volcano(State.null), "te_volcano")


@step("figure_te_network")
def f_net():
    return _save(nc.viz.figure_te_network(State.te, State.null, alpha=0.1),
                 "te_network")


@step("figure_bootstrap")
def f_boot():
    return _save(nc.viz.figure_bootstrap(State.boot), "branching_bootstrap")


def main():
    warnings.simplefilter("default")
    print(f"Loading window from {NWB_PATH.name}")
    print(f"Outputs -> {OUT_DIR}")
    if _PROC is not None:
        vm = psutil.virtual_memory()
        print(f"System RAM: total={vm.total/1e9:.1f}GB available={vm.available/1e9:.1f}GB")
    print(f"Start mem: {_mem_mb():.0f}MB", flush=True)

    load_step()
    build_pops()
    estimate_step()
    stat_step()
    crit_step()
    br_step()
    te_step()
    auto_step()
    dim_step()
    shape_step()
    pid_step()
    lmc_step()
    mse_step()
    man_step()
    null_step()
    boot_step()
    f_crit(); f_br(); f_dim(); f_shape(); f_pid()
    f_lmc(); f_mse(); f_man()
    f_sig(); f_volc(); f_net(); f_boot()

    n_pass = sum(1 for r in results if r[1] == "PASS")
    n_fail = sum(1 for r in results if r[1] == "FAIL")
    total = sum(r[2] for r in results)
    peak = max(_mem_mb(), 0)
    print("\n" + "=" * 70)
    print(f"SUMMARY: {n_pass} PASS / {n_fail} FAIL / {len(results)} total "
          f"({total:.1f}s) final_mem={peak:.0f}MB")
    if n_fail:
        for name, status, dt, info in results:
            if status == "FAIL":
                print(f"  FAIL {name}: {info}")
    print("=" * 70)


if __name__ == "__main__":
    main()
