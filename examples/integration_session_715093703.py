"""End-to-end smoke test of neurocomplexity on Allen NWB session 715093703.

Exercises every public analysis + viz + inference path on a real recording.
Crops aggressively (5 min) so it fits on an 8 GB laptop. Each step wrapped
in try/except + timed; final summary prints PASS/FAIL counts.

Run:
    python examples/integration_session_715093703.py
"""
from __future__ import annotations

import time
import warnings
from dataclasses import replace
from pathlib import Path
from traceback import format_exc

try:
    import psutil
    _PROC = psutil.Process()
except ImportError:
    _PROC = None


def _mem_mb() -> float:
    if _PROC is None:
        return float("nan")
    return _PROC.memory_info().rss / 1e6

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import neurocomplexity as nc

NWB_PATH = Path(r"C:\Users\Sazgar\OneDrive\Desktop\Arman_Dinarvand code sample"
                r"\neuropixel\NeuropixelVisCodingData_cache"
                r"\session_715093703\session_715093703.nwb")
OUT_DIR = Path(__file__).parent.parent / "outputs" / "integration_session_715093703"
OUT_DIR.mkdir(parents=True, exist_ok=True)

results: list[tuple[str, str, float, str]] = []  # name, status, secs, info


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
                dmem = mem1 - mem0
                print(f"[OK] {name}  ({dt:.1f}s) mem={mem1:.0f}MB (d{dmem:+.0f})  {info}",
                      flush=True)
                results.append((name, "PASS", dt, info))
            except Exception as exc:
                dt = time.perf_counter() - t0
                tb = format_exc(limit=2)
                print(f"[FAIL] {name}  ({dt:.1f}s) mem={_mem_mb():.0f}MB", flush=True)
                print(tb, flush=True)
                results.append((name, "FAIL", dt, type(exc).__name__))
        return wrap
    return decorator


# Container for state shared across steps.
class State:
    rec = None
    rec_curated = None
    rec_crop = None


@step("load NWB")
def load_nwb():
    State.rec = nc.io.from_nwb(NWB_PATH)
    r = State.rec
    return (f"units={len(r.units)} spikes={r.spike_times.size:,} "
            f"duration={r.duration:.1f}s populations={list(r.populations.keys())}")


@step("filter quality='good'")
def filter_quality():
    r = State.rec
    if "quality" in r.units.columns:
        State.rec_curated = r.filter_units(query="quality == 'good'")
    else:
        State.rec_curated = r
    return f"after curation: units={len(State.rec_curated.units)}"


@step("crop to 30-min window")
def crop_window():
    r = State.rec_curated
    # 30-min window starting at session midpoint - 15min
    target_window = 1800.0
    start = max(0.0, r.duration / 2 - target_window / 2)
    end = min(start + target_window, r.duration)
    State.rec_crop = r.crop(start, end)
    rc = State.rec_crop
    if len(rc.units) < 300:
        # fallback: use uncurated session if curated is too thin
        print(f"[warn] curated units={len(rc.units)} < 300; falling back to uncurated", flush=True)
        State.rec_crop = State.rec.crop(start, end)
        rc = State.rec_crop
    return (f"window=[{start:.0f},{end:.0f}]s ({(end-start):.0f}s) "
            f"units={len(rc.units)} spikes={rc.spike_times.size:,}")


@step("build brain-area populations")
def build_populations():
    rc = State.rec_crop
    if "brain_area" in rc.units.columns:
        # Top 4 most-populated areas
        counts = rc.units["brain_area"].value_counts().head(4)
        State.rec_crop = rc.with_populations(by="brain_area")
        # Keep only populations from top-4 areas; rebuild dict to that subset
        top = list(counts.index)
        new_pops = {k: v for k, v in State.rec_crop.populations.items() if k in top}
        if not new_pops:
            new_pops = State.rec_crop.populations
        State.rec_crop = replace(State.rec_crop, populations=new_pops)
        return f"populations={list(new_pops.keys())}"
    else:
        return "no brain_area column; using existing populations"


@step("estimate_bin_spikes_bytes")
def estimate_mem():
    bytes_ = nc.estimate_bin_spikes_bytes(State.rec_crop,
                                          list(State.rec_crop.populations.keys()),
                                          bin_size_ms=5.0)
    return f"binned counts ~ {bytes_/1e6:.1f} MB"


@step("stationarity diagnostic")
def stationarity_step():
    s = nc.analysis.stationarity(State.rec_crop, window_s=30.0)
    return (f"is_stationary={s.is_stationary} flags={len(s.flags)} "
            f"n_windows={s.n_windows}")


@step("criticality (bin sweep)")
def crit_step():
    State.crit = nc.analysis.criticality(State.rec_crop, populations=["all"]
                                         if "all" in State.rec_crop.populations
                                         else list(State.rec_crop.populations.keys())[:1],
                                         bin_size_ms=(2, 4, 8, 16))
    return f"alpha_s={State.crit.alpha_s:.3f} alpha_t={State.crit.alpha_t:.3f}"


@step("branching ratio (Wilting MR)")
def branching_step():
    State.branching = nc.analysis.wilting_mr(State.rec_crop,
                                             populations=list(State.rec_crop.populations.keys())[:1],
                                             bin_size_ms=4.0, k_max=20)
    return f"m={State.branching.m:.4f}"


@step("transfer_entropy matrix")
def te_step():
    pops = list(State.rec_crop.populations.keys())[:4]
    State.te = nc.analysis.transfer_entropy(State.rec_crop, populations=pops,
                                            bin_size_ms=5.0, delay_bins=1)
    return f"matrix.shape={State.te.matrix.shape} max={State.te.matrix.max():.4f}"


@step("autonomy per population")
def autonomy_step():
    pops = list(State.rec_crop.populations.keys())[:4]
    if len(pops) < 2:
        return "skip: need >=2 populations"
    State.auto = nc.analysis.autonomy(State.rec_crop, populations=pops,
                                       bin_size_ms=5.0, max_lag=3)
    return f"keys={list(State.auto.values.keys())}"


@step("dimensionality (PR)")
def dim_step():
    State.dim = nc.analysis.dimensionality(State.rec_crop,
                                            populations=list(State.rec_crop.populations.keys())[:4],
                                            bin_size_ms=20.0)
    return f"participation_ratio={State.dim.participation_ratio:.3f}"


@step("shape collapse")
def shape_step():
    State.shape = nc.analysis.shape_collapse(State.rec_crop,
                                              populations=list(State.rec_crop.populations.keys())[:1],
                                              bin_size_ms=4.0)
    return f"gamma={State.shape.gamma:.3f}"


@step("partial information (PID)")
def pid_step():
    pops = list(State.rec_crop.populations.keys())
    if len(pops) < 3:
        return "skip: need >=3 populations"
    State.pid = nc.analysis.partial_information(
        State.rec_crop, target_pop=pops[0], sources=(pops[1], pops[2]),
        bin_size_ms=10.0, n_levels=2)
    return (f"R={State.pid.redundancy:.4f} U1={State.pid.unique_1:.4f} "
            f"U2={State.pid.unique_2:.4f} S={State.pid.synergy:.4f}")


@step("LMC complexity (population mode)")
def lmc_step():
    State.lmc = nc.analysis.lmc_complexity(State.rec_crop,
                                            populations=list(State.rec_crop.populations.keys())[:4],
                                            bin_size_s=0.05, kind="population")
    return f"H={State.lmc.H_per_pop} C={State.lmc.C_per_pop}"


@step("LMC complexity (both mode)")
def lmc_both_step():
    State.lmc_both = nc.analysis.lmc_complexity(
        State.rec_crop,
        populations=list(State.rec_crop.populations.keys())[:4],
        bin_size_s=0.05, kind="both", window_seconds=5.0, step_seconds=2.5)
    return f"C_traj.shape={State.lmc_both.C_traj.shape}"


@step("multiscale entropy")
def mse_step():
    State.mse = nc.analysis.multiscale_entropy(State.rec_crop,
                                                populations=list(State.rec_crop.populations.keys())[:4],
                                                bin_size_s=0.05, scale_max=8)
    return f"sampen.shape={State.mse.sampen.shape}"


@step("manifold PCA")
def manifold_pca_step():
    State.man = nc.analysis.manifold(State.rec_crop, method="pca", dims=2,
                                      bin_size_s=0.05, sigma_ms=50.0)
    return (f"coords.shape={State.man.coords.shape} "
            f"var={State.man.explained_variance_ratio}")


@step("null_test (TE)")
def null_te_step():
    State.te_null = nc.inference.test(State.te, State.rec_crop,
                                       surrogate="spike_dither",
                                       n=50, seed=0, fdr=True)
    p_fdr = State.te_null.p_value_fdr
    sig = bool(np.any(np.asarray(p_fdr) < 0.05)) if p_fdr is not None else False
    return f"any sig (p_fdr<0.05): {sig}"


@step("bootstrap (branching m)")
def boot_step():
    State.boot = nc.inference.bootstrap(State.branching, State.rec_crop,
                                         n=30, seed=0)
    return f"CI=({State.boot.ci_lower:.3f},{State.boot.ci_upper:.3f})"


# ---- figures ----------------------------------------------------------------

def _save(fig, name):
    p = OUT_DIR / f"{name}.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(p.name)


@step("figure_criticality")
def fig_crit():
    fig = nc.viz.figure_criticality(State.crit)
    return _save(fig, "criticality")


@step("figure_branching")
def fig_branching():
    fig = nc.viz.figure_branching(State.branching)
    return _save(fig, "branching")


@step("figure_dimensionality")
def fig_dim():
    fig = nc.viz.figure_dimensionality(State.dim)
    return _save(fig, "dimensionality")


@step("figure_shape_collapse")
def fig_shape():
    fig = nc.viz.figure_shape_collapse(State.shape)
    return _save(fig, "shape_collapse")


@step("figure_pid")
def fig_pid():
    fig = nc.viz.figure_pid(State.pid)
    return _save(fig, "pid")


@step("figure_lmc_complexity")
def fig_lmc():
    fig = nc.viz.figure_lmc_complexity(State.lmc_both)
    return _save(fig, "lmc_complexity")


@step("figure_mse")
def fig_mse():
    fig = nc.viz.figure_mse(State.mse)
    return _save(fig, "mse")


@step("figure_manifold")
def fig_manifold():
    fig = nc.viz.figure_manifold(State.man)
    return _save(fig, "manifold")


@step("figure_significance_matrix (TE)")
def fig_sigmat():
    fig = nc.viz.figure_significance_matrix(State.te_null)
    return _save(fig, "te_significance_matrix")


@step("figure_volcano (TE)")
def fig_volc():
    fig = nc.viz.figure_volcano(State.te_null)
    return _save(fig, "te_volcano")


@step("figure_te_network")
def fig_net():
    fig = nc.viz.figure_te_network(State.te, State.te_null, alpha=0.1)
    return _save(fig, "te_network")


@step("figure_bootstrap")
def fig_boot():
    fig = nc.viz.figure_bootstrap(State.boot)
    return _save(fig, "branching_bootstrap")


def main():
    warnings.simplefilter("default")
    print(f"Loading {NWB_PATH.name}")
    print(f"Outputs -> {OUT_DIR}")
    if _PROC is not None:
        vm = psutil.virtual_memory()
        print(f"System RAM: total={vm.total/1e9:.1f}GB available={vm.available/1e9:.1f}GB")
    print(f"Start mem: {_mem_mb():.0f}MB\n", flush=True)

    # Sequence
    load_nwb()
    filter_quality()
    crop_window()
    build_populations()
    estimate_mem()
    stationarity_step()
    crit_step()
    branching_step()
    te_step()
    autonomy_step()
    dim_step()
    shape_step()
    pid_step()
    lmc_step()
    lmc_both_step()
    mse_step()
    manifold_pca_step()
    null_te_step()
    boot_step()
    fig_crit()
    fig_branching()
    fig_dim()
    fig_shape()
    fig_pid()
    fig_lmc()
    fig_mse()
    fig_manifold()
    fig_sigmat()
    fig_volc()
    fig_net()
    fig_boot()

    # Summary
    n_pass = sum(1 for r in results if r[1] == "PASS")
    n_fail = sum(1 for r in results if r[1] == "FAIL")
    total_time = sum(r[2] for r in results)
    print("\n" + "=" * 70)
    print(f"SUMMARY: {n_pass} PASS / {n_fail} FAIL / {len(results)} total "
          f"({total_time:.1f}s)")
    if n_fail:
        print("\nFailures:")
        for name, status, dt, info in results:
            if status == "FAIL":
                print(f"  - {name}: {info}")
    print("=" * 70)


if __name__ == "__main__":
    main()
