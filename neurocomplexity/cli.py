"""Command-line interface for neurocomplexity.

Subcommands
-----------
* ``info <nwb>``               — quick recording summary (units, populations).
* ``analyze <nwb> -o <dir>``   — run the full analysis pipeline,
                                  emit JSON results + figures.
* ``figure <results.json> -o`` — re-render figures from a cached results JSON.

Example
-------
    neurocomplexity analyze session_715093703.nwb \\
        --start 3765.6 --end 4066.9 \\
        --target VISp --sources LGd CA1 \\
        -o results/session_715093703/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from neurocomplexity import __version__

# ---------------------------------------------------------------------------
# JSON encoding for results (numpy arrays + dataclasses)
# ---------------------------------------------------------------------------

class _ResultEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if hasattr(obj, "__dict__") and not isinstance(obj, type):
            try:
                return {k: v for k, v in obj.__dict__.items()
                        if not k.startswith("_")}
            except Exception:
                pass
        return super().default(obj)


def _logger(args):
    """Return a ``log(msg)`` that writes human progress to the right stream.

    In ``--json`` mode all progress goes to **stderr** so that **stdout**
    carries only the machine-readable JSON payload (clean for piping into
    ``jq`` etc.). Otherwise progress goes to stdout as before.
    """
    stream = sys.stderr if getattr(args, "json", False) else sys.stdout

    def log(msg: str = "") -> None:
        print(msg, file=stream, flush=True)

    return log


def _result_to_dict(name: str, result) -> dict:
    if result is None:
        return {"name": name, "available": False}
    if is_dataclass(result):
        d = asdict(result)
    else:
        d = {k: v for k, v in result.__dict__.items() if not k.startswith("_")}
    # Drop the loader provenance from each result (kept once at top level)
    d.pop("source", None)
    return {"name": name, "available": True, **d}


# ---------------------------------------------------------------------------
# Loading & preparation
# ---------------------------------------------------------------------------

def _load_recording(path: Path, *, qualities, start, end, drop_unassigned,
                    log=print):
    import neurocomplexity as nc
    log(f"[{time.strftime('%H:%M:%S')}] loading NWB...")
    t0 = time.time()
    rec = nc.io.from_nwb(path)
    if qualities:
        rec = rec.filter_units(quality=list(qualities))
    rec = rec.with_populations(
        by="brain_area",
        on_unassigned="drop" if drop_unassigned else "other",
    )
    if start is not None or end is not None:
        rec = rec.crop(start if start is not None else 0.0,
                       end if end is not None else rec.duration)
    log(f"  loaded {rec.units.shape[0]} units, {rec.spike_times.size:,} spikes, "
        f"{rec.duration:.1f}s, {len(rec.populations)} populations  "
        f"(elapsed {time.time()-t0:.1f}s)")
    return rec


# ---------------------------------------------------------------------------
# Subcommand: info
# ---------------------------------------------------------------------------

def cmd_info(args):
    """``neurocomplexity info <nwb>`` — print a short recording summary.

    Loads the file via :func:`~neurocomplexity.io.from_nwb`, prints
    duration, unit-quality counts and population sizes. No analyses run.
    """
    import neurocomplexity as nc
    rec = nc.io.from_nwb(Path(args.nwb)).with_populations(
        by="brain_area", on_unassigned="other")
    pops = {name: int(mask.sum()) for name, mask in rec.populations.items()}
    qualities = sorted(rec.units["quality"].dropna().unique().tolist())
    if getattr(args, "json", False):
        payload = {
            "neurocomplexity_version": __version__,
            "input": str(args.nwb),
            "n_units": int(rec.units.shape[0]),
            "n_spikes": int(rec.spike_times.size),
            "duration_seconds": float(rec.duration),
            "unit_qualities": qualities,
            "populations": pops,
        }
        json.dump(payload, sys.stdout, cls=_ResultEncoder, indent=2)
        sys.stdout.write("\n")
        return 0
    print(rec)
    print(f"  duration = {rec.duration:.1f} s")
    print(f"  unit qualities = {qualities}")
    print("  populations:")
    for name, n in sorted(pops.items(), key=lambda kv: -kv[1]):
        print(f"    {name:>10s}  n={n}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: analyze
# ---------------------------------------------------------------------------

def _run_analyses(rec, args):
    import neurocomplexity as nc
    log = _logger(args)
    pops = args.populations or None
    target = args.target
    sources = args.sources or []

    out = {}

    log("[criticality] alpha_s, alpha_t, gamma_predicted ...")
    t = time.time()
    try:
        _crit_bin = (args.crit_bin_sweep
                      if args.crit_bin_sweep is not None
                      else args.crit_bin_ms)
        out["criticality"] = nc.analysis.criticality(
            rec, populations=pops, bin_size=_crit_bin)
        log(f"  alpha_s={out['criticality'].alpha_s:.3f}  "
            f"alpha_t={out['criticality'].alpha_t:.3f}  "
            f"R2={out['criticality'].r_squared:.3f}  "
            f"({time.time()-t:.1f}s)")
    except Exception as e:
        log(f"  failed: {e}"); out["criticality"] = None

    log("[branching] Wilting MR ...")
    t = time.time()
    try:
        out["branching"] = nc.analysis.wilting_mr(
            rec, populations=pops, bin_size_ms=args.bin_ms, k_max=args.k_max)
        log(f"  m={out['branching'].m:.3f}  R2={out['branching'].r_squared:.3f}  "
            f"({time.time()-t:.1f}s)")
    except Exception as e:
        log(f"  failed: {e}"); out["branching"] = None

    log("[shape collapse] ...")
    t = time.time()
    try:
        out["shape_collapse"] = nc.analysis.shape_collapse(
            rec, populations=pops, bin_size_ms=args.bin_ms)
        log(f"  gamma={out['shape_collapse'].gamma:.3f}  "
            f"resid={out['shape_collapse'].residual:.4g}  "
            f"({time.time()-t:.1f}s)")
    except Exception as e:
        log(f"  failed: {e}"); out["shape_collapse"] = None

    log("[dimensionality] participation ratio ...")
    t = time.time()
    try:
        out["dimensionality"] = nc.analysis.dimensionality(
            rec, populations=pops, bin_size_ms=args.dim_bins)
        log(f"  PR={out['dimensionality'].participation_ratio:.2f}  "
            f"N={out['dimensionality'].n_units}  ({time.time()-t:.1f}s)")
    except Exception as e:
        log(f"  failed: {e}"); out["dimensionality"] = None

    if target and len(sources) == 2:
        log(f"[PID] target={target}, sources={sources} ...")
        t = time.time()
        try:
            out["pid"] = nc.analysis.partial_information(
                rec, target_pop=target, sources=sources,
                bin_size_ms=args.pid_bins, n_levels=args.pid_levels)
            log(f"  R={out['pid'].redundancy:.4f}  "
                f"U1={out['pid'].unique_1:.4f}  "
                f"U2={out['pid'].unique_2:.4f}  "
                f"S={out['pid'].synergy:.4f}  "
                f"total_MI={out['pid'].total_mi:.4f}  ({time.time()-t:.1f}s)")
        except Exception as e:
            log(f"  failed: {e}"); out["pid"] = None
    else:
        out["pid"] = None
        if target or sources:
            log("[PID] skipped — need --target and exactly 2 --sources")

    return out


def cmd_analyze(args):
    """``neurocomplexity analyze <nwb> -o <dir>`` — full pipeline.

    Loads the recording, optionally crops it (``--start`` / ``--end``),
    filters by unit quality (``--quality``), restricts to a target
    population (``--target``) and chosen source populations
    (``--sources``), runs every analysis whose dependencies are met, and
    writes ``results.json`` plus per-analysis figures (SVG / TIFF / JPG)
    into ``--output``.

    Run ``neurocomplexity analyze --help`` for the full flag list.
    """
    log = _logger(args)
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    rec = _load_recording(
        Path(args.nwb),
        qualities=args.quality,
        start=args.start, end=args.end,
        drop_unassigned=args.drop_unassigned,
        log=log,
    )

    results = _run_analyses(rec, args)

    # Dump JSON
    payload: dict[str, Any] = {
        "neurocomplexity_version": __version__,
        "input": str(args.nwb),
        "crop": [args.start, args.end],
        "qualities": list(args.quality) if args.quality else None,
        "populations": args.populations,
        "n_units": int(rec.units.shape[0]),
        "n_spikes": int(rec.spike_times.size),
        "duration_seconds": float(rec.duration),
        "results": {k: _result_to_dict(k, v) for k, v in results.items()},
    }
    json_path = outdir / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, cls=_ResultEncoder, indent=2)
    log(f"\n  wrote {json_path}")
    if getattr(args, "json", False):
        # Machine-readable payload on stdout; progress already went to stderr.
        json.dump(payload, sys.stdout, cls=_ResultEncoder, indent=2)
        sys.stdout.write("\n")

    if not args.no_figures:
        import matplotlib
        matplotlib.use("Agg")
        from neurocomplexity import viz
        # Per-analysis figures (each rendered as its own SVG+TIFF+JPG triplet)
        if results.get("criticality") is not None:
            f1 = viz.figure_criticality(results["criticality"])
            viz.save_publication(f1, outdir / "criticality",
                                  formats=tuple(args.formats))
        if results.get("branching") is not None:
            f2 = viz.figure_branching(results["branching"])
            viz.save_publication(f2, outdir / "branching",
                                  formats=tuple(args.formats))
        if results.get("shape_collapse") is not None:
            f3 = viz.figure_shape_collapse(results["shape_collapse"])
            viz.save_publication(f3, outdir / "shape_collapse",
                                  formats=tuple(args.formats))
        if results.get("dimensionality") is not None:
            f4 = viz.figure_dimensionality(results["dimensionality"])
            viz.save_publication(f4, outdir / "dimensionality",
                                  formats=tuple(args.formats))
        if results.get("pid") is not None:
            f5 = viz.figure_pid(results["pid"])
            viz.save_publication(f5, outdir / "pid",
                                  formats=tuple(args.formats))

    return 0


# ---------------------------------------------------------------------------
# Subcommand: figure  (re-render from results.json)
# ---------------------------------------------------------------------------

class _Bunch:
    """Lightweight namespace so cached JSON behaves like a result object."""
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, list) and v and isinstance(v[0], (int, float)):
                v = np.asarray(v)
            setattr(self, k, v)


def _rebuild(name, d):
    if not d.get("available", True):
        return None
    payload = {k: v for k, v in d.items() if k not in ("name", "available")}
    if "rescaled_y" in payload:
        payload["rescaled_y"] = np.asarray(payload["rescaled_y"])
    if "rescaled_x" in payload:
        payload["rescaled_x"] = np.asarray(payload["rescaled_x"])
    if "mean_shapes" in payload:
        payload["mean_shapes"] = [np.asarray(a) for a in payload["mean_shapes"]]
    if "durations_used" in payload:
        payload["durations_used"] = np.asarray(payload["durations_used"])
    if "eigenvalues" in payload:
        payload["eigenvalues"] = np.asarray(payload["eigenvalues"])
    if "r_values" in payload:
        payload["r_values"] = np.asarray(payload["r_values"])
    if "k_lags" in payload:
        payload["k_lags"] = np.asarray(payload["k_lags"])
    if "sizes" in payload:
        payload["sizes"] = np.asarray(payload["sizes"])
    if "lifetimes" in payload:
        payload["lifetimes"] = np.asarray(payload["lifetimes"])
    if "sources" in payload and isinstance(payload["sources"], list):
        payload["sources"] = tuple(payload["sources"])
    return _Bunch(payload)


def cmd_figure(args):
    """``neurocomplexity figure <results.json> -o <dir>`` — re-render figures.

    Rebuilds result dataclasses from the JSON written by
    :func:`cmd_analyze` and re-renders every panel without re-loading the
    NWB or recomputing any statistic. Useful when iterating on styling.
    """
    import matplotlib
    matplotlib.use("Agg")
    from neurocomplexity import viz

    data = json.loads(Path(args.results).read_text(encoding="utf-8"))
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    rebuilt = {k: _rebuild(k, v) for k, v in data["results"].items()}
    pairs = [
        ("criticality",   viz.figure_criticality),
        ("branching",     viz.figure_branching),
        ("shape_collapse", viz.figure_shape_collapse),
        ("dimensionality", viz.figure_dimensionality),
        ("pid",           viz.figure_pid),
    ]
    for name, fn in pairs:
        res = rebuilt.get(name)
        if res is None:
            continue
        fig = fn(res)
        paths = viz.save_publication(fig, outdir / name,
                                     formats=tuple(args.formats))
        for p in paths:
            print(f"  wrote {p}")
    return 0


# ---------------------------------------------------------------------------
# argparse plumbing
# ---------------------------------------------------------------------------

def _add_common_analysis_args(p):
    p.add_argument("--start", type=float, default=None,
                    help="crop start (s)")
    p.add_argument("--end", type=float, default=None,
                    help="crop end (s)")
    p.add_argument("--quality", nargs="*", default=["good"],
                    help="unit quality filter (default: good)")
    p.add_argument("--drop-unassigned", action="store_true",
                    help="drop units missing brain_area instead of bucketing into 'other'")
    p.add_argument("--populations", nargs="*", default=None,
                    help="populations to analyse (default: all)")
    p.add_argument("--target", default=None, help="PID target population")
    p.add_argument("--sources", nargs="*", default=None,
                    help="PID source populations (exactly 2)")
    p.add_argument("--bin-ms", type=float, default=4.0,
                    help="bin size for branching/shape collapse (ms)")
    p.add_argument("--crit-bin-ms", type=float, default=4.0,
                    help="bin size for criticality (ms). Default 4 ms is "
                    "the principled single-bin choice. To run the legacy "
                    "R²-driven sweep, pass --crit-bin-sweep instead.")
    p.add_argument("--crit-bin-sweep", nargs="*", type=float, default=None,
                    help="candidate bin sizes for criticality (ms). "
                    "If supplied, runs the legacy R²-driven selection and "
                    "emits a forking-path warning; the full per-bin table "
                    "lands in CriticalityResult.fits.")
    p.add_argument("--k-max", type=int, default=50,
                    help="max lag for Wilting MR")
    p.add_argument("--dim-bins", type=float, default=10.0,
                    help="bin size for dimensionality (ms)")
    p.add_argument("--pid-bins", type=float, default=5.0,
                    help="bin size for PID (ms)")
    p.add_argument("--pid-levels", type=int, default=3,
                    help="quantile discretisation levels for PID")
    p.add_argument("--formats", nargs="*",
                    default=["svg", "tiff", "jpg"],
                    choices=["svg", "tiff", "jpg"],
                    help="figure formats to emit (default svg tiff jpg)")
    p.add_argument("--no-figures", action="store_true",
                    help="skip figure rendering")
    p.add_argument("--json", action="store_true",
                    help="emit the results JSON to stdout (machine-readable); "
                    "human progress is routed to stderr")


def build_parser():
    """Construct the ``argparse`` parser for the CLI.

    Exposed so external tools (tests, doc builders, completion plugins) can
    introspect the available flags without invoking :func:`main`.

    Returns
    -------
    argparse.ArgumentParser
    """
    p = argparse.ArgumentParser(
        prog="neurocomplexity",
        description="Criticality, branching, shape collapse, dimensionality, "
                    "and PID for spike-sorted recordings.",
    )
    p.add_argument("--version", action="version",
                    version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_info = sub.add_parser("info", help="describe an NWB recording")
    p_info.add_argument("nwb")
    p_info.add_argument("--json", action="store_true",
                         help="emit the summary as JSON on stdout")
    p_info.set_defaults(func=cmd_info)

    p_an = sub.add_parser("analyze", help="run analyses + figures")
    p_an.add_argument("nwb")
    p_an.add_argument("-o", "--output", required=True,
                       help="output directory")
    _add_common_analysis_args(p_an)
    p_an.set_defaults(func=cmd_analyze)

    p_fig = sub.add_parser("figure", help="render figures from cached results")
    p_fig.add_argument("results", help="path to results.json")
    p_fig.add_argument("-o", "--output", required=True)
    p_fig.add_argument("--formats", nargs="*",
                        default=["svg", "tiff", "jpg"],
                        choices=["svg", "tiff", "jpg"])
    p_fig.set_defaults(func=cmd_figure)

    p_bench = sub.add_parser(
        "benchmark",
        help="run benchmark cases against synthetic ground-truth data",
    )
    p_bench.add_argument(
        "--case", action="append", default=None,
        help="case name (repeatable). Default: run all registered cases.",
    )
    p_bench.add_argument(
        "--reps", type=int, default=200,
        help="reps per case (default 200; CI uses 2-20).",
    )
    p_bench.add_argument("--seed", type=int, default=0)
    p_bench.add_argument(
        "-o", "--out", type=str, default=None,
        help="path to write CSV summary. If omitted, prints to stdout.",
    )
    p_bench.set_defaults(func=cmd_benchmark)

    return p


def cmd_benchmark(args) -> int:
    """``neurocomplexity benchmark`` — run benchmark cases against ground truth.

    Selects cases via ``--case`` (repeatable, default = all), runs them at
    ``--reps`` reps with the given ``--seed``, and writes a CSV summary
    to ``--out`` or stdout.
    """
    from neurocomplexity.benchmarks import run_all
    df = run_all(
        cases=args.case, n_reps=args.reps, seed=args.seed, verbose=True,
    )
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"[benchmarks] wrote {args.out}")
    else:
        print(df.to_string(index=False))
    return 0 if df["passed"].all() else 1


def main(argv=None):
    """Entry point for ``python -m neurocomplexity`` and the
    ``neurocomplexity`` console script.

    Parses ``argv`` (or ``sys.argv[1:]``) and dispatches to the
    appropriate subcommand. Returns the subcommand's exit code (0 on
    success).
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
