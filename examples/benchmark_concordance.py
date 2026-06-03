"""Run cross-tool concordance benchmark and print a results summary.

Usage:
    python examples/benchmark_concordance.py

This runs the registered concordance cases (see
``neurocomplexity/benchmarks/cases/concordance.py``) and prints a summary
table. Cases skip cleanly when the reference tool is not installed.
"""
from __future__ import annotations

import json
import os
import sys

# Allow direct execution from repo root: `python examples/benchmark_concordance.py`
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from neurocomplexity.benchmarks.cases.concordance import CONCORDANCE_CASES


def _fmt(x):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.4g}"
    return str(x)


def main():
    print("\nCross-tool concordance benchmark")
    print("=" * 78)
    all_pass = True
    summary = []
    for case_fn in CONCORDANCE_CASES:
        r = case_fn()
        summary.append(r)
        name = r.get("name", "?")
        tol = r.get("tolerance", float("nan"))
        if r.get("skipped"):
            print(f"  SKIP  {name:<30s} {r.get('reason', '')}")
            continue
        status = "PASS" if r.get("pass") else "FAIL"
        if not r.get("pass"):
            all_pass = False
        diff_str = ""
        if "diff" in r:
            diff_str = f"|diff|={_fmt(r['diff'])}"
        elif "diff_red_analytic" in r:
            diff_str = (
                f"|diff_red(an)|={_fmt(r['diff_red_analytic'])} "
                f"|diff_syn(an)|={_fmt(r['diff_syn_analytic'])}"
            )
            if r.get("dit_status") == "ok":
                diff_str += (
                    f" |diff_red(dit)|={_fmt(r['diff_red_dit'])} "
                    f"|diff_syn(dit)|={_fmt(r['diff_syn_dit'])}"
                )
        print(f"  {status}  {name:<30s} tol={_fmt(tol)}  {diff_str}")

    print("=" * 78)
    print("Overall:", "ALL PASS" if all_pass else "FAILURES FOUND")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "benchmark_concordance.json")
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
