"""Bin-size sensitivity of the Steinmetz criticality result (M3 robustness).

For every crackling-consistent population (sigma-free 37-pop funnel, matching
the manuscript), re-fit tau (alpha_s), alpha (alpha_t), gamma_fit, and the
Sethna deviation at bins {2, 4, 8} ms and the per-population adaptive (mean-IEI)
bin. Writes a tidy long-format CSV. NB the raw exponents are *not* bin-invariant
(they track the bin-set avalanche timescale); the manuscript reports the
crackling-noise *consistency* as the bin-robust quantity, not the raw exponents.

DEFERRED: multi-hour run; execute for the camera-ready revision.
Run:  py -3 datasets/steinmetz_bin_sensitivity.py
Out:  datasets/steinmetz_bin_sensitivity.csv
"""
import csv
import warnings
import numpy as np
import neurocomplexity as nc
from steinmetz_analysis import build_area_units, WINDOW_S

warnings.simplefilter("ignore")
SIG_LO, SIG_HI, DG_THRESH, R2_THRESH, MIN_N = 0.85, 1.15, 0.10, 0.85, 30
TRACT = {"em", "fr", "or", "fp", "ccg", "dhc", "fi", "ccb", "int", "cing",
         "alv", "VL", "opt", "st", "scwm", "ml", "arb", "sm", "aco"}
BINS = [2.0, 4.0, 8.0, "adaptive"]


def F(r, k):
    try:
        return float(r[k])
    except Exception:
        return float("nan")


def main():
    with open("datasets/steinmetz_results.csv") as fh:
        rows = list(csv.DictReader(fh))
    by_mouse = {}
    for r in rows:
        by_mouse.setdefault(r["mouse"], []).append(r)
    out = []
    for mouse, mrows in by_mouse.items():
        au, _ = build_area_units(mouse)
        for r in mrows:
            # sigma-FREE funnel: naive branching sigma is the negative control
            # in the manuscript, NOT a selection criterion (gives 37 pops).
            if not (int(r["N"]) >= MIN_N and r["area"] not in TRACT
                    and F(r, "sethna_delta") <= DG_THRESH
                    and F(r, "r2") >= R2_THRESH):
                continue
            w0, w1 = float(r["win_start"]), float(r["win_end"])
            units = {}
            for i, sp in enumerate(au[r["area"]].values()):
                s = np.asarray(sp, float)
                s = s[(s >= w0) & (s < w1)] - w0
                if s.size >= 2:
                    units[i] = s
            rec = nc.io.from_dict(units, WINDOW_S)
            for b in BINS:
                bs = b if isinstance(b, str) else (b,)
                c = nc.analysis.criticality(rec, bin_size=bs)
                out.append(dict(
                    mouse=mouse, area=r["area"], bin=str(b),
                    bin_ms=f"{c.optimal_bin:.3f}",
                    tau=f"{c.alpha_s:.3f}", alpha=f"{c.alpha_t:.3f}",
                    gamma_fit=f"{c.gamma_fit:.3f}",
                    gamma_pred=f"{c.gamma_predicted:.3f}",
                    sethna_delta=f"{abs(c.gamma_fit - c.gamma_predicted) / c.gamma_predicted:.3f}",
                    r2=f"{c.r_squared:.3f}"))
                print(f"{mouse} {r['area']:8s} bin={str(b):8s} "
                      f"tau={c.alpha_s:.2f} alpha={c.alpha_t:.2f} "
                      f"dG={out[-1]['sethna_delta']}", flush=True)
    with open("datasets/steinmetz_bin_sensitivity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"\nwrote datasets/steinmetz_bin_sensitivity.csv ({len(out)} rows)")


if __name__ == "__main__":
    main()
