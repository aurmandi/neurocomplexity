"""Per-population BCa bootstrap CIs (tau=alpha_s, alpha=alpha_t, gamma_fit)
for the criticality-passing Steinmetz populations. Appends CI columns to
datasets/steinmetz_results.csv. Allen-grade: BCa, n=2000 avalanche resamples.

Run:  py -3 datasets/steinmetz_table3_ci.py
"""
import csv
import warnings
import numpy as np
import neurocomplexity as nc
from neurocomplexity.inference.bootstrap import bootstrap_avalanche_exponents
from steinmetz_analysis import build_area_units, WINDOW_S

warnings.simplefilter("ignore")
SEED, NBOOT = 0, 2000
SIG_LO, SIG_HI, DG_THRESH, R2_THRESH, MIN_N = 0.85, 1.15, 0.10, 0.85, 30
TRACT = {"em", "fr", "or", "fp", "ccg", "dhc", "fi", "ccb", "int", "cing",
         "alv", "VL", "opt", "st", "scwm", "ml", "arb", "sm", "aco"}


def F(r, k):
    try:
        return float(r[k])
    except Exception:
        return float("nan")


def main():
    rows = list(csv.DictReader(open("datasets/steinmetz_results.csv")))
    by_mouse = {}
    for r in rows:
        by_mouse.setdefault(r["mouse"], []).append(r)
    ci = {}
    for mouse, mrows in by_mouse.items():
        au, _ = build_area_units(mouse)
        for r in mrows:
            sig = F(r, "branching")
            passes = (int(r["N"]) >= MIN_N and r["area"] not in TRACT
                      and SIG_LO <= sig <= SIG_HI
                      and F(r, "sethna_delta") <= DG_THRESH
                      and F(r, "r2") >= R2_THRESH)
            if not passes:
                continue
            w0, w1 = float(r["win_start"]), float(r["win_end"])
            units = {}
            for i, sp in enumerate(au[r["area"]].values()):
                s = np.asarray(sp, float)
                s = s[(s >= w0) & (s < w1)] - w0
                if s.size >= 2:
                    units[i] = s
            rec = nc.io.from_dict(units, WINDOW_S)
            res = nc.analysis.criticality(rec, bin_size=(4.0,))
            inf = bootstrap_avalanche_exponents(res, rec, n=NBOOT, seed=SEED)
            lo, hi = np.asarray(inf.ci_lower), np.asarray(inf.ci_upper)
            ci[(mouse, r["area"])] = dict(
                tau_lo=f"{lo[0]:.3f}", tau_hi=f"{hi[0]:.3f}",
                alpha_lo=f"{lo[1]:.3f}", alpha_hi=f"{hi[1]:.3f}",
                gfit_lo=f"{lo[2]:.3f}", gfit_hi=f"{hi[2]:.3f}")
            print(f"{mouse} {r['area']}: tau[{lo[0]:.2f},{hi[0]:.2f}] "
                  f"alpha[{lo[1]:.2f},{hi[1]:.2f}] gfit[{lo[2]:.2f},{hi[2]:.2f}]",
                  flush=True)
    new = ["tau_lo", "tau_hi", "alpha_lo", "alpha_hi", "gfit_lo", "gfit_hi"]
    fields = list(rows[0].keys()) + [c for c in new if c not in rows[0]]
    with open("datasets/steinmetz_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            r.update(ci.get((r["mouse"], r["area"]),
                            {c: "" for c in new}))
            w.writerow(r)
    print(f"\nattached BCa CIs for {len(ci)} populations")


if __name__ == "__main__":
    main()
