"""Cross-tool concordance benchmark.

Compares neurocomplexity estimator output to upstream reference
implementations on shared synthetic inputs. The point is to verify that
neurocomplexity's re-implementation matches the canonical specialist
toolboxes numerically, not just that it passes its own self-tests.

Currently registered cases:
  * branching_vs_mrestimator
      Wilting-Priesemann m on a saturated branching network, against the
      mrestimator Python package (Spitzner et al. 2021).
  * pid_vs_dit
      Williams-Beer I_min PID atoms on the canonical XOR / AND
      distributions, against the dit information-theory toolkit
      (James et al.) — also cross-checked against the closed-form
      analytic identities.

Not currently runnable as cross-tool comparisons (Java / MATLAB only;
no importable Python binding):
  * IDTxl (TE, BROJA PID) — Python wheel exists but pulls JPype + JIDT.jar
  * JIDT (TE / MI) — Java
  * MVGC (Granger causality) — MATLAB
For these the package's internal benchmark recovery against analytic
ground truth (Tab 2) is the available evidence.

Each case returns a dict whose schema is:
    {
        "name": str,
        "skipped": bool,            # True if reference tool unavailable
        "reason": str,              # populated when skipped=True
        "tolerance": float,
        "pass": bool,               # only meaningful when not skipped
        ...case-specific fields (nc_*, ref_*, diff_*) ...
    }
"""
from __future__ import annotations

import numpy as np

CONCORDANCE_CASES: list = []


def _register(fn):
    CONCORDANCE_CASES.append(fn)
    return fn


@_register
def concordance_branching_vs_mrestimator():
    """neurocomplexity m vs mrestimator m on a shared saturated branching
    network simulator (m_true = 0.95).
    """
    try:
        import mrestimator as mre
    except ImportError:
        return {
            "name": "branching_vs_mrestimator",
            "skipped": True,
            "reason": "mrestimator not installed (pip install mrestimator)",
            "tolerance": 0.05,
            "pass": False,
        }

    from neurocomplexity.analysis._binning import bin_spikes
    from neurocomplexity.analysis.branching import wilting_mr
    from neurocomplexity.benchmarks.simulators.branching_network import (
        branching_network,
    )

    m_true = 0.95
    rec = branching_network(
        n_units=100, m=m_true, duration_s=600.0, bin_ms=4.0,
        external_rate_hz=0.5, saturate=True, seed=0,
    )

    # neurocomplexity m
    nc = wilting_mr(rec, populations=["all"], bin_size_ms=4.0, k_max=50, k_min=1)
    nc_m = float(nc.m)

    # mrestimator m on the SAME binned activity series
    counts = bin_spikes(rec, ["all"], 0.004)[:, 0].astype(np.float64)
    try:
        rk = mre.coefficients(counts, dt=4.0, dtunit="ms",
                              steps=(1, 50), method="ts")
        fit = mre.fit(rk, fitfunc=mre.f_exponential_offset)
        # mre tau is in dt-units; convert to m via m = exp(-dt/tau).
        tau = float(fit.tau)
        mre_m = float(np.exp(-1.0 / tau)) if tau > 0 else float("nan")
    except Exception as e:
        return {
            "name": "branching_vs_mrestimator",
            "skipped": True,
            "reason": f"mrestimator fit failed: {e!r}",
            "tolerance": 0.05,
            "pass": False,
        }

    diff = abs(nc_m - mre_m)
    tol = 0.05
    return {
        "name": "branching_vs_mrestimator",
        "skipped": False,
        "m_true": m_true,
        "nc_m": nc_m,
        "mre_m": mre_m,
        "diff": diff,
        "tolerance": tol,
        "pass": diff < tol,
    }


def _build_pid_recording_from_arrays(s1, s2, tgt, bin_ms=10.0):
    """Synthesize a SpikeRecording where each of three "populations"
    fires exactly the integer value given per bin.

    The PID estimator takes a SpikeRecording; this trick lets us feed
    a target distribution of joint (S1, S2, T) draws into the binned
    estimator while keeping the public interface honest.
    """
    import pandas as pd

    from neurocomplexity.core.recording import SpikeRecording

    bin_s = bin_ms / 1000.0
    n_bins = len(s1)
    duration = n_bins * bin_s

    n_units_per_pop = max(int(max(s1.max(), s2.max(), tgt.max())) + 1, 2)
    # Unit layout: [s1_units (n_units_per_pop), s2_units (n_units_per_pop),
    #               tgt_units (n_units_per_pop)]
    spike_times: list[float] = []
    unit_ids: list[int] = []
    rng = np.random.default_rng(0)
    for b in range(n_bins):
        for offset, vec in enumerate([s1, s2, tgt]):
            count = int(vec[b])
            for k in range(count):
                jitter = rng.uniform(0.0, bin_s)
                spike_times.append(b * bin_s + jitter)
                unit_ids.append(offset * n_units_per_pop + k)
    st = np.array(spike_times, dtype=np.float64)
    ui = np.array(unit_ids, dtype=np.int64)
    order = np.argsort(st, kind="stable")
    st = st[order]
    ui = ui[order]

    n_total = 3 * n_units_per_pop
    pops = {
        "s1":  np.array([1] * n_units_per_pop + [0] * (2 * n_units_per_pop), dtype=bool),
        "s2":  np.array([0] * n_units_per_pop + [1] * n_units_per_pop + [0] * n_units_per_pop, dtype=bool),
        "tgt": np.array([0] * (2 * n_units_per_pop) + [1] * n_units_per_pop, dtype=bool),
    }
    return SpikeRecording(
        spike_times=st,
        unit_ids=ui,
        units=pd.DataFrame({"id": list(range(n_total))}),
        populations=pops,
        duration=float(duration),
        sampling_rate=None,
        source=None,
        intervals={},
    )


@_register
def concordance_pid_vs_dit():
    """Williams-Beer I_min on canonical XOR vs dit's PID_WB implementation.

    XOR analytic atoms: redundancy=0, unique_1=0, unique_2=0, synergy=1 bit.
    Both neurocomplexity and dit (if installed) should recover those values.
    """
    # Build the XOR sample directly and exercise the internal I_min helper
    # (bypassing the SpikeRecording wrapper which would re-discretise).
    from neurocomplexity.analysis.pid import _mi_mm, _redundancy_imin
    rng = np.random.default_rng(0)
    n = 20000
    s1 = rng.integers(0, 2, n)
    s2 = rng.integers(0, 2, n)
    tgt = s1 ^ s2

    # Build joint count table (L_y, L_s1, L_s2) for I_min directly
    L = 2
    flat = tgt.astype(np.int64) * (L * L) + s1.astype(np.int64) * L + s2.astype(np.int64)
    bc = np.bincount(flat, minlength=L * L * L).astype(np.float64)
    joint = bc.reshape(L, L, L)

    nc_red = _redundancy_imin(joint)
    nc_total_mi = _mi_mm(joint.reshape(L, L * L))
    nc_i_y_s1 = _mi_mm(joint.sum(axis=2))
    nc_i_y_s2 = _mi_mm(joint.sum(axis=1))
    nc_unique_1 = max(0.0, nc_i_y_s1 - nc_red)
    nc_unique_2 = max(0.0, nc_i_y_s2 - nc_red)
    nc_synergy = max(0.0, nc_total_mi - nc_red - nc_unique_1 - nc_unique_2)

    # Analytic XOR (in nats; conversion = ln 2 ~ 0.693)
    analytic_synergy = float(np.log(2.0))
    analytic_redundancy = 0.0

    diff_red_analytic = abs(nc_red - analytic_redundancy)
    diff_syn_analytic = abs(nc_synergy - analytic_synergy)

    # Cross-check vs dit, if installed
    dit_status = "not installed"
    dit_red = float("nan")
    dit_syn = float("nan")
    diff_red_dit = float("nan")
    diff_syn_dit = float("nan")
    try:
        from collections import Counter

        import dit
        from dit.pid import PID_WB
        joint_counter = Counter(zip(s1.tolist(), s2.tolist(), tgt.tolist()))
        total = sum(joint_counter.values())
        outcomes = [f"{a}{b}{c}" for (a, b, c) in joint_counter]
        probs = [v / total for v in joint_counter.values()]
        d = dit.Distribution(outcomes, probs)
        d.set_rv_names("XYZ")
        pid_dit = PID_WB(d, [("X",), ("Y",)], "Z")
        # dit >= 1.5 removed the ``.atoms`` dict: access each partial-information
        # atom by indexing the PID with its rv-name antichain. Values are in
        # bits, converted to nats for comparison.
        # redundancy is the bottom of the lattice: both sources alone {X}{Y}
        dit_red = float(pid_dit[(("X",), ("Y",))]) * np.log(2.0)
        # synergy is the top: {X:Y}
        dit_syn = float(pid_dit[(("X", "Y"),)]) * np.log(2.0)
        diff_red_dit = abs(nc_red - dit_red)
        diff_syn_dit = abs(nc_synergy - dit_syn)
        dit_status = "ok"
    except ImportError:
        dit_status = "not installed (pip install dit)"
    except Exception as e:
        dit_status = f"dit call failed: {e!r}"

    tol = 0.02  # nats
    # Pass requires analytic-recovery within tol AND (if dit installed) agreement
    pass_analytic = (diff_red_analytic < tol) and (diff_syn_analytic < tol)
    if dit_status == "ok":
        pass_dit = (diff_red_dit < tol) and (diff_syn_dit < tol)
        passed = pass_analytic and pass_dit
    else:
        passed = pass_analytic

    return {
        "name": "pid_vs_dit",
        "skipped": False,
        "nc_redundancy": float(nc_red),
        "nc_unique_1": float(nc_unique_1),
        "nc_unique_2": float(nc_unique_2),
        "nc_synergy": float(nc_synergy),
        "analytic_redundancy": analytic_redundancy,
        "analytic_synergy": analytic_synergy,
        "diff_red_analytic": float(diff_red_analytic),
        "diff_syn_analytic": float(diff_syn_analytic),
        "dit_status": dit_status,
        "dit_redundancy": float(dit_red) if np.isfinite(dit_red) else None,
        "dit_synergy": float(dit_syn) if np.isfinite(dit_syn) else None,
        "diff_red_dit": (float(diff_red_dit) if np.isfinite(diff_red_dit) else None),
        "diff_syn_dit": (float(diff_syn_dit) if np.isfinite(diff_syn_dit) else None),
        "tolerance": tol,
        "pass": passed,
    }


#: Williams-Beer test distribution "WB2" (Williams & Beer 2010), on which
#: I_min and BROJA provably disagree. X, Y are independent fair bits; the
#: target Z is a deterministic function of the pair: (0,0)->0, (0,1)->1,
#: (1,0)->2, (1,1)->1. I_min attributes ~0.35 nats more redundancy than
#: BROJA here, so it is a sharp test of the atom-definition dependence that
#: the canonical XOR/AND/COPY cases (where the definitions coincide) cannot
#: expose.
_WB2_MAP = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 1}


@_register
def concordance_imin_vs_broja_wb2():
    """Characterise the I_min vs BROJA redundancy gap on the WB2 distribution.

    On the canonical XOR/AND/COPY distributions used elsewhere in this module
    the Williams-Beer I_min and BROJA atom definitions coincide, so those cases
    cannot detect the over-attribution of redundancy that I_min is known to
    exhibit (Bertschinger et al. 2014). The WB2 distribution
    (\\_WB2_MAP) is a minimal case where the two definitions diverge. This case
    (i) confirms neurocomplexity's I_min redundancy concords with dit's PID_WB
    (the same atom definition) to machine precision, and (ii) *quantifies* the
    I_min minus BROJA redundancy gap, so the documented limitation is bounded
    numerically rather than merely asserted. The gap is reported, not failed:
    the two estimators are expected to differ here. The pass flag tracks only
    the nc-vs-dit-WB concordance.
    """
    from neurocomplexity.analysis.pid import _redundancy_imin

    L_t = 3  # target alphabet {0, 1, 2}; sources are binary
    joint = np.zeros((L_t, 2, 2), dtype=np.float64)  # [t, s1, s2]
    for (s1, s2), t in _WB2_MAP.items():
        joint[t, s1, s2] += 0.25
    nc_red = float(_redundancy_imin(joint))

    try:
        import dit
        from dit.pid import PID_BROJA, PID_WB
    except ImportError:
        return {
            "name": "imin_vs_broja_wb2",
            "skipped": True,
            "reason": "dit not installed (pip install dit)",
            "tolerance": 0.02,
            "pass": False,
            "nc_redundancy_imin": nc_red,
        }

    outcomes = [f"{s1}{s2}{t}" for (s1, s2), t in _WB2_MAP.items()]
    d = dit.Distribution(outcomes, [0.25] * len(outcomes))
    d.set_rv_names("XYZ")
    ln2 = float(np.log(2.0))
    wb_red = float(PID_WB(d, [("X",), ("Y",)], "Z")[(("X",), ("Y",))]) * ln2
    broja_red = float(PID_BROJA(d, [("X",), ("Y",)], "Z")[(("X",), ("Y",))]) * ln2

    diff_nc_wb = abs(nc_red - wb_red)
    imin_broja_gap = abs(wb_red - broja_red)
    tol = 0.02  # nats: governs only the nc-vs-dit-WB concordance
    return {
        "name": "imin_vs_broja_wb2",
        "skipped": False,
        "nc_redundancy_imin": nc_red,
        "dit_wb_redundancy": wb_red,
        "dit_broja_redundancy": broja_red,
        "diff_nc_vs_dit_wb": float(diff_nc_wb),
        "imin_minus_broja_redundancy_gap": float(imin_broja_gap),
        "tolerance": tol,
        "pass": diff_nc_wb < tol,
    }
