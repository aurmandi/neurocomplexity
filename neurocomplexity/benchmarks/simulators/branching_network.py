"""Discrete-time critical / sub-critical branching-process simulator.

At each time-step a Poisson number of "offspring" spikes is generated with
mean equal to ``m * (population spike count in the previous bin)``;
offspring are distributed uniformly across units. Per-bin offspring is
capped at ``n_units`` (saturation: each unit may emit at most one spike
per bin, matching cortical refractoriness) which keeps the simulator
memory-safe at and above criticality. An external Poisson drive per unit
seeds the activity floor so the process never dies out in the sub-critical
regime.

Memory footprint is dominated by ``counts``, a single ``(n_units, n_bins)``
int8 array. For 120 units × 60 s at 4 ms bins this is ~1.8 MB.

Reference
---------
Wilting J, Priesemann V (2018). "Inferring collective dynamical states
from widely unobserved systems." Nature Communications 9, 2325.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from neurocomplexity.core.recording import SpikeRecording


def branching_network(
    *,
    n_units: int = 100,
    m: float,
    duration_s: float = 600.0,
    bin_ms: float = 4.0,
    external_rate_hz: float = 0.5,
    saturate: bool = True,
    seed: int | None = None,
) -> SpikeRecording:
    """Simulate a population-wide branching process and return a SpikeRecording.

    Parameters
    ----------
    n_units : int
        Number of units in the population.
    m : float
        Branching ratio (expected offspring per parent spike); ``m < 1``
        sub-critical, ``m = 1`` critical, ``m > 1`` super-critical. With
        positive external drive the stationary regime is strictly
        ``m < 1``; ``m >= 1`` will produce activity at the saturation
        cap (one spike per unit per bin).
    duration_s : float
        Recording duration in seconds.
    bin_ms : float
        Simulation time-step; also defines spike-time discretisation.
    external_rate_hz : float
        Per-unit Poisson drive rate (Hz). Use 0.0 with ``m >= 1`` to keep
        the process formally stationary on its own initial seed.
    saturate : bool
        If True (default), per-bin counts are clipped to one spike per
        unit (saturation cap). If False, the unconstrained Galton-Watson
        count-based branching process is simulated — needed to recover
        mean-field avalanche exponents but only safe at modest
        ``n_units * duration_s`` (memory grows with total spike count).
    seed : int | None
        RNG seed for reproducibility.

    Notes
    -----
    With ``saturate=True``, per-bin offspring counts are clipped to
    ``n_units`` (so at most one spike per unit per bin); this acts as a
    refractoriness-like saturation and prevents memory blow-ups for
    ``m`` near or above 1. With ``saturate=False``, the simulator follows
    the standard count-based branching process: each parent spike has
    Poisson(``m``) offspring distributed uniformly across units; spike
    times are jittered within their bin to produce a SpikeRecording.
    """
    rng = np.random.default_rng(seed)
    bin_s = bin_ms / 1000.0
    n_bins = int(duration_s / bin_s)

    if saturate:
        return _simulate_saturated(
            rng, n_units, n_bins, bin_s, m, external_rate_hz, duration_s,
        )
    return _simulate_unsaturated(
        rng, n_units, n_bins, bin_s, m, external_rate_hz, duration_s,
    )


def _simulate_saturated(rng, n_units, n_bins, bin_s, m, external_rate_hz, duration_s):
    counts = np.zeros((n_units, n_bins), dtype=np.int8)
    p_ext_per_unit = external_rate_hz * bin_s
    counts[:, 0] = (rng.random(n_units) < p_ext_per_unit).astype(np.int8)
    for t in range(1, n_bins):
        parent_total = int(counts[:, t - 1].sum())
        n_offspring = int(rng.poisson(m * parent_total)) if parent_total > 0 else 0
        n_offspring = min(n_offspring, n_units)
        if n_offspring > 0:
            who = rng.choice(n_units, size=n_offspring, replace=False)
            counts[who, t] = 1
        if p_ext_per_unit > 0:
            silent = counts[:, t] == 0
            if silent.any():
                ext_fires = rng.random(int(silent.sum())) < p_ext_per_unit
                idx_silent = np.flatnonzero(silent)
                counts[idx_silent[ext_fires], t] = 1
    unit_idx, bin_idx = np.nonzero(counts)
    return _counts_to_recording(rng, unit_idx, bin_idx, n_units, bin_s, duration_s)


def _simulate_unsaturated(rng, n_units, n_bins, bin_s, m, external_rate_hz, duration_s):
    """Unbounded count-based Galton-Watson branching process.

    Per-bin spike counts can exceed n_units. Memory scales with total
    spike count; safe only at modest n_units * duration_s.
    """
    counts = np.zeros((n_units, n_bins), dtype=np.int32)
    p_ext_per_unit = external_rate_hz * bin_s
    if p_ext_per_unit > 0:
        counts[:, 0] = rng.poisson(p_ext_per_unit, size=n_units)
    for t in range(1, n_bins):
        parent_total = int(counts[:, t - 1].sum())
        if parent_total > 0:
            n_offspring = int(rng.poisson(m * parent_total))
            if n_offspring > 0:
                targets = rng.integers(0, n_units, size=n_offspring)
                np.add.at(counts[:, t], targets, 1)
        if p_ext_per_unit > 0:
            counts[:, t] += rng.poisson(p_ext_per_unit, size=n_units).astype(np.int32)

    # Convert per-bin counts to (unit, bin) pairs, one entry per spike.
    times_chunks: list[np.ndarray] = []
    uids_chunks: list[np.ndarray] = []
    for u in range(n_units):
        nz = np.flatnonzero(counts[u])
        for b in nz:
            c = int(counts[u, b])
            jitter = rng.uniform(0.0, bin_s, c)
            times_chunks.append(b * bin_s + jitter)
            uids_chunks.append(np.full(c, u, dtype=np.int64))
    if times_chunks:
        st = np.concatenate(times_chunks)
        uu = np.concatenate(uids_chunks)
        order = np.argsort(st, kind="stable")
        return _build_recording(st[order], uu[order], n_units, duration_s)
    return _build_recording(
        np.array([], dtype=np.float64),
        np.array([], dtype=np.int64),
        n_units, duration_s,
    )


def _counts_to_recording(rng, unit_idx, bin_idx, n_units, bin_s, duration_s):
    if unit_idx.size:
        jitter = rng.uniform(0.0, bin_s, size=unit_idx.size)
        spike_times = bin_idx * bin_s + jitter
        unit_ids = unit_idx.astype(np.int64)
        order = np.argsort(spike_times, kind="stable")
        spike_times = spike_times[order]
        unit_ids = unit_ids[order]
    else:
        spike_times = np.array([], dtype=np.float64)
        unit_ids = np.array([], dtype=np.int64)
    return _build_recording(spike_times, unit_ids, n_units, duration_s)


def _build_recording(spike_times, unit_ids, n_units, duration_s):

    return SpikeRecording(
        spike_times=spike_times,
        unit_ids=unit_ids,
        units=pd.DataFrame({"id": list(range(n_units))}),
        populations={"all": np.ones(n_units, dtype=bool)},
        duration=float(duration_s),
        sampling_rate=None,
        source=None,
        intervals={},
    )


def trial_based_avalanches(
    *,
    n_units: int = 50,
    n_trials: int = 5000,
    bin_ms: float = 4.0,
    inter_trial_quiet_bins: int = 5,
    m: float = 1.0,
    max_trial_bins: int = 200,
    seed: int | None = None,
) -> SpikeRecording:
    """Concatenate many independent Galton-Watson avalanche trials.

    Each trial seeds 1 unit with a single spike, propagates an unbounded
    branching process at branching ratio ``m`` until the activity dies
    out (or ``max_trial_bins`` is reached), and is separated from the
    next trial by ``inter_trial_quiet_bins`` empty bins so the criticality
    analysis can cleanly delineate avalanches.

    At m=1 (exact mean-field branching) the population-wide pooled
    avalanche-size distribution follows P(s) ~ s^{-3/2} and the
    duration distribution follows P(t) ~ t^{-2} — the universal
    Galton-Watson exponents. Used by the criticality.exponents benchmark
    case.

    Reference: Galton-Watson branching process; Sethna et al. (2001),
    "Crackling noise", Nature 410, 242.
    """
    rng = np.random.default_rng(seed)
    bin_s = bin_ms / 1000.0

    # Each row: a list of per-bin spike counts for one trial.
    times_chunks: list[np.ndarray] = []
    uids_chunks: list[np.ndarray] = []
    current_bin = 0
    for trial in range(n_trials):
        # Seed a single spike in a random unit.
        counts_t = [np.zeros(n_units, dtype=np.int32)]
        seed_unit = int(rng.integers(0, n_units))
        counts_t[0][seed_unit] = 1
        # Propagate until extinction or max length.
        parent_total = 1
        b = 0
        while parent_total > 0 and b < max_trial_bins - 1:
            b += 1
            n_offspring = int(rng.poisson(m * parent_total))
            new_bin = np.zeros(n_units, dtype=np.int32)
            if n_offspring > 0:
                targets = rng.integers(0, n_units, size=n_offspring)
                np.add.at(new_bin, targets, 1)
            counts_t.append(new_bin)
            parent_total = int(new_bin.sum())
        # Emit spike times for this trial.
        for trial_bin, vec in enumerate(counts_t):
            nz = np.flatnonzero(vec)
            for u in nz:
                c = int(vec[u])
                jitter = rng.uniform(0.0, bin_s, c)
                times_chunks.append((current_bin + trial_bin) * bin_s + jitter)
                uids_chunks.append(np.full(c, u, dtype=np.int64))
        current_bin += len(counts_t) + inter_trial_quiet_bins

    duration_s = current_bin * bin_s
    if times_chunks:
        st = np.concatenate(times_chunks)
        uu = np.concatenate(uids_chunks)
        order = np.argsort(st, kind="stable")
        st = st[order]
        uu = uu[order]
    else:
        st = np.array([], dtype=np.float64)
        uu = np.array([], dtype=np.int64)
    return _build_recording(st, uu, n_units, duration_s)
