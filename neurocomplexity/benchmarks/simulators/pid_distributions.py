"""Canonical bivariate PID distributions encoded as :class:`SpikeRecording`.

Each bin is one i.i.d. draw of ``(s1, s2, t)`` from the named joint
distribution; a unit emits a spike at the bin centre iff its variable is
1 in that bin. Three units (s1, s2, t) are exposed via three
single-element populations ``source_1``, ``source_2``, ``target``, which
the PID benchmark cases consume directly.

Distributions
-------------
xor   : s1, s2 ~ Bern(1/2), t = s1 XOR s2 (pure synergy)
and   : s1, s2 ~ Bern(1/2), t = s1 AND s2 (synergy + redundancy)
copy  : s1, s2 ~ Bern(1/2), t = s1 (unique from s1)
rdn   : s1 ~ Bern(1/2), s2 = s1, t = s1 (pure redundancy)
unq   : s1, s2 ~ Bern(1/2) independent, t = s1 (unique from s1)

References
----------
Williams PL, Beer RD (2010). "Nonnegative decomposition of multivariate
information." arXiv:1004.2515.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from neurocomplexity.core.recording import SpikeRecording


def _sample(distribution: str, n_bins: int, rng: np.random.Generator):
    if distribution == "xor":
        s1 = rng.integers(0, 2, n_bins)
        s2 = rng.integers(0, 2, n_bins)
        t = s1 ^ s2
    elif distribution == "and":
        s1 = rng.integers(0, 2, n_bins)
        s2 = rng.integers(0, 2, n_bins)
        t = s1 & s2
    elif distribution == "copy":
        s1 = rng.integers(0, 2, n_bins)
        s2 = rng.integers(0, 2, n_bins)
        t = s1
    elif distribution == "rdn":
        s1 = rng.integers(0, 2, n_bins)
        s2 = s1.copy()
        t = s1.copy()
    elif distribution == "unq":
        s1 = rng.integers(0, 2, n_bins)
        s2 = rng.integers(0, 2, n_bins)
        t = s1
    else:
        raise ValueError(f"unknown PID distribution: {distribution!r}")
    return s1.astype(np.int8), s2.astype(np.int8), t.astype(np.int8)


def pid_recording(
    distribution: str,
    *,
    n_bins: int = 20_000,
    bin_ms: float = 10.0,
    seed: int | None = None,
) -> SpikeRecording:
    """Encode one of the canonical PID distributions as a SpikeRecording.

    Three units (ids 0, 1, 2) correspond to s1, s2, t respectively; each
    fires at the bin centre when its variable equals 1. Populations
    ``source_1``, ``source_2``, ``target`` (single-unit each) and ``all``
    are provided.
    """
    rng = np.random.default_rng(seed)
    s1, s2, t = _sample(distribution, n_bins, rng)
    bin_s = bin_ms / 1000.0
    centres = (np.arange(n_bins) + 0.5) * bin_s

    times_chunks: list[np.ndarray] = []
    uids_chunks: list[np.ndarray] = []
    for uid, arr in enumerate((s1, s2, t)):
        mask = arr == 1
        times_chunks.append(centres[mask])
        uids_chunks.append(np.full(int(mask.sum()), uid, dtype=np.int64))
    spike_times = np.concatenate(times_chunks)
    unit_ids = np.concatenate(uids_chunks)
    order = np.argsort(spike_times, kind="stable")
    spike_times = spike_times[order]
    unit_ids = unit_ids[order]

    populations = {
        "source_1": np.array([True, False, False]),
        "source_2": np.array([False, True, False]),
        "target": np.array([False, False, True]),
        "all": np.array([True, True, True]),
    }
    units = pd.DataFrame({"id": [0, 1, 2]})
    return SpikeRecording(
        spike_times=spike_times,
        unit_ids=unit_ids,
        units=units,
        populations=populations,
        duration=float(n_bins * bin_s),
        sampling_rate=None,
        source=None,
        intervals={},
    )
