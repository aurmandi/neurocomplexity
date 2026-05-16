"""Synthetic populations with rank-r covariance for participation-ratio benchmarks.

The latent factor model ``X = L Z + noise * E`` produces binned activity
whose covariance is rank-``rank`` plus a small diagonal noise floor. After
Poisson thinning the resulting spike-train recording has participation
ratio approaching ``rank`` as ``noise → 0``.

Reference:
    Rajan K, Abbott LF, Sompolinsky H (2010). "Stimulus-dependent
    suppression of chaos in recurrent neural networks." Phys. Rev. E 82,
    011903.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from neurocomplexity.core.recording import SpikeRecording


def rank_r_population(
    *,
    n_units: int = 50,
    rank: int,
    n_bins: int = 5_000,
    bin_ms: float = 10.0,
    noise: float = 0.1,
    base_rate_hz: float = 5.0,
    modulation: float = 0.5,
    seed: int | None = None,
) -> SpikeRecording:
    """Generate a synthetic population with rank-``rank`` covariance.

    The latent activity ``X = L @ Z + noise * E`` is mapped to per-bin
    Poisson rates via a half-rectified affine, then thinned to spikes.
    """
    rng = np.random.default_rng(seed)
    bin_s = bin_ms / 1000.0
    L = rng.normal(0, 1, size=(n_units, rank))
    Z = rng.normal(0, 1, size=(rank, n_bins))
    E = rng.normal(0, 1, size=(n_units, n_bins))
    X = L @ Z + noise * E
    rates = base_rate_hz * (1.0 + modulation * X / (np.std(X) + 1e-9))
    rates = np.clip(rates, 0.1, None)
    counts = rng.poisson(rates * bin_s)

    times_chunks: list[np.ndarray] = []
    uids_chunks: list[np.ndarray] = []
    centres = (np.arange(n_bins) + 0.5) * bin_s
    for uid in range(n_units):
        nz_bins = np.flatnonzero(counts[uid])
        for b in nz_bins:
            c = int(counts[uid, b])
            t = centres[b] + rng.uniform(-bin_s / 2, bin_s / 2, c)
            times_chunks.append(t)
            uids_chunks.append(np.full(c, uid, dtype=np.int64))
    if times_chunks:
        st = np.concatenate(times_chunks)
        uu = np.concatenate(uids_chunks)
        order = np.argsort(st, kind="stable")
        st = st[order]
        uu = uu[order]
    else:
        st = np.array([], dtype=np.float64)
        uu = np.array([], dtype=np.int64)

    return SpikeRecording(
        spike_times=st,
        unit_ids=uu,
        units=pd.DataFrame({"id": list(range(n_units))}),
        populations={"all": np.ones(n_units, dtype=bool)},
        duration=float(n_bins * bin_s),
        sampling_rate=None,
        source=None,
        intervals={},
    )
