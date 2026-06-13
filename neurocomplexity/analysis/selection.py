"""Auditable K-selection helpers for spike-train analyses.

These were previously inlined in example scripts (notably the Fig 6 driver
``examples/integration_session_715093703_spont.py`` and the gitignored
``datasets/window_search.py``). Hiding the K rule in script-side code
makes the resulting analyses unauditable: a future user copying the
pattern silently inherits the rate-floor / rate-cap choices. Moving them
to the public API ``neurocomplexity.analysis`` makes the selection logic
testable and documentable.

References
----------
Timme N M, Lapish C (2018). A tutorial for information theory in
neuroscience. *eNeuro* 5(3) -- recommends a rate-matched-subsample
control for selection-induced inflation in spike-train TE.
"""
from __future__ import annotations

import numpy as np

from neurocomplexity.core.recording import SpikeRecording


def _rates(rec: SpikeRecording) -> dict[int, float]:
    out: dict[int, float] = {}
    for uid in np.unique(rec.unit_ids):
        n = int((rec.unit_ids == uid).sum())
        out[int(uid)] = n / rec.duration
    return out


def _restrict_to_area(rec: SpikeRecording, area: str | None) -> SpikeRecording:
    if area is None:
        return rec
    if area not in rec.populations:
        raise KeyError(f"area {area!r} not in populations: "
                       f"{list(rec.populations)}")
    return rec.with_populations(
        {area: rec.populations[area]}, on_unassigned="drop",
    )


def top_firing_band(
    rec: SpikeRecording,
    *,
    K: int,
    rate_min: float,
    rate_max: float,
    area: str | None = None,
) -> list[int]:
    """Return up to ``K`` unit ids whose mean firing rate over ``rec.duration``
    lies in ``[rate_min, rate_max]``, sorted by descending rate.

    Parameters
    ----------
    rec
        Recording to draw units from.
    K
        Maximum number of units to return. Must be ``>= 1``.
    rate_min, rate_max
        Inclusive rate band in Hz. ``rate_min`` must be ``< rate_max``.
    area
        Optional population name. When supplied, only units in
        ``rec.populations[area]`` are considered.

    Returns
    -------
    list[int]
        Up to ``K`` unit ids. Empty if no qualifying units exist.

    Notes
    -----
    Selection is deterministic. For randomised tie-break (e.g. for a
    selection-robustness control), use :func:`rate_matched_subsample`.
    """
    if K < 1:
        raise ValueError("K must be >= 1")
    if not (rate_min < rate_max):
        raise ValueError(f"need rate_min < rate_max; got {rate_min}, {rate_max}")
    sub = _restrict_to_area(rec, area)
    rates = _rates(sub)
    cand = [(r, uid) for uid, r in rates.items() if rate_min <= r <= rate_max]
    cand.sort(reverse=True)
    return [uid for _, uid in cand[:K]]


def rate_matched_subsample(
    rec: SpikeRecording,
    *,
    K: int,
    rate_min: float,
    rate_max: float,
    seed: int | None = None,
    area: str | None = None,
) -> list[int]:
    """Return ``K`` unit ids drawn uniformly from the rate band.

    Use as a selection-robustness control against the deterministic
    :func:`top_firing_band`: if a downstream estimate (TE matrix,
    significant-edge count) is invariant under a rate-matched random
    subsample of the same band, the result is not driven by the
    top-K rule.
    """
    if K < 1:
        raise ValueError("K must be >= 1")
    if not (rate_min < rate_max):
        raise ValueError(f"need rate_min < rate_max; got {rate_min}, {rate_max}")
    sub = _restrict_to_area(rec, area)
    rates = _rates(sub)
    pool = [uid for uid, r in rates.items() if rate_min <= r <= rate_max]
    if len(pool) <= K:
        return sorted(pool)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.asarray(pool, dtype=np.int64),
                        size=K, replace=False)
    return sorted(int(u) for u in chosen)
