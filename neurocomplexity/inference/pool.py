"""Lazy, reproducible surrogate pool.

A :class:`SurrogatePool` generates ``N`` surrogate recordings on demand
with reproducible per-draw seeds derived from a single master seed via
:class:`numpy.random.SeedSequence`. Drawn surrogates are kept in a small
LRU cache so re-accessing the same index does not regenerate the data.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np

from neurocomplexity.core.recording import SpikeRecording
from neurocomplexity.inference.surrogates import (
    interval_shuffle,
    isi_shuffle,
    spike_dither,
)

_METHODS = {
    "spike_dither": spike_dither,
    "isi_shuffle": isi_shuffle,
    "interval_shuffle": interval_shuffle,
}


class SurrogatePool:
    """Lazy, reproducible pool of ``n`` surrogate recordings.

    Surrogates are generated on demand by indexing the pool
    (``pool[i]``). Each draw uses a child seed deterministically derived
    from the master ``seed``, so any subset of indices is reproducible.

    Parameters
    ----------
    rec
        Source recording to surrogate.
    surrogate
        Name of the surrogate method. One of:

        * ``"spike_dither"`` — Louis-Gerstein-Grün uniform spike dithering.
          Kwargs: ``delta_ms`` (default 5), ``repair_refractory_ms``
          (default 1).
        * ``"isi_shuffle"`` — independent ISI shuffle per unit. No kwargs.
        * ``"interval_shuffle"`` — within-population interval shuffling.
          Kwargs: ``intervals_name`` (required).
    n
        Number of surrogates in the pool.
    seed
        Master RNG seed. Spawned via :class:`numpy.random.SeedSequence`.
    cache_size
        LRU cache capacity for already-drawn surrogates (default 64).
    **method_kwargs
        Forwarded to the surrogate function on every draw.

    Raises
    ------
    ValueError
        On unknown ``surrogate`` name.
    """

    def __init__(
        self,
        rec: SpikeRecording,
        *,
        surrogate: str,
        n: int,
        seed: int,
        cache_size: int = 64,
        **method_kwargs: Any,
    ):
        if surrogate not in _METHODS:
            raise ValueError(f"unknown surrogate method: {surrogate!r}")
        self._rec = rec
        self.method = surrogate
        self.n = int(n)
        self.seed = int(seed)
        self.metadata = dict(method_kwargs)
        self._fn = _METHODS[surrogate]
        self._kwargs = method_kwargs
        self._cache: OrderedDict[int, SpikeRecording] = OrderedDict()
        self._cap = int(cache_size)
        ss = np.random.SeedSequence(seed)
        self._child_seeds = [int(s.generate_state(1)[0]) for s in ss.spawn(self.n)]

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> SpikeRecording:
        if not 0 <= i < self.n:
            raise IndexError(i)
        if i in self._cache:
            self._cache.move_to_end(i)
            return self._cache[i]
        surr = self._fn(self._rec, seed=self._child_seeds[i], **self._kwargs)
        self._cache[i] = surr
        if len(self._cache) > self._cap:
            self._cache.popitem(last=False)
        return surr
