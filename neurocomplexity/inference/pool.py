from __future__ import annotations
from collections import OrderedDict
from typing import Any
import numpy as np

from neurocomplexity.core.recording import SpikeRecording
from neurocomplexity.inference.surrogates import (
    spike_dither, isi_shuffle, interval_shuffle,
)

_METHODS = {
    "spike_dither": spike_dither,
    "isi_shuffle": isi_shuffle,
    "interval_shuffle": interval_shuffle,
}


class SurrogatePool:
    """Lazy, reproducible pool of N surrogate SpikeRecordings."""

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
