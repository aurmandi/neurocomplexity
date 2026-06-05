from dataclasses import replace as dc_replace

import numpy as np
import pytest
import pandas as pd
from neurocomplexity.core.recording import SpikeRecording
from neurocomplexity.inference.surrogates import spike_dither, isi_shuffle, interval_shuffle


def _make_rec(rates=(10.0, 12.0), duration=60.0, seed=1):
    rng = np.random.default_rng(seed)
    times, uids = [], []
    for uid, r in enumerate(rates):
        n = rng.poisson(r * duration)
        t = np.sort(rng.uniform(0, duration, n))
        times.append(t)
        uids.append(np.full(n, uid, dtype=np.int64))
    spike_times = np.concatenate(times)
    unit_ids = np.concatenate(uids)
    order = np.argsort(spike_times)
    return SpikeRecording(
        spike_times=spike_times[order], unit_ids=unit_ids[order],
        units=pd.DataFrame({"id": list(range(len(rates)))}),
        populations={"all": np.ones(len(rates), dtype=bool)},
        duration=duration, sampling_rate=None, source=None, intervals={},
    )


def test_spike_dither_preserves_count():
    rec = _make_rec()
    surr = spike_dither(rec, delta_ms=5.0, seed=0)
    assert surr.spike_times.size == rec.spike_times.size


def test_spike_dither_keeps_rate_within_tolerance():
    rec = _make_rec()
    surr = spike_dither(rec, delta_ms=5.0, seed=0)
    bins = np.arange(0, rec.duration + 1.0, 1.0)
    h_orig, _ = np.histogram(rec.spike_times, bins=bins)
    h_surr, _ = np.histogram(surr.spike_times, bins=bins)
    rms = np.sqrt(np.mean((h_orig - h_surr) ** 2)) / (np.mean(h_orig) + 1e-9)
    assert rms < 0.5


def test_spike_dither_respects_refractory_when_repaired():
    rec = _make_rec(rates=(100.0,))
    surr = spike_dither(rec, delta_ms=10.0, seed=0, repair_refractory_ms=1.0)
    for uid in np.unique(surr.unit_ids):
        t = np.sort(surr.spike_times[surr.unit_ids == uid])
        if t.size > 1:
            assert np.min(np.diff(t)) >= 0.001 - 1e-9


def test_spike_dither_reproducible():
    rec = _make_rec()
    s1 = spike_dither(rec, delta_ms=5.0, seed=42)
    s2 = spike_dither(rec, delta_ms=5.0, seed=42)
    assert np.array_equal(s1.spike_times, s2.spike_times)
    assert np.array_equal(s1.unit_ids, s2.unit_ids)


def test_isi_shuffle_preserves_isi_distribution():
    rec = _make_rec(rates=(15.0,), duration=120.0)
    surr = isi_shuffle(rec, seed=0)
    isi_orig = np.sort(np.diff(np.sort(rec.spike_times[rec.unit_ids == 0])))
    isi_surr = np.sort(np.diff(np.sort(surr.spike_times[surr.unit_ids == 0])))
    assert np.allclose(isi_orig, isi_surr, atol=1e-12)


def test_isi_shuffle_preserves_count_per_unit():
    rec = _make_rec(rates=(8.0, 20.0, 5.0))
    surr = isi_shuffle(rec, seed=0)
    for uid in (0, 1, 2):
        assert (rec.unit_ids == uid).sum() == (surr.unit_ids == uid).sum()


def test_isi_shuffle_reproducible():
    rec = _make_rec()
    s1 = isi_shuffle(rec, seed=7)
    s2 = isi_shuffle(rec, seed=7)
    assert np.array_equal(s1.spike_times, s2.spike_times)


def _rec_with_intervals(n_trials=20, trial_s=1.0, rates=(10.0, 10.0), seed=2):
    rng = np.random.default_rng(seed)
    starts = np.arange(n_trials) * (trial_s + 0.5)
    stops = starts + trial_s
    duration = float(stops[-1] + 0.5)
    times, uids = [], []
    for uid, r in enumerate(rates):
        all_t = []
        for s, e in zip(starts, stops):
            n = rng.poisson(r * trial_s)
            all_t.append(rng.uniform(s, e, n))
        all_t = np.concatenate(all_t) if all_t else np.array([])
        all_t.sort()
        times.append(all_t)
        uids.append(np.full(all_t.size, uid, dtype=np.int64))
    spike_times = np.concatenate(times)
    unit_ids = np.concatenate(uids)
    order = np.argsort(spike_times)
    intervals = {"trials": pd.DataFrame({"start_time": starts, "stop_time": stops})}
    return SpikeRecording(
        spike_times=spike_times[order], unit_ids=unit_ids[order],
        units=pd.DataFrame({"id": list(range(len(rates)))}),
        populations={"all": np.ones(len(rates), dtype=bool)},
        duration=duration, sampling_rate=None, source=None, intervals=intervals,
    )


def test_interval_shuffle_preserves_total_count_per_unit():
    rec = _rec_with_intervals()
    surr = interval_shuffle(rec, "trials", seed=0)
    for uid in (0, 1):
        assert (rec.unit_ids == uid).sum() == (surr.unit_ids == uid).sum()


def test_interval_shuffle_requires_named_table():
    rec = _rec_with_intervals()
    with pytest.raises(KeyError):
        interval_shuffle(rec, "nope", seed=0)


def test_interval_shuffle_rejects_overlapping_intervals():
    """Overlap would silently re-assign the same spike twice."""
    rec = _rec_with_intervals()
    df = rec.intervals["trials"].copy()
    # Extend interval 5 past start of interval 6 by 20 ms
    j_stop = df.columns.get_loc("stop_time")
    j_start = df.columns.get_loc("start_time")
    df.iloc[5, j_stop] = df.iloc[6, j_start] + 0.02
    rec_bad = dc_replace(rec, intervals={"trials": df})
    with pytest.raises(ValueError, match="non-overlapping"):
        interval_shuffle(rec_bad, "trials", seed=0)


def test_interval_shuffle_accepts_touching_intervals():
    """Back-to-back intervals (stop[i] == start[i+1]) must NOT raise."""
    rec = _rec_with_intervals()
    df = rec.intervals["trials"].copy()
    starts = np.arange(len(df)) * 1.0
    df["start_time"] = starts
    df["stop_time"] = starts + 1.0
    rec_touch = dc_replace(rec, intervals={"trials": df},
                           duration=float(df["stop_time"].iloc[-1] + 0.1))
    surr = interval_shuffle(rec_touch, "trials", seed=0)
    assert surr is not None


