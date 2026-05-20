"""TE / PID interaction with ContinuousSignal via signals= kwarg."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neurocomplexity.analysis.pid import partial_information
from neurocomplexity.analysis.transfer_entropy import transfer_entropy
from neurocomplexity.core.continuous import ContinuousSignal
from neurocomplexity.core.provenance import ProvenanceRecord
from neurocomplexity.core.recording import SpikeRecording


def _rec_with_stim_driven_population(duration=120.0, seed=0):
    """Stim is a square-wave; population 'driven' fires only when stim is high."""
    rng = np.random.default_rng(seed)
    n_units = 6
    units = pd.DataFrame({"id": np.arange(n_units, dtype=np.int64)})
    fs = 100.0
    n_samples = int(duration * fs)
    # 0.5 Hz square wave for stim
    t_grid = np.arange(n_samples) / fs
    stim_values = (np.sign(np.sin(2 * np.pi * 0.5 * t_grid)) > 0).astype(float)
    pop_a_units = np.array([True, True, True, False, False, False])
    pop_b_units = ~pop_a_units

    times = []
    owners = []
    # population a (driven): high firing only when stim is 1
    for u in np.where(pop_a_units)[0]:
        gate = stim_values
        lam = (5.0 + 25.0 * gate) / fs  # 5 Hz baseline, 30 Hz when stim high
        counts = rng.poisson(lam)
        bin_starts = t_grid
        t_unit = []
        for i, c in enumerate(counts):
            if c:
                t_unit.append(rng.uniform(bin_starts[i], bin_starts[i] + 1 / fs, c))
        if t_unit:
            tt = np.sort(np.concatenate(t_unit))
        else:
            tt = np.empty(0)
        times.append(tt)
        owners.append(np.full(tt.size, u, dtype=np.int64))
    # population b: independent Poisson
    for u in np.where(pop_b_units)[0]:
        n = rng.poisson(8.0 * duration)
        tt = np.sort(rng.uniform(0, duration, size=n))
        times.append(tt)
        owners.append(np.full(tt.size, u, dtype=np.int64))

    st = np.concatenate(times)
    uid = np.concatenate(owners)
    order = np.argsort(st, kind="stable")
    rec = SpikeRecording(
        spike_times=st[order].astype(np.float64),
        unit_ids=uid[order],
        units=units,
        populations={"driven": pop_a_units, "indep": pop_b_units},
        duration=float(duration),
        sampling_rate=None,
        source=ProvenanceRecord.for_memory("test"),
    )
    sig = ContinuousSignal(values=stim_values, sampling_rate=fs, label="stim")
    return rec.with_signal("stim", sig)


def test_te_runs_with_signal_argument():
    rec = _rec_with_stim_driven_population()
    res = transfer_entropy(rec, populations=["driven", "indep"],
                          signals=["stim"], bin_size_ms=10.0)
    assert res.matrix.shape == (3, 3)
    assert res.populations == ("driven", "indep", "stim")
    assert res.params["signals"] == ["stim"]


def test_te_signal_drives_population_more_than_population_drives_signal():
    rec = _rec_with_stim_driven_population()
    res = transfer_entropy(rec, populations=["driven", "indep"],
                          signals=["stim"], bin_size_ms=10.0)
    names = res.populations
    s_idx = names.index("stim")
    d_idx = names.index("driven")
    i_idx = names.index("indep")
    # stim -> driven should exceed driven -> stim
    assert res.matrix[s_idx, d_idx] > res.matrix[d_idx, s_idx]
    # stim -> indep should be roughly equal to indep -> stim (both small)
    assert res.matrix[s_idx, d_idx] > res.matrix[s_idx, i_idx]


def test_te_rejects_unknown_signal_name():
    rec = _rec_with_stim_driven_population()
    with pytest.raises(ValueError, match="unknown signal"):
        transfer_entropy(rec, populations=["driven", "indep"],
                        signals=["nope"], bin_size_ms=10.0)


def test_te_existing_population_only_behaviour_unchanged():
    rec = _rec_with_stim_driven_population()
    res = transfer_entropy(rec, populations=["driven", "indep"], bin_size_ms=10.0)
    assert res.matrix.shape == (2, 2)
    assert res.populations == ("driven", "indep")
    assert res.params.get("signals", []) == []


def test_pid_with_signal_as_source():
    rec = _rec_with_stim_driven_population()
    res = partial_information(rec,
                              target_pop="driven",
                              sources=["indep", "stim"],
                              bin_size_ms=10.0,
                              n_levels=2)
    # All atoms finite and non-negative
    assert np.isfinite(res.redundancy)
    assert np.isfinite(res.unique_1)
    assert np.isfinite(res.unique_2)
    assert np.isfinite(res.synergy)
    assert res.redundancy >= 0
    assert res.unique_1 >= 0
    assert res.unique_2 >= 0
    assert res.synergy >= 0


def test_pid_unknown_stream_name_raises():
    rec = _rec_with_stim_driven_population()
    with pytest.raises(ValueError, match="unknown stream"):
        partial_information(rec, target_pop="nope",
                          sources=["indep", "stim"], bin_size_ms=10.0)


def test_nwb_roundtrip_preserves_signals(tmp_path):
    pytest.importorskip("pynwb")
    from neurocomplexity import io as nc_io
    rec = _rec_with_stim_driven_population()
    p = tmp_path / "sig.nwb"
    nc_io.to_nwb(rec, p)
    rec2 = nc_io.from_nwb(p)
    assert set(rec2.signals.keys()) == set(rec.signals.keys())
    for name in rec.signals:
        a, b = rec.signals[name], rec2.signals[name]
        np.testing.assert_array_equal(a.values, b.values)
        assert a.sampling_rate == b.sampling_rate
        assert a.t_start == b.t_start
        assert a.label == b.label
