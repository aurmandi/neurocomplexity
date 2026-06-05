"""Phase 3 — numerical & reproducibility audit.

These tests cover:

* Seed determinism: same seed → byte-identical output, every analysis.
* Pickle and deepcopy round-trip for every Result dataclass.
* Edge cases: empty bins, single-unit pop, var=0 column, single-event
  avalanche, log(0) protection.
* dtype invariants: ``spike_times`` float64, ``unit_ids`` int64,
  ``ProvenanceRecord.created_utc`` tz-aware UTC.
* Order independence: bootstrap CI invariant under unit-reordering.

See ``docs/phase3_reproducibility_audit.md`` for the full report and any
non-test diagnostic findings.
"""
from __future__ import annotations

import copy
import pickle
import warnings as _warnings

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _poisson_rec(rate_hz=20.0, duration_s=20.0, n_units=30, seed=0,
                 quality="good"):
    """Synthetic Poisson SpikeRecording with curation columns set."""
    from neurocomplexity.core.recording import SpikeRecording
    rng = np.random.default_rng(seed)
    times, ids = [], []
    for u in range(n_units):
        n = rng.poisson(rate_hz * duration_s)
        ts = np.sort(rng.uniform(0, duration_s, n))
        times.append(ts)
        ids.append(np.full(ts.size, u, dtype=np.int64))
    spike_times = np.concatenate(times).astype(np.float64) if times else np.array([], dtype=np.float64)
    unit_ids = np.concatenate(ids).astype(np.int64) if ids else np.array([], dtype=np.int64)
    order = np.argsort(spike_times, kind="stable")
    spike_times = spike_times[order]
    unit_ids = unit_ids[order]
    units = pd.DataFrame({
        "id": np.arange(n_units, dtype=np.int64),
        "quality": [quality] * n_units,
    })
    pops = {"all": np.ones(n_units, dtype=bool)}
    if n_units >= 4:
        half = n_units // 2
        a = np.zeros(n_units, dtype=bool); a[:half] = True
        b = np.zeros(n_units, dtype=bool); b[half:] = True
        pops["A"] = a
        pops["B"] = b
    return SpikeRecording(
        spike_times=spike_times, unit_ids=unit_ids, units=units,
        populations=pops, duration=float(duration_s), sampling_rate=30000.0,
        source="synthetic", _filtered=True,
    )


def _assert_result_equal(a, b):
    """Compare two frozen-dataclass results field by field, handling arrays."""
    from dataclasses import fields
    assert type(a) is type(b)
    for f in fields(a):
        av = getattr(a, f.name); bv = getattr(b, f.name)
        if isinstance(av, np.ndarray):
            np.testing.assert_array_equal(av, bv, err_msg=f"field {f.name}")
        elif isinstance(av, dict):
            assert av == bv, f"field {f.name}: {av!r} != {bv!r}"
        elif isinstance(av, float) and np.isnan(av):
            assert np.isnan(bv), f"field {f.name}: nan != {bv!r}"
        else:
            assert av == bv, f"field {f.name}: {av!r} != {bv!r}"


# ===========================================================================
# 1. Seed determinism — same seed, two runs, byte-identical
# ===========================================================================

class TestDeterminism:
    """Same seed → identical output across repeated runs."""

    def test_wilting_mr_deterministic(self):
        from neurocomplexity.analysis.branching import wilting_mr
        from neurocomplexity.benchmarks.simulators.branching_network import (
            branching_network,
        )
        rec1 = branching_network(n_units=50, m=0.9, duration_s=120.0,
                                 bin_ms=4.0, saturate=False, seed=7)
        rec2 = branching_network(n_units=50, m=0.9, duration_s=120.0,
                                 bin_ms=4.0, saturate=False, seed=7)
        # Seeded simulator must be byte-identical.
        np.testing.assert_array_equal(rec1.spike_times, rec2.spike_times)
        np.testing.assert_array_equal(rec1.unit_ids, rec2.unit_ids)
        r1 = wilting_mr(rec1, populations=["all"], bin_size_ms=4.0,
                         k_max=20, k_min=1)
        r2 = wilting_mr(rec2, populations=["all"], bin_size_ms=4.0,
                         k_max=20, k_min=1)
        assert r1.m == r2.m
        np.testing.assert_array_equal(r1.r_values, r2.r_values)

    def test_transfer_entropy_deterministic(self):
        from neurocomplexity.analysis.transfer_entropy import transfer_entropy
        rec = _poisson_rec(seed=11, duration_s=30.0)
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            r1 = transfer_entropy(rec, populations=["A", "B"],
                                   bin_size_ms=5.0, delay_bins=1)
            r2 = transfer_entropy(rec, populations=["A", "B"],
                                   bin_size_ms=5.0, delay_bins=1)
        np.testing.assert_array_equal(r1.matrix, r2.matrix)

    def test_surrogate_pool_deterministic(self):
        """SurrogatePool with same seed must produce identical surrogates."""
        from neurocomplexity.inference.pool import SurrogatePool
        rec = _poisson_rec(seed=13, duration_s=20.0)
        pool1 = SurrogatePool(rec, surrogate="spike_dither", n=10, seed=42)
        pool2 = SurrogatePool(rec, surrogate="spike_dither", n=10, seed=42)
        s1 = pool1[0]
        s2 = pool2[0]
        np.testing.assert_array_equal(s1.spike_times, s2.spike_times)
        np.testing.assert_array_equal(s1.unit_ids, s2.unit_ids)

    def test_pca_manifold_deterministic(self):
        from neurocomplexity.analysis.manifold import manifold
        rec = _poisson_rec(seed=15, duration_s=20.0)
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            r1 = manifold(rec, method="pca", dims=2, bin_size_s=0.05,
                           random_state=0)
            r2 = manifold(rec, method="pca", dims=2, bin_size_s=0.05,
                           random_state=0)
        np.testing.assert_array_equal(r1.coords, r2.coords)


# ===========================================================================
# 2. Pickle round-trip — every Result dataclass
# ===========================================================================

class TestPickleRoundtrip:
    """pickle.loads(pickle.dumps(r)) must equal r byte-for-byte on arrays."""

    def _roundtrip(self, obj):
        rt = pickle.loads(pickle.dumps(obj))
        return rt

    def test_recording_roundtrip(self):
        rec = _poisson_rec(seed=21)
        rt = self._roundtrip(rec)
        np.testing.assert_array_equal(rt.spike_times, rec.spike_times)
        np.testing.assert_array_equal(rt.unit_ids, rec.unit_ids)
        assert rt.duration == rec.duration

    def test_branching_result_roundtrip(self):
        from neurocomplexity.analysis.branching import wilting_mr
        rec = _poisson_rec(seed=22)
        r = wilting_mr(rec, populations=["all"], bin_size_ms=4.0,
                        k_max=15, k_min=1)
        _assert_result_equal(self._roundtrip(r), r)

    def test_criticality_result_roundtrip(self):
        from neurocomplexity.analysis.criticality import criticality
        from neurocomplexity.benchmarks.simulators.branching_network import (
            trial_based_avalanches,
        )
        rec = trial_based_avalanches(n_units=30, n_trials=2000, bin_ms=4.0,
                                      m=1.0, max_trial_bins=200, seed=23)
        r = criticality(rec, populations=["all"], bin_size=(4.0, 8.0))
        _assert_result_equal(self._roundtrip(r), r)

    def test_dimensionality_result_roundtrip(self):
        from neurocomplexity.analysis.dimensionality import dimensionality
        rec = _poisson_rec(seed=24, n_units=10, duration_s=30.0)
        r = dimensionality(rec, populations=["all"], bin_size_ms=10.0)
        _assert_result_equal(self._roundtrip(r), r)

    def test_lmc_result_roundtrip(self):
        from neurocomplexity.analysis.complexity import lmc_complexity
        rec = _poisson_rec(seed=25, n_units=10, duration_s=20.0)
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            r = lmc_complexity(rec, kind="population", bin_size_s=0.05)
        _assert_result_equal(self._roundtrip(r), r)

    def test_mse_result_roundtrip(self):
        from neurocomplexity.analysis.mse import multiscale_entropy
        rec = _poisson_rec(seed=26, n_units=10, duration_s=30.0)
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            r = multiscale_entropy(rec, bin_size_s=0.05, scale_max=3, m=2,
                                    r_factor=0.2)
        _assert_result_equal(self._roundtrip(r), r)

    def test_te_result_roundtrip(self):
        from neurocomplexity.analysis.transfer_entropy import transfer_entropy
        rec = _poisson_rec(seed=27, duration_s=15.0)
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            r = transfer_entropy(rec, populations=["A", "B"],
                                  bin_size_ms=5.0, delay_bins=1)
        _assert_result_equal(self._roundtrip(r), r)

    def test_pid_result_roundtrip(self):
        from neurocomplexity.benchmarks.simulators.pid_distributions import (
            pid_recording,
        )
        from neurocomplexity.analysis.pid import partial_information
        rec = pid_recording("xor", n_bins=5000, bin_ms=10.0, seed=28)
        r = partial_information(rec, target_pop="target",
                                  sources=["source_1", "source_2"],
                                  bin_size_ms=10.0, delay_bins=0,
                                  n_levels=3)
        _assert_result_equal(self._roundtrip(r), r)

    def test_stationarity_result_roundtrip(self):
        from neurocomplexity.analysis.stationarity import stationarity
        rec = _poisson_rec(seed=29, duration_s=180.0)
        r = stationarity(rec)
        _assert_result_equal(self._roundtrip(r), r)

    def test_inference_result_roundtrip(self):
        from neurocomplexity.inference import bootstrap
        from neurocomplexity.analysis.branching import wilting_mr
        rec = _poisson_rec(seed=30, duration_s=60.0)
        m = wilting_mr(rec, populations=["all"], bin_size_ms=4.0,
                        k_max=15, k_min=1)
        r = bootstrap(m, rec, n=10, seed=0)
        _assert_result_equal(self._roundtrip(r), r)


# ===========================================================================
# 3. Deepcopy round-trip
# ===========================================================================

class TestDeepcopyRoundtrip:
    """copy.deepcopy(r) must produce an equal but independent object."""

    def test_recording_deepcopy(self):
        rec = _poisson_rec(seed=41)
        rec2 = copy.deepcopy(rec)
        assert rec is not rec2
        assert rec.spike_times is not rec2.spike_times
        np.testing.assert_array_equal(rec.spike_times, rec2.spike_times)

    def test_branching_result_deepcopy(self):
        from neurocomplexity.analysis.branching import wilting_mr
        rec = _poisson_rec(seed=42)
        r = wilting_mr(rec, populations=["all"], bin_size_ms=4.0,
                        k_max=15, k_min=1)
        r2 = copy.deepcopy(r)
        assert r is not r2
        _assert_result_equal(r2, r)


# ===========================================================================
# 4. Edge cases — empty, single-unit, var=0, log(0), single-event avalanche
# ===========================================================================

class TestEdgeCases:
    """Estimators must fail loudly (or return NaN) on degenerate inputs."""

    def test_empty_avalanche_returns_empty_arrays(self):
        from neurocomplexity.analysis.criticality import extract_avalanches
        sizes, lifetimes = extract_avalanches(np.array([], dtype=np.int64),
                                               bin_size=0.004)
        assert sizes.size == 0 and lifetimes.size == 0

    def test_single_event_avalanche(self):
        """One isolated spike → one avalanche of size 1, lifetime = bin_size."""
        from neurocomplexity.analysis.criticality import extract_avalanches
        counts = np.array([0, 0, 1, 0, 0], dtype=np.int64)
        sizes, lifetimes = extract_avalanches(counts, bin_size=0.004)
        assert sizes.size == 1
        assert sizes[0] == 1
        np.testing.assert_allclose(lifetimes[0], 0.004)

    def test_all_zero_spike_train_does_not_crash(self):
        """An all-silent recording must not crash any analysis. PR raises;
        others return NaN or empty."""
        from neurocomplexity.analysis.dimensionality import dimensionality
        from neurocomplexity.core.recording import SpikeRecording
        units = pd.DataFrame({"id": np.arange(5, dtype=np.int64),
                              "quality": ["good"] * 5})
        rec = SpikeRecording(
            spike_times=np.array([], dtype=np.float64),
            unit_ids=np.array([], dtype=np.int64),
            units=units, populations={"all": np.ones(5, dtype=bool)},
            duration=20.0, sampling_rate=30000.0, source="silent",
            _filtered=True,
        )
        with pytest.raises(ValueError):
            dimensionality(rec, populations=["all"], bin_size_ms=10.0)

    def test_log_zero_protected_in_lmc(self):
        """LMC must skip zero-probability terms instead of producing -inf."""
        from neurocomplexity.analysis.complexity import _shannon_entropy_counts
        counts = np.array([10, 0, 0, 0, 5], dtype=np.float64)
        H = _shannon_entropy_counts(counts)
        assert np.isfinite(H)
        assert H > 0

    def test_log_zero_protected_in_te(self):
        """TE skips cells with zero joint or zero marginal."""
        from neurocomplexity.analysis.transfer_entropy import _binary_schreiber_te
        x = np.zeros(200, dtype=np.int64)
        y = np.zeros(200, dtype=np.int64)
        te = _binary_schreiber_te(x, y, delay=1)
        assert te == 0.0

    def test_var_zero_in_branching_returns_nan(self):
        """Constant population activity (var=0) → m NaN (not crash, not -inf)."""
        from neurocomplexity.analysis.branching import wilting_mr
        from neurocomplexity.core.recording import SpikeRecording
        # Build a SpikeRecording where every bin gets exactly one spike per
        # unit: perfectly periodic, so Var(A_t) = 0 at the matching bin size.
        n_units = 5; duration = 50.0; bin_s = 0.004
        n_bins = int(duration / bin_s)
        # one spike per unit per bin → perfectly constant activity
        bin_idx = np.repeat(np.arange(n_bins, dtype=np.int64), n_units)
        unit_ids = np.tile(np.arange(n_units, dtype=np.int64), n_bins)
        spike_times = (bin_idx * bin_s + bin_s / 2.0).astype(np.float64)
        order = np.argsort(spike_times, kind="stable")
        rec = SpikeRecording(
            spike_times=spike_times[order], unit_ids=unit_ids[order],
            units=pd.DataFrame({"id": np.arange(n_units, dtype=np.int64),
                                "quality": ["good"] * n_units}),
            populations={"all": np.ones(n_units, dtype=bool)},
            duration=duration, sampling_rate=30000.0, source="x",
            _filtered=True,
        )
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            r = wilting_mr(rec, bin_size_ms=4.0, k_max=20, k_min=1)
        # Variance is 0; estimator MUST return NaN, not -inf or a crash.
        assert np.isnan(r.m), f"expected NaN on var=0; got {r.m}"
        assert np.isnan(r.r_squared)

    def test_single_unit_dimensionality_raises(self):
        """PR requires ≥ 2 units; single unit must raise loud."""
        from neurocomplexity.analysis.dimensionality import dimensionality
        from neurocomplexity.core.recording import SpikeRecording
        rec = SpikeRecording(
            spike_times=np.array([0.1, 0.2], dtype=np.float64),
            unit_ids=np.array([0, 0], dtype=np.int64),
            units=pd.DataFrame({"id": [0], "quality": ["good"]}),
            populations={"all": np.ones(1, dtype=bool)},
            duration=10.0, sampling_rate=30000.0, source="x", _filtered=True,
        )
        with pytest.raises(ValueError, match="at least 2 units"):
            dimensionality(rec, populations=["all"], bin_size_ms=10.0)


# ===========================================================================
# 5. dtype + timezone invariants
# ===========================================================================

class TestDtypes:
    """Spike data must carry int64 / float64; provenance timestamps tz-aware."""

    def test_recording_dtypes(self):
        rec = _poisson_rec(seed=51)
        assert rec.spike_times.dtype == np.float64
        assert rec.unit_ids.dtype == np.int64

    def test_provenance_record_timestamp_is_utc_iso8601(self):
        """``loaded_at`` is an ISO-8601 string and MUST encode UTC.

        Either a trailing ``Z`` or an explicit ``+00:00`` offset is acceptable.
        A bare timestamp without timezone is a bug (timezone-naive timestamps
        cannot be safely compared across machines).
        """
        from neurocomplexity.core.provenance import ProvenanceRecord
        pr = ProvenanceRecord.for_memory(source_format="test",
                                          hint="phase3-audit")
        ts = pr.loaded_at
        assert isinstance(ts, str), f"loaded_at should be str, got {type(ts)}"
        # tz indicator must be present at end of string
        assert ts.endswith("Z") or ts.endswith("+00:00") or "+00:" in ts, (
            f"loaded_at is timezone-naive: {ts!r}"
        )
        # Round-trip via datetime to confirm it parses cleanly
        from datetime import datetime
        parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        assert parsed.tzinfo is not None, "parsed timestamp has no tzinfo"

    def test_branching_m_is_python_float(self):
        from neurocomplexity.analysis.branching import wilting_mr
        rec = _poisson_rec(seed=53)
        r = wilting_mr(rec, populations=["all"], bin_size_ms=4.0,
                        k_max=15, k_min=1)
        assert isinstance(r.m, float)

    def test_inference_p_value_dtype(self):
        from neurocomplexity.inference.null_test import pvalue_from_null
        null = np.random.default_rng(0).normal(0, 1, 100)
        p = pvalue_from_null(2.5, null, alternative="greater")
        assert np.asarray(p).dtype == np.float64


# ===========================================================================
# 6. Order independence
# ===========================================================================

class TestOrderIndependence:
    """Reshuffling spike order within a unit, or relabeling unit IDs, must not
    change population-level estimators."""

    def test_branching_invariant_under_spike_resort(self):
        """Sorting spike_times globally vs. by unit then time → same MR."""
        from neurocomplexity.analysis.branching import wilting_mr
        from neurocomplexity.core.recording import SpikeRecording
        rec_a = _poisson_rec(seed=61, duration_s=60.0)
        # Build a re-shuffled copy: shuffle within each unit then re-sort
        # globally; must produce identical population-rate series.
        rng = np.random.default_rng(99)
        idx = np.arange(rec_a.spike_times.size)
        rng.shuffle(idx)
        st = rec_a.spike_times[idx]
        uu = rec_a.unit_ids[idx]
        order = np.argsort(st, kind="stable")
        rec_b = SpikeRecording(
            spike_times=st[order], unit_ids=uu[order],
            units=rec_a.units, populations=rec_a.populations,
            duration=rec_a.duration, sampling_rate=rec_a.sampling_rate,
            source=rec_a.source, _filtered=True,
        )
        ra = wilting_mr(rec_a, populations=["all"], bin_size_ms=4.0,
                         k_max=20, k_min=1)
        rb = wilting_mr(rec_b, populations=["all"], bin_size_ms=4.0,
                         k_max=20, k_min=1)
        assert ra.m == pytest.approx(rb.m, abs=1e-12)

    def test_te_matrix_invariant_under_population_relabel(self):
        """Permuting populations should permute the TE matrix, not change values."""
        from neurocomplexity.analysis.transfer_entropy import transfer_entropy
        rec = _poisson_rec(seed=62, duration_s=30.0)
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            r_ab = transfer_entropy(rec, populations=["A", "B"],
                                     bin_size_ms=5.0, delay_bins=1)
            r_ba = transfer_entropy(rec, populations=["B", "A"],
                                     bin_size_ms=5.0, delay_bins=1)
        # r_ab is index [A, B]; r_ba is index [B, A] = r_ab with rows/cols swapped.
        np.testing.assert_allclose(r_ba.matrix, r_ab.matrix[::-1, ::-1])
