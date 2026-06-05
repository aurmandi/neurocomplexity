"""Phase 1: invariant / property tests for every Result dataclass.

These tests lock in mathematical invariants documented in
``docs/publication_plan.md``. They should never need to change. If one fails,
it is a bug in the analysis code (Phase 2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings, HealthCheck, strategies as st

from neurocomplexity.core.recording import SpikeRecording


# ---------------------------------------------------------------------------
# Fixtures: minimal synthetic recordings
# ---------------------------------------------------------------------------

def _poisson_rec(rate_hz: float = 20.0, duration_s: float = 20.0,
                 n_units: int = 30, seed: int = 0,
                 populations: dict | None = None) -> SpikeRecording:
    rng = np.random.default_rng(seed)
    spike_times = []
    unit_ids = []
    for u in range(n_units):
        n = rng.poisson(rate_hz * duration_s)
        ts = np.sort(rng.uniform(0, duration_s, n))
        spike_times.append(ts)
        unit_ids.append(np.full(ts.size, u, dtype=np.int64))
    spike_times = np.concatenate(spike_times) if spike_times else np.array([])
    unit_ids = np.concatenate(unit_ids) if unit_ids else np.array([], dtype=np.int64)
    order = np.argsort(spike_times, kind="stable")
    spike_times = spike_times.astype(np.float64)[order]
    unit_ids = unit_ids[order]
    units = pd.DataFrame({"id": np.arange(n_units, dtype=np.int64),
                          "quality": ["good"] * n_units})
    pops = populations or {"all": np.ones(n_units, dtype=bool)}
    return SpikeRecording(
        spike_times=spike_times, unit_ids=unit_ids, units=units,
        populations=pops, duration=float(duration_s), sampling_rate=30000.0,
        source="synthetic", _filtered=True,
    )


def _two_pop_rec(rate_hz: float = 20.0, duration_s: float = 20.0,
                 n_units: int = 40, seed: int = 0) -> SpikeRecording:
    mask_a = np.zeros(n_units, dtype=bool); mask_a[: n_units // 2] = True
    mask_b = ~mask_a
    return _poisson_rec(rate_hz=rate_hz, duration_s=duration_s, n_units=n_units,
                        seed=seed, populations={"a": mask_a, "b": mask_b})


def _three_pop_rec(rate_hz: float = 20.0, duration_s: float = 20.0,
                   n_units: int = 60, seed: int = 0) -> SpikeRecording:
    pops = {}
    third = n_units // 3
    for i, name in enumerate(("a", "b", "c")):
        m = np.zeros(n_units, dtype=bool)
        m[i * third:(i + 1) * third] = True
        pops[name] = m
    return _poisson_rec(rate_hz=rate_hz, duration_s=duration_s, n_units=n_units,
                        seed=seed, populations=pops)


# ===========================================================================
# BranchingResult invariants
# ===========================================================================

class TestBranchingResult:
    """`m >= 0`, finite; agrees with single-step Pearson at k_max=1 (tol)."""

    def test_m_is_finite_and_nonneg(self):
        from neurocomplexity.analysis.branching import wilting_mr
        rec = _poisson_rec(rate_hz=30.0, duration_s=20.0, seed=1)
        r = wilting_mr(rec, bin_size_ms=4.0, k_max=20, k_min=1)
        assert np.isfinite(r.m), f"m not finite: {r.m}"
        assert r.m >= 0.0, f"m must be >= 0, got {r.m}"

    def test_m_matches_pearson_at_k1(self):
        """At k_min=k_max=1 the slope is just log(r_1); m = r_1.
        Note: the implementation requires k_max > k_min (strict), so we
        cannot fit at k_max=k_min=1. Instead verify r_values[0] == Pearson
        single-step autocorrelation directly.
        """
        from neurocomplexity.analysis.branching import wilting_mr
        from neurocomplexity.analysis._binning import bin_all_active
        rec = _poisson_rec(rate_hz=50.0, duration_s=20.0, seed=2)
        bs_ms = 4.0
        r = wilting_mr(rec, bin_size_ms=bs_ms, k_max=10, k_min=1)
        A = bin_all_active(rec, list(rec.populations.keys()),
                           bs_ms / 1000.0).astype(np.float64)
        # Pearson single-step autocorrelation
        a0 = A[:-1] - A[:-1].mean()
        a1 = A[1:] - A[1:].mean()
        pearson = float(np.sum(a0 * a1) /
                        np.sqrt(np.sum(a0 ** 2) * np.sum(a1 ** 2)))
        # r_values uses var/cov definition; should equal Pearson up to a
        # 1-sample shift in the normalisation. Allow generous tolerance.
        r1 = float(r.r_values[0])
        assert r1 == pytest.approx(pearson, abs=0.05), (
            f"r_values[0]={r1} disagrees with Pearson single-step={pearson}")


# ===========================================================================
# CriticalityResult invariants
# ===========================================================================

class TestCriticalityResult:
    """alpha_s, alpha_t > 1 for valid fits; Sethna gamma identity exact.

    Uses ``trial_based_avalanches(m=1.0)`` (Galton-Watson critical branching)
    as the fixture — Poisson does not produce heavy-tailed avalanche
    distributions, which is why the earlier Phase-1 version skipped.
    """

    @staticmethod
    def _critical_rec(seed: int):
        from neurocomplexity.benchmarks.simulators.branching_network import (
            trial_based_avalanches,
        )
        return trial_based_avalanches(
            n_units=50, n_trials=8000, bin_ms=4.0,
            m=1.0, max_trial_bins=500, seed=seed,
        )

    def test_sethna_identity_exact(self):
        """gamma_predicted == (alpha_t - 1) / (alpha_s - 1) exactly."""
        from neurocomplexity.analysis.criticality import criticality
        rec = self._critical_rec(seed=3)
        r = criticality(rec, bin_size=(4.0, 8.0))
        assert not np.isnan(r.alpha_s), "fixture produced degenerate avalanches"
        assert not np.isnan(r.alpha_t)
        expected = (r.alpha_t - 1.0) / (r.alpha_s - 1.0)
        assert r.gamma_predicted == pytest.approx(expected, rel=1e-12, abs=1e-12)

    def test_alpha_s_alpha_t_greater_than_one_when_valid(self):
        from neurocomplexity.analysis.criticality import criticality
        rec = self._critical_rec(seed=4)
        r = criticality(rec, bin_size=(4.0, 8.0))
        assert not np.isnan(r.alpha_s)
        assert not np.isnan(r.alpha_t)
        assert r.alpha_s > 1.0, f"alpha_s must be > 1 for power-law tail, got {r.alpha_s}"
        assert r.alpha_t > 1.0, f"alpha_t must be > 1 for power-law tail, got {r.alpha_t}"

    def test_wilting_mr_recovers_known_m(self):
        """Wilting multi-step regression unbiased near criticality.

        Galton-Watson branching at m_true=0.95 (sub-critical, stationary)
        should recover m_hat within 0.05.
        """
        from neurocomplexity.analysis.branching import wilting_mr
        from neurocomplexity.benchmarks.simulators.branching_network import (
            branching_network,
        )
        m_true = 0.95
        rec = branching_network(
            n_units=100, m=m_true, duration_s=600.0, bin_ms=4.0,
            external_rate_hz=0.5, saturate=False, seed=5,
        )
        r = wilting_mr(rec, populations=["all"], bin_size_ms=4.0,
                        k_max=30, k_min=1)
        assert abs(r.m - m_true) < 0.05, (
            f"Wilting MR off by {abs(r.m - m_true):.3f}: "
            f"expected {m_true}, got {r.m}"
        )
        assert r.r_squared > 0.95, f"poor MR fit, R^2 = {r.r_squared}"


# ===========================================================================
# DimensionalityResult invariants
# ===========================================================================

class TestDimensionalityResult:
    """1 <= PR <= n_units; PR(identity covariance) ~= n_units."""

    def test_pr_bounds(self):
        from neurocomplexity.analysis.dimensionality import dimensionality
        rec = _poisson_rec(rate_hz=20.0, duration_s=20.0, n_units=30, seed=5)
        r = dimensionality(rec, bin_size_ms=10.0)
        assert 1.0 <= r.participation_ratio <= r.n_units + 1e-9, (
            f"PR={r.participation_ratio} outside [1, {r.n_units}]")

    def test_pr_identity_covariance_equals_n(self):
        """For independent Poisson units (≈ identity correlation matrix), PR ≈ N.

        Test the internal _participation_ratio helper directly with a perfect
        identity covariance — this is the canonical ground-truth invariant.
        """
        from neurocomplexity.analysis.dimensionality import _participation_ratio
        for N in (5, 20, 100):
            eig = np.ones(N)
            pr = _participation_ratio(eig)
            assert pr == pytest.approx(float(N), rel=1e-12)

    def test_pr_single_dominant_mode_equals_one(self):
        from neurocomplexity.analysis.dimensionality import _participation_ratio
        eig = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        pr = _participation_ratio(eig)
        assert pr == pytest.approx(1.0)


# ===========================================================================
# LMCResult invariants
# ===========================================================================

class TestLMCResult:
    """C >= 0; C ~ 0 for uniform; C ~ 0 for delta."""

    def test_C_nonnegative(self):
        from neurocomplexity.analysis.complexity import lmc_complexity
        rec = _poisson_rec(rate_hz=20.0, duration_s=10.0, seed=6)
        r = lmc_complexity(rec, kind="population", bin_size_s=0.05)
        assert np.all(r.C_per_pop >= 0.0)

    def test_C_uniform_distribution_is_zero(self):
        from neurocomplexity.analysis.complexity import (
            _shannon_entropy_counts, _lmc_disequilibrium,
        )
        # Uniform over 100 states: D=0 (since p_i=1/N for all i), so C=H*D=0
        counts = np.full(100, 50)
        H = _shannon_entropy_counts(counts)
        D = _lmc_disequilibrium(counts)
        C = H * D
        assert H == pytest.approx(1.0)
        assert D == pytest.approx(0.0, abs=1e-12)
        assert C == pytest.approx(0.0, abs=1e-12)

    def test_C_delta_distribution_is_zero(self):
        from neurocomplexity.analysis.complexity import (
            _shannon_entropy_counts, _lmc_disequilibrium,
        )
        # Delta distribution: H=0 -> C=0 regardless of D
        counts = np.array([1000, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        H = _shannon_entropy_counts(counts)
        D = _lmc_disequilibrium(counts)
        C = H * D
        assert H == pytest.approx(0.0)
        assert C == pytest.approx(0.0, abs=1e-12)


# ===========================================================================
# MSEResult invariants
# ===========================================================================

class TestMSEResult:
    """sampen[:, 0] equals sample entropy at scale 1 (i.e. on the raw series)."""

    def test_sampen_scale1_equals_direct_sample_entropy(self):
        from neurocomplexity.analysis.mse import (
            multiscale_entropy, _sample_entropy,
        )
        from neurocomplexity.analysis._binning import bin_spikes
        rec = _poisson_rec(rate_hz=20.0, duration_s=15.0, n_units=20, seed=7)
        r = multiscale_entropy(rec, bin_size_s=0.05, scale_max=3, m=2,
                                r_factor=0.2)
        # Re-compute sample entropy directly at scale 1
        counts = bin_spikes(rec, list(rec.populations.keys()),
                            0.05).astype(np.float64)
        for p in range(counts.shape[1]):
            series = counts[:, p]
            r_tol = 0.2 * float(series.std(ddof=0))
            direct = _sample_entropy(series, m=2, r=r_tol)
            mse_val = r.sampen[p, 0]
            if np.isnan(direct) and np.isnan(mse_val):
                continue
            assert mse_val == pytest.approx(direct, rel=1e-10, abs=1e-10), (
                f"scale=1 sampen for pop {p}: MSE={mse_val} direct={direct}")
        # scale 1 corresponds to scales[0] == 1
        assert int(r.scales[0]) == 1


# ===========================================================================
# PIDResult invariants
# ===========================================================================

class TestPIDResult:
    """All four atoms >= 0; redundancy + u1 + u2 + synergy ~= I(target; sources)."""

    def test_atoms_nonnegative(self):
        from neurocomplexity.analysis.pid import partial_information
        rec = _three_pop_rec(rate_hz=30.0, duration_s=20.0, seed=8)
        r = partial_information(rec, target_pop="a", sources=("b", "c"),
                                 bin_size_ms=5.0, delay_bins=1, n_levels=3)
        assert r.redundancy >= 0.0, f"redundancy {r.redundancy} < 0"
        assert r.unique_1 >= 0.0, f"unique_1 {r.unique_1} < 0"
        assert r.unique_2 >= 0.0, f"unique_2 {r.unique_2} < 0"
        assert r.synergy >= 0.0, f"synergy {r.synergy} < 0"

    def test_atoms_sum_equals_total_mi(self):
        from neurocomplexity.analysis.pid import partial_information
        rec = _three_pop_rec(rate_hz=30.0, duration_s=20.0, seed=9)
        r = partial_information(rec, target_pop="a", sources=("b", "c"),
                                 bin_size_ms=5.0, delay_bins=1, n_levels=3)
        total = r.redundancy + r.unique_1 + r.unique_2 + r.synergy
        assert total == pytest.approx(r.total_mi, abs=1e-9, rel=1e-9), (
            f"atom sum {total} != total_mi {r.total_mi}")


# ===========================================================================
# TransferEntropyResult invariants
# ===========================================================================

class TestTransferEntropyResult:
    """diag == 0; non-negative; row=source convention."""

    def test_diag_is_zero(self):
        from neurocomplexity.analysis.transfer_entropy import transfer_entropy
        rec = _three_pop_rec(rate_hz=20.0, duration_s=20.0, seed=10)
        r = transfer_entropy(rec, bin_size_ms=5.0, delay_bins=1)
        assert np.allclose(np.diag(r.matrix), 0.0), (
            f"diagonal not zero: {np.diag(r.matrix)}")

    def test_matrix_nonnegative(self):
        from neurocomplexity.analysis.transfer_entropy import transfer_entropy
        rec = _three_pop_rec(rate_hz=20.0, duration_s=20.0, seed=11)
        r = transfer_entropy(rec, bin_size_ms=5.0, delay_bins=1)
        assert np.all(r.matrix >= 0.0), (
            f"negative TE found: min={r.matrix.min()}")

    def test_row_equals_source_convention(self):
        """matrix[i, j] = TE from i (source) to j (target).

        Build a recording where population 'a' DRIVES population 'b' with a
        1-bin lag, and 'c' is independent. Then matrix[a, b] >> matrix[b, a]
        and matrix[a, b] >> matrix[c, b].
        """
        from neurocomplexity.analysis.transfer_entropy import transfer_entropy
        rng = np.random.default_rng(42)
        duration = 30.0
        bs = 0.005  # 5 ms
        T = int(duration / bs)
        # Driver process: random binary at each bin
        drive = rng.integers(0, 2, size=T)
        # 'b' follows 'a' with 1-bin delay
        follower = np.zeros(T, dtype=int)
        follower[1:] = drive[:-1]
        # convert binary bin sequences to spike times
        def _binary_to_spikes(binary, unit_ids_start, n_units):
            sts, uids = [], []
            for bi, val in enumerate(binary):
                if val:
                    # All units fire in this bin (single representative spike)
                    t = (bi + 0.5) * bs
                    for u in range(n_units):
                        sts.append(t)
                        uids.append(unit_ids_start + u)
            return np.array(sts, dtype=np.float64), np.array(uids, dtype=np.int64)
        sts_a, uids_a = _binary_to_spikes(drive, 0, 5)
        sts_b, uids_b = _binary_to_spikes(follower, 5, 5)
        # 'c' independent
        indep = rng.integers(0, 2, size=T)
        sts_c, uids_c = _binary_to_spikes(indep, 10, 5)
        sts = np.concatenate([sts_a, sts_b, sts_c])
        uids = np.concatenate([uids_a, uids_b, uids_c])
        order = np.argsort(sts, kind="stable")
        sts = sts[order]; uids = uids[order]
        units = pd.DataFrame({"id": np.arange(15, dtype=np.int64),
                              "quality": ["good"] * 15})
        pops = {
            "a": np.array([True]*5 + [False]*10),
            "b": np.array([False]*5 + [True]*5 + [False]*5),
            "c": np.array([False]*10 + [True]*5),
        }
        rec = SpikeRecording(
            spike_times=sts, unit_ids=uids, units=units, populations=pops,
            duration=duration, sampling_rate=30000.0, source="synthetic",
            _filtered=True,
        )
        r = transfer_entropy(rec, bin_size_ms=5.0, delay_bins=1)
        # populations is ("a","b","c") in that order
        pops_order = list(r.populations)
        ia = pops_order.index("a"); ib = pops_order.index("b")
        ic = pops_order.index("c")
        te_a_to_b = r.matrix[ia, ib]
        te_b_to_a = r.matrix[ib, ia]
        te_c_to_b = r.matrix[ic, ib]
        # row=source means a (driver) -> b (follower) is large
        assert te_a_to_b > te_b_to_a, (
            f"row=source violated: TE[a->b]={te_a_to_b} not > TE[b->a]={te_b_to_a}")
        assert te_a_to_b > te_c_to_b, (
            f"row=source violated: TE[a->b]={te_a_to_b} not > TE[c->b]={te_c_to_b}")


# ===========================================================================
# InferenceResult invariants
# ===========================================================================

class TestInferenceResult:
    """p_value ∈ (0, 1] with Phipson-Smyth floor; FDR >= raw p elementwise."""

    def test_phipson_smyth_floor_scalar(self):
        from neurocomplexity.inference.null_test import pvalue_from_null
        # observed exceeds every null draw → ge=0 → p = 1/(1+n) > 0
        rng = np.random.default_rng(0)
        null = rng.normal(0, 1, size=100)
        observed = 100.0  # huge
        p = pvalue_from_null(observed, null, alternative="greater")
        assert p > 0.0, f"Phipson-Smyth floor violated: p={p}"
        assert p == pytest.approx(1.0 / 101.0)
        assert p <= 1.0

    def test_pvalue_range_array(self):
        from neurocomplexity.inference.null_test import pvalue_from_null
        rng = np.random.default_rng(1)
        null = rng.normal(0, 1, size=(200, 5))
        observed = np.array([-3.0, 0.0, 1.0, 2.0, 5.0])
        p = pvalue_from_null(observed, null, alternative="greater")
        assert np.all(p > 0.0), f"some p=0: {p}"
        assert np.all(p <= 1.0), f"some p>1: {p}"

    def test_fdr_ge_raw_pvalue_elementwise(self):
        """Benjamini-Hochberg adjusted p must be >= raw p elementwise."""
        from neurocomplexity.inference.null_test import fdr_bh
        rng = np.random.default_rng(2)
        # Mix of small and large p-values
        p_raw = np.concatenate([
            rng.uniform(0.001, 0.01, size=10),
            rng.uniform(0.1, 0.9, size=40),
        ])
        p_fdr = fdr_bh(p_raw)
        assert p_fdr.shape == p_raw.shape
        assert np.all(p_fdr >= p_raw - 1e-12), (
            f"FDR < raw found: max violation = {(p_raw - p_fdr).max()}")
        assert np.all(p_fdr <= 1.0)

    @given(
        n_null=st.integers(min_value=10, max_value=500),
        observed=st.floats(min_value=-10.0, max_value=10.0,
                           allow_nan=False, allow_infinity=False),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(deadline=None, max_examples=30,
              suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_pvalue_always_positive_property(self, n_null, observed, seed):
        from neurocomplexity.inference.null_test import pvalue_from_null
        rng = np.random.default_rng(seed)
        null = rng.normal(0, 1, size=n_null)
        for alt in ("greater", "less", "two-sided"):
            p = pvalue_from_null(observed, null, alternative=alt)
            assert p > 0.0, f"alt={alt}: p={p} <= 0"
            assert p <= 1.0 + 1e-12, f"alt={alt}: p={p} > 1"


# ===========================================================================
# Hypothesis property tests on _participation_ratio
# ===========================================================================

class TestPRProperty:
    @given(
        n=st.integers(min_value=2, max_value=50),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(deadline=None, max_examples=30)
    def test_pr_within_bounds_for_random_eigvals(self, n, seed):
        from neurocomplexity.analysis.dimensionality import _participation_ratio
        rng = np.random.default_rng(seed)
        eig = np.abs(rng.normal(size=n)) + 1e-9  # strictly positive
        pr = _participation_ratio(eig)
        assert 1.0 - 1e-9 <= pr <= n + 1e-9, (
            f"PR={pr} out of [1, {n}] for eig={eig}")
