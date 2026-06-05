"""Regression tests for `pvalue_from_null` two-sided + less alternatives.

Pre-fix the two-sided test used the mean of the null as a pivot
(|null - mean| >= |obs - mean|). That under-powers for skewed null
distributions. Post-fix it uses the conventional double-tail
`2 * min(p_greater, p_less)` clipped at 1.
"""
from __future__ import annotations

import numpy as np
import pytest

from neurocomplexity.inference.null_test import pvalue_from_null


def test_greater_alternative_unchanged():
    null = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    p = pvalue_from_null(0.35, null, alternative="greater")
    # 1 of 5 null >= 0.35 → (1+1)/(1+5) = 2/6
    assert p == pytest.approx(2 / 6)


def test_less_alternative_new():
    null = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    p = pvalue_from_null(0.05, null, alternative="less")
    # 1 of 5 null <= 0.05 → (1+1)/6
    assert p == pytest.approx(2 / 6)


def test_two_sided_symmetric_null_matches_old_behaviour():
    """For symmetric nulls the new formula gives intuitive answers."""
    rng = np.random.default_rng(0)
    null = rng.normal(0, 1, size=1000)
    # Observed 1.96 → ~2.5% in each tail → two-sided p ~5%
    p = pvalue_from_null(1.96, null, alternative="two-sided")
    assert 0.02 < p < 0.10


def test_two_sided_skewed_null_robust():
    """Heavily right-skewed null: mean-centred test under-powers; double-tail
    correctly flags an observation far in the LEFT tail."""
    null = np.array([0.0]*90 + list(np.linspace(1, 100, 10)))  # 90 zeros + heavy right
    obs = -1.0  # clearly to the left of every null sample
    p = pvalue_from_null(obs, null, alternative="two-sided")
    # 1 of 100 null <= -1 (none); Phipson floor (0+1)/(100+1) per tail
    # → 2 * 1/101 ≈ 0.0198
    assert p == pytest.approx(2.0 / 101)


def test_two_sided_clipped_at_one():
    """When obs is at the median of a symmetric null, both tails give ~0.5 →
    2 * 0.5 = 1.0 (clipped, not 1.5)."""
    null = np.array([-1.0, 0.0, 1.0])
    p = pvalue_from_null(0.0, null, alternative="two-sided")
    assert p <= 1.0


def test_invalid_alternative_raises():
    with pytest.raises(ValueError, match="alternative"):
        pvalue_from_null(0.5, np.array([0.1, 0.2, 0.3]), alternative="bogus")


def test_vector_observed_two_sided_shape():
    rng = np.random.default_rng(1)
    null = rng.normal(0, 1, size=(200, 3, 3))
    obs = np.full((3, 3), 1.5)
    p = pvalue_from_null(obs, null, alternative="two-sided")
    assert p.shape == (3, 3)
    assert np.all(p >= 0.0) and np.all(p <= 1.0)


def test_vector_observed_less_shape():
    rng = np.random.default_rng(2)
    null = rng.normal(0, 1, size=(100, 4))
    obs = np.array([-2.0, 0.0, 2.0, 10.0])
    p = pvalue_from_null(obs, null, alternative="less")
    assert p.shape == (4,)
    assert p[0] < p[1] < p[2] < p[3]
