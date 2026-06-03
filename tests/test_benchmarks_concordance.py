"""Tests for the cross-tool concordance benchmark (P0-1).

Verifies:
  * Registry populated with at least two cases.
  * Each case returns a dict with the documented schema.
  * Cases skip cleanly when reference tools are unavailable.
  * The PID-vs-dit case recovers the analytic XOR identities.
"""
from __future__ import annotations

import numpy as np

from neurocomplexity.benchmarks.cases.concordance import (
    CONCORDANCE_CASES,
    concordance_branching_vs_mrestimator,
    concordance_pid_vs_dit,
)


REQUIRED_KEYS = {"name", "skipped", "tolerance", "pass"}


def test_registry_has_two_cases():
    assert len(CONCORDANCE_CASES) >= 2
    names = {fn.__name__ for fn in CONCORDANCE_CASES}
    assert "concordance_branching_vs_mrestimator" in names
    assert "concordance_pid_vs_dit" in names


def test_branching_concordance_schema():
    """Result dict has the required keys regardless of skip status."""
    r = concordance_branching_vs_mrestimator()
    assert REQUIRED_KEYS.issubset(r.keys())
    assert isinstance(r["skipped"], bool)
    assert isinstance(r["tolerance"], float)
    if r["skipped"]:
        assert "reason" in r


def test_pid_concordance_schema():
    r = concordance_pid_vs_dit()
    assert REQUIRED_KEYS.issubset(r.keys())
    assert "nc_redundancy" in r
    assert "nc_synergy" in r
    assert "analytic_redundancy" in r
    assert "analytic_synergy" in r
    assert "dit_status" in r


def test_pid_xor_recovers_analytic_identity():
    """Williams-Beer I_min on XOR -> R ~ 0, S ~ ln 2 nats."""
    r = concordance_pid_vs_dit()
    # XOR analytic: R=0 nats, U1=U2=0 nats, S=ln(2) nats
    assert r["nc_redundancy"] < 0.02, (
        f"I_min redundancy on XOR = {r['nc_redundancy']:.4g} should be ~0"
    )
    expected_synergy = float(np.log(2.0))
    assert abs(r["nc_synergy"] - expected_synergy) < 0.05, (
        f"I_min synergy on XOR = {r['nc_synergy']:.4g} "
        f"should be ~{expected_synergy:.4g}"
    )


def test_pid_passes_analytic_check_even_when_dit_unavailable():
    """The analytic-recovery half of the case is independent of dit."""
    r = concordance_pid_vs_dit()
    # diff_red_analytic and diff_syn_analytic are always populated
    assert r["diff_red_analytic"] < r["tolerance"]
    assert r["diff_syn_analytic"] < r["tolerance"]


def test_branching_concordance_runs_or_skips_cleanly():
    """If mrestimator is installed, run; otherwise skip with a reason."""
    r = concordance_branching_vs_mrestimator()
    if r["skipped"]:
        assert r["reason"]
    else:
        assert "nc_m" in r
        assert "mre_m" in r
        assert "diff" in r
        assert np.isfinite(r["nc_m"])
