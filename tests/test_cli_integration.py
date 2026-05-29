"""End-to-end CLI integration tests (Phase-4 Tier 5.20).

Exercises the README quickstart command surface through ``cli.main`` on a
synthetic NWB: ``info``/``analyze`` ``--json`` machine-readable output (JSON
on stdout, human progress on stderr) and the ``analyze`` -> ``figure``
re-render round-trip from a cached ``results.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pynwb = pytest.importorskip("pynwb")

from neurocomplexity import io as nc_io
from neurocomplexity.cli import main
from neurocomplexity.core.provenance import ProvenanceRecord
from neurocomplexity.core.recording import SpikeRecording


def _synth_rec(seed: int = 0, duration: float = 30.0, n_units: int = 20):
    rng = np.random.default_rng(seed)
    unit_ids = np.arange(n_units, dtype=np.int64)
    rates = rng.uniform(5.0, 30.0, size=n_units)
    spike_times_list = []
    spike_owners_list = []
    for uid, rate in zip(unit_ids, rates):
        n = rng.poisson(rate * duration)
        t = np.sort(rng.uniform(0.0, duration, size=n))
        spike_times_list.append(t)
        spike_owners_list.append(np.full(n, uid, dtype=np.int64))
    spike_times = np.concatenate(spike_times_list)
    owners = np.concatenate(spike_owners_list)
    order = np.argsort(spike_times, kind="stable")
    spike_times = spike_times[order].astype(np.float64)
    owners = owners[order]
    half = n_units // 2
    units = pd.DataFrame({
        "id": unit_ids,
        "quality": pd.Categorical(["good"] * n_units,
                                  categories=["noise", "mua", "good"]),
        "brain_area": ["VISp"] * half + ["LGd"] * (n_units - half),
        "firing_rate": rates,
    })
    populations = {
        "VISp": np.array([i < half for i in range(n_units)]),
        "LGd":  np.array([i >= half for i in range(n_units)]),
    }
    return SpikeRecording(
        spike_times=spike_times,
        unit_ids=owners,
        units=units,
        populations=populations,
        duration=duration,
        sampling_rate=30000.0,
        source=ProvenanceRecord.for_memory("synth", "integration"),
        intervals={},
    )


@pytest.fixture
def synth_nwb(tmp_path: Path) -> Path:
    rec = _synth_rec()
    p = tmp_path / "synth.nwb"
    nc_io.to_nwb(rec, p, session_description="integ", identifier="integ",
                 session_start_time=None)
    return p


def test_cli_info_json(synth_nwb, capsys):
    """``info --json`` prints a single parseable JSON object on stdout."""
    rc = main(["info", str(synth_nwb), "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["n_units"] == 20
    assert set(payload["populations"]) == {"VISp", "LGd"}
    assert payload["duration_seconds"] == pytest.approx(30.0, abs=1.0)


def test_cli_analyze_json_stdout_stderr_split(synth_nwb, tmp_path, capsys):
    """``analyze --json`` puts JSON on stdout and progress on stderr.

    stdout must be pure JSON (parseable as one object); the human progress
    lines (``[criticality] ...`` etc.) must land on stderr instead.
    """
    outdir = tmp_path / "out"
    rc = main(["analyze", str(synth_nwb), "-o", str(outdir),
               "--no-figures", "--json"])
    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)  # stdout is pure JSON
    assert payload["neurocomplexity_version"]
    assert "criticality" in payload["results"]
    # Progress markers are on stderr, not stdout.
    assert "[criticality]" in captured.err
    assert "[criticality]" not in captured.out


def test_cli_analyze_then_figure_roundtrip(synth_nwb, tmp_path):
    """``analyze`` writes results.json; ``figure`` re-renders from it.

    Covers the documented two-step workflow without recomputing statistics.
    """
    outdir = tmp_path / "out"
    rc = main(["analyze", str(synth_nwb), "-o", str(outdir), "--no-figures"])
    assert rc == 0
    results_json = outdir / "results.json"
    assert results_json.exists()

    figdir = tmp_path / "figs"
    rc2 = main(["figure", str(results_json), "-o", str(figdir),
                "--formats", "svg"])
    assert rc2 == 0
    assert len(list(figdir.glob("*.svg"))) >= 1
