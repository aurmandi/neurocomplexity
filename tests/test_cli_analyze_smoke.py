"""Smoke tests for the ``neurocomplexity`` CLI.

Confirms the README headline `analyze` invocation runs end-to-end on a
synthetic NWB and writes the documented figure formats. Guards Tier 1.1 of
the Phase 4 revision punch-list: prior to fix, ``--formats`` defaulted to
``pdf svg png`` while ``save_publication`` only accepts ``svg tiff jpg``,
so the headline command crashed.
"""
from __future__ import annotations

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
        source=ProvenanceRecord.for_memory("synth", "smoke"),
        intervals={},
    )


@pytest.fixture
def synth_nwb(tmp_path: Path) -> Path:
    rec = _synth_rec()
    p = tmp_path / "synth.nwb"
    nc_io.to_nwb(rec, p, session_description="smoke", identifier="smoke",
                 session_start_time=None)
    return p


def test_cli_analyze_default_formats_smoke(synth_nwb, tmp_path):
    """``neurocomplexity analyze`` runs end-to-end with default --formats.

    Exits 0 and writes ≥ 1 SVG figure file. Guards Tier 1.1.
    """
    outdir = tmp_path / "out"
    rc = main(["analyze", str(synth_nwb), "-o", str(outdir)])
    assert rc == 0
    assert (outdir / "results.json").exists()
    # Default formats are svg + tiff + jpg. At least one analysis must
    # have produced a figure triplet.
    svgs = list(outdir.glob("*.svg"))
    assert len(svgs) >= 1, f"no SVG figures emitted into {outdir}"


def test_cli_analyze_rejects_pdf_format(synth_nwb, tmp_path):
    """``--formats pdf`` is rejected by argparse before the pipeline runs.

    Guards against the regression where ``pdf`` could leak to
    ``save_publication`` and crash mid-pipeline.
    """
    outdir = tmp_path / "out"
    with pytest.raises(SystemExit):
        main(["analyze", str(synth_nwb), "-o", str(outdir),
              "--formats", "pdf"])


def test_cli_analyze_no_figures(synth_nwb, tmp_path):
    """``--no-figures`` short-circuits rendering; only results.json written."""
    outdir = tmp_path / "out"
    rc = main(["analyze", str(synth_nwb), "-o", str(outdir),
               "--no-figures"])
    assert rc == 0
    assert (outdir / "results.json").exists()
    assert list(outdir.glob("*.svg")) == []
