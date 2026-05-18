from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neurocomplexity.io._trials import add_trials
from neurocomplexity.core.recording import SpikeRecording
from neurocomplexity.core.provenance import ProvenanceRecord


def _rec(duration=10.0):
    return SpikeRecording(
        spike_times=np.array([0.1, 0.2], dtype=np.float64),
        unit_ids=np.array([0, 0], dtype=np.int64),
        units=pd.DataFrame({"id": [0]}),
        populations={"all": np.array([True])},
        duration=duration,
        sampling_rate=None,
        source=ProvenanceRecord.for_memory("test"),
    )


def _write_trials_csv(path, n=3, stop_max=5.0):
    df = pd.DataFrame({
        "start_time": np.linspace(0.0, stop_max - 1.0, n),
        "stop_time": np.linspace(1.0, stop_max, n),
        "condition": ["A", "B", "A"][:n],
    })
    df.to_csv(path, index=False)
    return path


def test_add_trials_csv_populates_intervals(tmp_path):
    rec = _rec()
    p = _write_trials_csv(tmp_path / "trials.csv")
    rec2 = add_trials(rec, p, name="stim")
    assert "stim" in rec2.intervals
    assert list(rec2.intervals["stim"].columns)[:2] == ["start_time", "stop_time"]
    assert "condition" in rec2.intervals["stim"].columns
    assert len(rec2.intervals["stim"]) == 3


def test_add_trials_tsv(tmp_path):
    rec = _rec()
    p = tmp_path / "trials.tsv"
    pd.DataFrame({
        "start_time": [0.1, 1.0],
        "stop_time": [0.5, 2.0],
    }).to_csv(p, sep="\t", index=False)
    rec2 = add_trials(rec, p, name="t")
    assert len(rec2.intervals["t"]) == 2


def test_add_trials_renames_custom_columns(tmp_path):
    rec = _rec()
    p = tmp_path / "trials.csv"
    pd.DataFrame({
        "onset": [0.1, 1.0],
        "offset": [0.5, 2.0],
    }).to_csv(p, index=False)
    rec2 = add_trials(rec, p, name="t", start_column="onset", stop_column="offset")
    assert "start_time" in rec2.intervals["t"].columns
    assert "stop_time" in rec2.intervals["t"].columns


def test_add_trials_out_of_bounds_raises(tmp_path):
    rec = _rec(duration=2.0)
    p = tmp_path / "trials.csv"
    pd.DataFrame({
        "start_time": [0.1, 1.0],
        "stop_time": [0.5, 5.0],
    }).to_csv(p, index=False)
    with pytest.raises(ValueError, match="row 1"):
        add_trials(rec, p, name="t")


def test_add_trials_start_after_stop_raises(tmp_path):
    rec = _rec()
    p = tmp_path / "trials.csv"
    pd.DataFrame({
        "start_time": [0.1, 2.0],
        "stop_time": [0.5, 1.0],
    }).to_csv(p, index=False)
    with pytest.raises(ValueError, match="row 1"):
        add_trials(rec, p, name="t")


def test_add_trials_name_collision_raises(tmp_path):
    from dataclasses import replace
    rec = _rec()
    rec = replace(rec, intervals={"stim": pd.DataFrame({"start_time": [0], "stop_time": [1]})})
    p = _write_trials_csv(tmp_path / "trials.csv")
    with pytest.raises(KeyError, match="stim"):
        add_trials(rec, p, name="stim")


def test_add_trials_appends_provenance(tmp_path):
    rec = _rec()
    p = _write_trials_csv(tmp_path / "trials.csv")
    rec2 = add_trials(rec, p, name="stim")
    assert len(rec2.attachments) == 1
    assert rec2.attachments[0].source_format == "trials:csv"
