from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neurocomplexity.io._qc import add_quality
from neurocomplexity.core.recording import SpikeRecording
from neurocomplexity.core.provenance import ProvenanceRecord


def _rec_with_ids(ids):
    return SpikeRecording(
        spike_times=np.array([0.1] * len(ids), dtype=np.float64),
        unit_ids=np.array(ids, dtype=np.int64),
        units=pd.DataFrame({"id": ids, "peak_channel": list(range(len(ids)))}),
        populations={"all": np.array([True] * len(ids))},
        duration=1.0,
        sampling_rate=None,
        source=ProvenanceRecord.for_memory("test"),
    )


def _write_bombcell_csv(path: Path, ids):
    df = pd.DataFrame({
        "cluster_id": ids,
        "useTheseTimesStart": [0.0] * len(ids),
        "nPeaks": [1] * len(ids),
        "rawAmplitude": [60.0] * len(ids),
        "percentageSpikesMissing_gaussian": [2.0] * len(ids),
        "unitType": [1, 2, 0][:len(ids)],
        "presenceRatio": [0.95, 0.7, 0.4][:len(ids)],
        "fractionRPVs_estimatedTauR": [0.001, 0.05, 0.2][:len(ids)],
        "rawAmplitude_p1": [60.0] * len(ids),
        "signalToNoiseRatio": [10.0, 5.0, 2.0][:len(ids)],
    })
    df.to_csv(path, index=False)
    return path


def test_add_quality_bombcell_normalises_columns(tmp_path):
    rec = _rec_with_ids([0, 1, 2])
    qc_path = _write_bombcell_csv(tmp_path / "bc.csv", [0, 1, 2])
    rec2 = add_quality(rec, qc_path, format="bombcell")
    assert "quality" in rec2.units.columns
    assert list(rec2.units["quality"]) == ["good", "mua", "noise"]
    assert "presence_ratio" in rec2.units.columns
    assert "qc_source" in rec2.units.columns
    assert (rec2.units["qc_source"] == "bombcell").all()


def test_add_quality_auto_detects_bombcell(tmp_path):
    rec = _rec_with_ids([0, 1, 2])
    qc_path = _write_bombcell_csv(tmp_path / "bc.csv", [0, 1, 2])
    rec2 = add_quality(rec, qc_path)
    assert (rec2.units["qc_source"] == "bombcell").all()


def test_add_quality_appends_provenance(tmp_path):
    rec = _rec_with_ids([0, 1, 2])
    qc_path = _write_bombcell_csv(tmp_path / "bc.csv", [0, 1, 2])
    rec2 = add_quality(rec, qc_path, format="bombcell")
    assert len(rec2.attachments) == 1
    assert rec2.attachments[0].source_format == "quality:bombcell"
    assert rec2.attachments[0].source_hash != ""


def test_add_quality_unit_id_mismatch_raises(tmp_path):
    rec = _rec_with_ids([0, 1, 2])
    qc_path = _write_bombcell_csv(tmp_path / "bc.csv", [10, 20, 30])
    with pytest.raises(ValueError, match="unit_id"):
        add_quality(rec, qc_path, format="bombcell")


def test_add_quality_auto_raises_on_unknown_format(tmp_path):
    rec = _rec_with_ids([0])
    p = tmp_path / "unknown.csv"
    pd.DataFrame({"foo": [1], "bar": [2]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="auto-detect|format"):
        add_quality(rec, p)


def test_add_quality_does_not_set_filtered_flag(tmp_path):
    rec = _rec_with_ids([0, 1, 2])
    qc_path = _write_bombcell_csv(tmp_path / "bc.csv", [0, 1, 2])
    rec2 = add_quality(rec, qc_path, format="bombcell")
    assert rec2._filtered is False
