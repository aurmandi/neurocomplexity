import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neurocomplexity.io._anatomy import add_anatomy
from neurocomplexity.core.recording import SpikeRecording
from neurocomplexity.core.provenance import ProvenanceRecord


def _rec_with_channels(channels):
    n = len(channels)
    return SpikeRecording(
        spike_times=np.array([0.1] * n, dtype=np.float64),
        unit_ids=np.arange(n, dtype=np.int64),
        units=pd.DataFrame({"id": list(range(n)), "peak_channel": channels}),
        populations={"all": np.array([True] * n)},
        duration=1.0,
        sampling_rate=None,
        source=ProvenanceRecord.for_memory("test"),
    )


def _write_brainglobe(path, n_channels=4):
    df = pd.DataFrame({
        "Channel": list(range(n_channels)),
        "Brain region acronym": ["CA1", "CA1", "DG", "DG"][:n_channels],
        "Brain region": ["Field CA1", "Field CA1", "Dentate gyrus", "Dentate gyrus"][:n_channels],
        "AP": [-2.0, -2.0, -2.1, -2.1][:n_channels],
        "DV": [-1.5, -1.5, -1.8, -1.8][:n_channels],
        "ML": [1.0, 1.0, 1.1, 1.1][:n_channels],
    })
    df.to_csv(path, index=False)
    return path


def test_add_anatomy_brainglobe_joins_by_peak_channel(tmp_path):
    rec = _rec_with_channels([0, 1, 2, 3])
    p = _write_brainglobe(tmp_path / "anat.csv")
    rec2 = add_anatomy(rec, p, format="brainglobe")
    assert list(rec2.units["brain_area"]) == ["CA1", "CA1", "DG", "DG"]
    assert (rec2.units["anatomy_source"] == "brainglobe").all()
    assert "ccf_ap" in rec2.units.columns


def test_add_anatomy_auto_detects_brainglobe(tmp_path):
    rec = _rec_with_channels([0, 1, 2, 3])
    p = _write_brainglobe(tmp_path / "anat.csv")
    rec2 = add_anatomy(rec, p)
    assert (rec2.units["anatomy_source"] == "brainglobe").all()


def test_add_anatomy_warns_on_missing_channels(tmp_path):
    rec = _rec_with_channels([0, 1, 99, 100])
    p = _write_brainglobe(tmp_path / "anat.csv", n_channels=4)  # has channels 0..3
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rec2 = add_anatomy(rec, p, format="brainglobe")
    msgs = [str(w.message) for w in caught]
    assert any("2 channel" in m or "missing" in m.lower() for m in msgs)
    assert pd.isna(rec2.units.loc[rec2.units["peak_channel"] == 99, "brain_area"].iloc[0])


def test_add_anatomy_generic_csv(tmp_path):
    rec = _rec_with_channels([0, 1])
    p = tmp_path / "simple.csv"
    pd.DataFrame({"channel": [0, 1], "area": ["V1", "V2"]}).to_csv(p, index=False)
    rec2 = add_anatomy(rec, p)
    assert list(rec2.units["brain_area"]) == ["V1", "V2"]
    assert (rec2.units["anatomy_source"] == "csv").all()


def test_add_anatomy_appends_provenance(tmp_path):
    rec = _rec_with_channels([0, 1, 2, 3])
    p = _write_brainglobe(tmp_path / "anat.csv")
    rec2 = add_anatomy(rec, p, format="brainglobe")
    assert len(rec2.attachments) == 1
    assert rec2.attachments[0].source_format == "anatomy:brainglobe"


def _write_sharptrack_mat(path, n_channels=4):
    from scipy.io import savemat
    probe_ccf = {
        "channels": np.arange(n_channels, dtype=np.int64) + 1,  # MATLAB 1-indexed
        "areas": np.array(["CA1", "CA1", "DG", "DG"][:n_channels], dtype=object),
        "areas_full": np.array(["Field CA1", "Field CA1", "Dentate gyrus", "Dentate gyrus"][:n_channels], dtype=object),
        "ap_um": np.array([-2000.0, -2000.0, -2100.0, -2100.0][:n_channels]),
        "dv_um": np.array([-1500.0, -1500.0, -1800.0, -1800.0][:n_channels]),
        "ml_um": np.array([1000.0, 1000.0, 1100.0, 1100.0][:n_channels]),
    }
    savemat(path, {"probe_ccf": probe_ccf})
    return path


def test_add_anatomy_sharptrack_mat(tmp_path):
    rec = _rec_with_channels([0, 1, 2, 3])
    p = _write_sharptrack_mat(tmp_path / "anat.mat")
    rec2 = add_anatomy(rec, p, format="sharptrack")
    assert list(rec2.units["brain_area"]) == ["CA1", "CA1", "DG", "DG"]
    assert (rec2.units["anatomy_source"] == "sharptrack").all()


def test_add_anatomy_missing_probe_ccf_struct_raises(tmp_path):
    from scipy.io import savemat
    p = tmp_path / "wrong.mat"
    savemat(p, {"not_probe_ccf": np.array([1, 2, 3])})
    rec = _rec_with_channels([0, 1])
    with pytest.raises(ValueError, match="probe_ccf"):
        add_anatomy(rec, p, format="sharptrack")
