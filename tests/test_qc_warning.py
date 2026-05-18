import warnings as _warnings

import numpy as np
import pandas as pd
import pytest

from neurocomplexity._warnings import (
    QualityControlWarning,
    _warn_if_uncurated,
    _reset_dedup,
)
from neurocomplexity.core.recording import SpikeRecording
from neurocomplexity.core.provenance import ProvenanceRecord


def _rec(units_df):
    return SpikeRecording(
        spike_times=np.array([0.1], dtype=np.float64),
        unit_ids=np.array([0], dtype=np.int64),
        units=units_df,
        populations={"all": np.array([True] * len(units_df))},
        duration=1.0,
        sampling_rate=None,
        source=ProvenanceRecord.for_memory("test"),
    )


def setup_function():
    _reset_dedup()


def test_warns_when_no_curation_columns_and_not_filtered():
    rec = _rec(pd.DataFrame({"id": [0]}))
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        _warn_if_uncurated(rec, "branching_ratio")
    assert any(issubclass(w.category, QualityControlWarning) for w in caught)
    msg = str(caught[-1].message)
    assert "branching_ratio" in msg
    assert "add_quality" in msg


def test_no_warning_when_quality_column_present():
    rec = _rec(pd.DataFrame({"id": [0], "quality": ["good"]}))
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        _warn_if_uncurated(rec, "branching_ratio")
    assert not any(issubclass(w.category, QualityControlWarning) for w in caught)


def test_no_warning_when_filtered_flag_true():
    from dataclasses import replace
    rec = replace(_rec(pd.DataFrame({"id": [0]})), _filtered=True)
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        _warn_if_uncurated(rec, "branching_ratio")
    assert not any(issubclass(w.category, QualityControlWarning) for w in caught)


def test_no_warning_when_presence_ratio_column_present():
    rec = _rec(pd.DataFrame({"id": [0], "presence_ratio": [0.9]}))
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        _warn_if_uncurated(rec, "branching_ratio")
    assert not any(issubclass(w.category, QualityControlWarning) for w in caught)


def test_no_warning_when_isi_violations_ratio_present():
    rec = _rec(pd.DataFrame({"id": [0], "isi_violations_ratio": [0.01]}))
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        _warn_if_uncurated(rec, "branching_ratio")
    assert not any(issubclass(w.category, QualityControlWarning) for w in caught)


def test_no_warning_when_kslabel_present():
    rec = _rec(pd.DataFrame({"id": [0], "KSLabel": ["good"]}))
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        _warn_if_uncurated(rec, "branching_ratio")
    assert not any(issubclass(w.category, QualityControlWarning) for w in caught)


def test_dedup_within_session_same_rec_same_analysis():
    rec = _rec(pd.DataFrame({"id": [0]}))
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        _warn_if_uncurated(rec, "branching_ratio")
        _warn_if_uncurated(rec, "branching_ratio")
    qcw = [w for w in caught if issubclass(w.category, QualityControlWarning)]
    assert len(qcw) == 1


def test_no_dedup_across_different_analyses():
    rec = _rec(pd.DataFrame({"id": [0]}))
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        _warn_if_uncurated(rec, "branching_ratio")
        _warn_if_uncurated(rec, "transfer_entropy")
    qcw = [w for w in caught if issubclass(w.category, QualityControlWarning)]
    assert len(qcw) == 2


def test_branching_ratio_emits_warning_on_uncurated_recording():
    from neurocomplexity.analysis.branching import branching_ratio
    rec = _rec(pd.DataFrame({"id": [0]}))
    # Add enough spikes to satisfy branching_ratio's input requirements
    import numpy as np
    from dataclasses import replace
    st = np.linspace(0.01, 0.99, 500)
    uid = np.zeros(500, dtype=np.int64)
    rec = replace(rec, spike_times=st, unit_ids=uid)
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        try:
            branching_ratio(rec, bin_size=0.001, k_max=10)
        except Exception:
            pass  # we only care about the warning, not the numerical result
    qcw = [w for w in caught if issubclass(w.category, QualityControlWarning)]
    assert len(qcw) == 1
    assert "branching_ratio" in str(qcw[0].message)


def test_filtered_recording_does_not_warn_in_analysis():
    from dataclasses import replace
    from neurocomplexity.analysis.branching import branching_ratio
    import numpy as np
    rec = _rec(pd.DataFrame({"id": [0]}))
    st = np.linspace(0.01, 0.99, 500)
    uid = np.zeros(500, dtype=np.int64)
    rec = replace(rec, spike_times=st, unit_ids=uid, _filtered=True)
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        try:
            branching_ratio(rec, bin_size=0.001, k_max=10)
        except Exception:
            pass
    qcw = [w for w in caught if issubclass(w.category, QualityControlWarning)]
    assert len(qcw) == 0


def test_warnings_submodule_accessible_as_nc_warnings():
    import neurocomplexity as nc
    assert hasattr(nc, "warnings")
    assert nc.warnings.QualityControlWarning is QualityControlWarning
