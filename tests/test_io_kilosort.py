"""Tests for from_kilosort."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import neurocomplexity as nc
from tests._sorter_fixtures import write_sorter_directory, DEFAULT_SAMPLE_RATE


def test_from_kilosort_roundtrip(tmp_path):
    trains = {
        7: np.array([0.20, 0.55, 1.00]),
        8: np.array([0.10, 0.90, 1.80, 2.40]),
    }
    ks_label = pd.DataFrame({
        "cluster_id": [7, 8],
        "KSLabel": ["good", "mua"],
    })
    write_sorter_directory(tmp_path, trains, cluster_kslabel=ks_label)

    rec = nc.io.from_kilosort(tmp_path)

    assert rec.n_units == 2
    assert rec.n_spikes == sum(len(v) for v in trains.values())
    assert set(rec.units["id"]) == {7, 8}
    assert dict(zip(rec.units["id"], rec.units["quality"])) == {7: "good", 8: "mua"}
    assert rec.source.source_format == "kilosort"
    assert rec.sampling_rate == DEFAULT_SAMPLE_RATE


def test_from_kilosort_missing_kslabel_raises(tmp_path):
    trains = {0: np.array([0.1, 0.2])}
    write_sorter_directory(tmp_path, trains)
    with pytest.raises(Exception, match="cluster_KSLabel.tsv"):
        nc.io.from_kilosort(tmp_path)
