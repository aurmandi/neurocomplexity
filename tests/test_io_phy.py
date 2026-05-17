"""Tests for from_phy (and the shared _sorter_output helper through it)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import neurocomplexity as nc
from tests._sorter_fixtures import write_sorter_directory, DEFAULT_SAMPLE_RATE


def _phy_cluster_info(unit_ids, groups):
    return pd.DataFrame({
        "cluster_id": list(unit_ids),
        "group": list(groups),
        "KSLabel": ["good"] * len(unit_ids),
        "Amplitude": np.linspace(50.0, 100.0, len(unit_ids)),
        "ContamPct": np.linspace(0.1, 5.0, len(unit_ids)),
        "depth": np.linspace(100.0, 500.0, len(unit_ids)),
        "ch": list(range(len(unit_ids))),
        "fr": np.linspace(1.0, 10.0, len(unit_ids)),
        "n_spikes": [5, 4, 6][: len(unit_ids)],
        "sh": [0] * len(unit_ids),
    })


def test_from_phy_roundtrip(tmp_path):
    trains = {
        0: np.array([0.10, 0.42, 1.07, 2.55, 3.91]),
        1: np.array([0.05, 0.93, 2.10, 4.80]),
        2: np.array([0.30, 0.70, 1.50, 2.20, 3.00, 4.10]),
    }
    info = _phy_cluster_info([0, 1, 2], ["good", "mua", "good"])
    write_sorter_directory(tmp_path, trains, cluster_info=info)

    rec = nc.io.from_phy(tmp_path)

    assert rec.n_units == 3
    assert rec.n_spikes == sum(len(v) for v in trains.values())
    assert np.all(np.diff(rec.spike_times) >= 0)

    for uid, st in trains.items():
        recovered = np.sort(rec.spike_times[rec.unit_ids == uid])
        np.testing.assert_allclose(recovered, st, atol=1.0 / DEFAULT_SAMPLE_RATE)

    cols = set(rec.units.columns)
    assert {"id", "quality", "firing_rate", "depth",
            "peak_channel", "amplitude", "contam_pct", "n_spikes"} <= cols

    assert list(rec.units.sort_values("id")["quality"]) == ["good", "mua", "good"]
    assert rec.source.source_format == "phy"
    assert rec.populations["all"].sum() == 3


def test_from_phy_uses_cluster_group_fallback(tmp_path):
    trains = {0: np.array([0.1, 0.3]), 1: np.array([0.2])}
    group = pd.DataFrame({"cluster_id": [0, 1], "group": ["good", "noise"]})
    write_sorter_directory(tmp_path, trains, cluster_group=group)

    rec = nc.io.from_phy(tmp_path)
    assert dict(zip(rec.units["id"], rec.units["quality"])) == {0: "good", 1: "noise"}


def test_from_phy_missing_spike_clusters_falls_back_to_templates(tmp_path):
    trains = {0: np.array([0.1]), 1: np.array([0.2])}
    info = _phy_cluster_info([0, 1], ["good", "good"])
    write_sorter_directory(tmp_path, trains, cluster_info=info,
                            omit_spike_clusters=True)
    with pytest.warns(UserWarning, match="spike_templates"):
        rec = nc.io.from_phy(tmp_path)
    assert rec.n_spikes == 2


def test_from_phy_unknown_cluster_id_gets_unsorted_row(tmp_path):
    trains = {0: np.array([0.1]), 99: np.array([0.2])}
    info = _phy_cluster_info([0], ["good"])
    write_sorter_directory(tmp_path, trains, cluster_info=info)
    rec = nc.io.from_phy(tmp_path)
    assert set(rec.units["id"]) == {0, 99}
    assert rec.units.set_index("id").loc[99, "quality"] == "unsorted"


def test_from_phy_duration_override(tmp_path):
    trains = {0: np.array([0.1, 0.5])}
    info = _phy_cluster_info([0], ["good"])
    write_sorter_directory(tmp_path, trains, cluster_info=info)
    rec = nc.io.from_phy(tmp_path, duration=10.0)
    assert rec.duration == 10.0


def test_from_phy_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        nc.io.from_phy(tmp_path / "does_not_exist")


def test_from_phy_no_label_tables_raises(tmp_path):
    trains = {0: np.array([0.1])}
    write_sorter_directory(tmp_path, trains)
    with pytest.raises(Exception, match="cluster_info|cluster_group"):
        nc.io.from_phy(tmp_path)
