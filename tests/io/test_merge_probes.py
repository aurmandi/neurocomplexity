import warnings

import numpy as np
import pandas as pd
import pytest

from neurocomplexity.core.recording import SpikeRecording
from neurocomplexity.core.provenance import ProvenanceRecord


def _rec(ids, n_spikes_each=2, duration=1.0):
    spike_times = []
    spike_uids = []
    for i, u in enumerate(ids):
        for k in range(n_spikes_each):
            spike_times.append((i + 1) * 0.1 + k * 0.01)
            spike_uids.append(u)
    units = pd.DataFrame({"id": ids, "peak_channel": list(range(len(ids)))})
    return SpikeRecording(
        spike_times=np.asarray(spike_times, dtype=np.float64),
        unit_ids=np.asarray(spike_uids, dtype=np.int64),
        units=units,
        populations={"all": np.array([True] * len(ids))},
        duration=duration,
        sampling_rate=None,
        source=ProvenanceRecord.for_memory("test"),
    )


def test_merge_two_probes_no_id_collision():
    a = _rec([0, 1])
    b = _rec([10, 11])
    merged = SpikeRecording.merge_probes({"A": a, "B": b})
    assert set(merged.units["id"]) == {0, 1, 2, 3}
    assert set(merged.units["probe"]) == {"A", "B"}
    assert "probe_A" in merged.populations
    assert "probe_B" in merged.populations
    assert merged.populations["probe_A"].sum() == 2
    assert merged.populations["probe_B"].sum() == 2


def test_merge_with_id_collision_emits_warning_and_tuple_ids():
    a = _rec([0, 1, 42])
    b = _rec([10, 42])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        merged = SpikeRecording.merge_probes({"A": a, "B": b})
    collision_warnings = [w for w in caught if "collision" in str(w.message).lower()]
    assert len(collision_warnings) == 1
    # Colliding IDs preserved in original_id column; canonical id is a fresh integer code
    orig = list(merged.units["original_id"])
    assert any(str(x) == "('A', 42)" for x in orig)
    assert any(str(x) == "('B', 42)" for x in orig)
    # Non-colliding ids preserved as-is
    assert 0 in [o for o in orig if isinstance(o, int)]


def test_merge_align_durations_max():
    a = _rec([0], duration=1.0)
    b = _rec([10], duration=2.0)
    merged = SpikeRecording.merge_probes({"A": a, "B": b}, align_durations="max")
    assert merged.duration == 2.0


def test_merge_align_durations_min_drops_late_spikes():
    a = _rec([0], duration=1.0)
    b = _rec([10], duration=2.0)
    from dataclasses import replace
    b = replace(b, spike_times=np.array([0.1, 1.5], dtype=np.float64),
                unit_ids=np.array([10, 10], dtype=np.int64))
    merged = SpikeRecording.merge_probes({"A": a, "B": b}, align_durations="min")
    assert merged.duration == 1.0
    # Late B spike should be dropped — look it up by original_id since id is recoded
    b_idx = merged.units.loc[merged.units["original_id"] == 10, "id"].iloc[0]
    b_spikes_in_merged = (merged.unit_ids == b_idx).sum()
    assert b_spikes_in_merged == 1


def test_merge_align_durations_strict_raises_on_mismatch():
    a = _rec([0], duration=1.0)
    b = _rec([10], duration=2.0)
    with pytest.raises(ValueError, match="differ"):
        SpikeRecording.merge_probes({"A": a, "B": b}, align_durations="strict")


def test_merge_interval_collision_raises():
    from dataclasses import replace
    a = _rec([0])
    a = replace(a, intervals={"stim": pd.DataFrame({"start_time": [0.0], "stop_time": [0.5]})})
    b = _rec([10])
    b = replace(b, intervals={"stim": pd.DataFrame({"start_time": [0.1], "stop_time": [0.6]})})
    with pytest.raises(KeyError, match="stim"):
        SpikeRecording.merge_probes({"A": a, "B": b})


def test_merge_preserves_original_populations_with_prefix():
    from dataclasses import replace
    a = _rec([0, 1])
    a = replace(a, populations={
        "all": np.array([True, True]),
        "excitatory": np.array([True, False]),
    })
    b = _rec([10, 11])
    merged = SpikeRecording.merge_probes({"A": a, "B": b})
    assert "probe_A_excitatory" in merged.populations
    assert merged.populations["probe_A_excitatory"].sum() == 1


def test_merge_propagates_filtered_when_all_inputs_filtered():
    from dataclasses import replace
    a = replace(_rec([0]), _filtered=True)
    b = replace(_rec([10]), _filtered=True)
    merged = SpikeRecording.merge_probes({"A": a, "B": b})
    assert merged._filtered is True


def test_merge_filtered_false_when_any_input_not_filtered():
    from dataclasses import replace
    a = replace(_rec([0]), _filtered=True)
    b = _rec([10])  # _filtered=False
    merged = SpikeRecording.merge_probes({"A": a, "B": b})
    assert merged._filtered is False


def test_anatomy_csv_explicit_format_missing_columns_raises_value_error(tmp_path):
    """Regression: format='csv' with missing channel/area raised KeyError: None."""
    import pandas as pd
    from neurocomplexity.io._anatomy import add_anatomy
    p = tmp_path / "bad.csv"
    pd.DataFrame({"foo": [1, 2], "bar": [3, 4]}).to_csv(p, index=False)
    a = _rec([0, 1])
    with pytest.raises(ValueError, match="channel.*area"):
        add_anatomy(a, p, format="csv")
