"""End-to-end acceptance test: from_kilosort -> add_quality -> filter_units -> analysis
produces no QualityControlWarning, while skipping add_quality + filter_units does."""
import warnings

import numpy as np
import pandas as pd

from neurocomplexity._warnings import QualityControlWarning, _reset_dedup
from neurocomplexity.core.recording import SpikeRecording
from neurocomplexity.core.provenance import ProvenanceRecord


def _synthetic_uncurated_rec():
    """Mimic what from_kilosort returns: no quality column, _filtered=False."""
    rng = np.random.default_rng(0)
    n_units = 5
    n_spikes = 1000
    spike_times = np.sort(rng.uniform(0.0, 10.0, n_spikes))
    spike_uids = rng.integers(0, n_units, n_spikes)
    return SpikeRecording(
        spike_times=spike_times,
        unit_ids=spike_uids,
        units=pd.DataFrame({"id": list(range(n_units)),
                            "peak_channel": list(range(n_units))}),
        populations={"all": np.array([True] * n_units)},
        duration=10.0,
        sampling_rate=30000.0,
        source=ProvenanceRecord.for_memory("synthetic_kilosort"),
    )


def setup_function():
    _reset_dedup()


def test_full_workflow_kilosort_then_quality_then_filter_then_analysis(tmp_path):
    from neurocomplexity.io import add_quality
    from neurocomplexity.analysis.branching import branching_ratio

    rec = _synthetic_uncurated_rec()

    qc = pd.DataFrame({
        "cluster_id": [0, 1, 2, 3, 4],
        "useTheseTimesStart": [0.0] * 5,
        "nPeaks": [1] * 5,
        "rawAmplitude": [60.0] * 5,
        "percentageSpikesMissing_gaussian": [2.0, 3.0, 50.0, 1.5, 80.0],
        "unitType": [1, 1, 2, 1, 0],
        "presenceRatio": [0.95, 0.9, 0.7, 0.92, 0.4],
        "fractionRPVs_estimatedTauR": [0.001, 0.002, 0.05, 0.001, 0.2],
        "signalToNoiseRatio": [10.0, 8.0, 5.0, 12.0, 1.5],
    })
    qc_path = tmp_path / "bc.csv"
    qc.to_csv(qc_path, index=False)

    rec = add_quality(rec, qc_path, format="bombcell")
    rec = rec.filter_units(quality="good")
    assert rec._filtered is True
    assert len(rec.units) == 3

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            branching_ratio(rec, bin_size=0.005, k_max=10)
        except Exception:
            pass
    qcw = [w for w in caught if issubclass(w.category, QualityControlWarning)]
    assert len(qcw) == 0, f"workflow that includes filter_units must not warn, got: {qcw}"


def test_skipping_qc_workflow_still_warns():
    from neurocomplexity.analysis.branching import branching_ratio
    rec = _synthetic_uncurated_rec()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            branching_ratio(rec, bin_size=0.005, k_max=10)
        except Exception:
            pass
    qcw = [w for w in caught if issubclass(w.category, QualityControlWarning)]
    assert len(qcw) == 1
