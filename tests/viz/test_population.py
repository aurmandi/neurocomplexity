import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from neurocomplexity.core.recording import SpikeRecording
from neurocomplexity.core.provenance import ProvenanceRecord


def _make_rec(n_units=20, duration=10.0, sr=30000.0):
    rng = np.random.default_rng(0)
    spike_times = []
    spike_uids = []
    for u in range(n_units):
        peak = rng.uniform(0, duration)
        t = peak + rng.normal(0, 0.5, size=50)
        t = np.clip(t, 0.01, duration - 0.01)
        spike_times.extend(t.tolist())
        spike_uids.extend([u] * len(t))
    spike_times = np.array(spike_times, dtype=np.float64)
    spike_uids = np.array(spike_uids, dtype=np.int64)
    order = np.argsort(spike_times)
    return SpikeRecording(
        spike_times=spike_times[order], unit_ids=spike_uids[order],
        units=pd.DataFrame({"id": list(range(n_units)),
                            "peak_channel": list(range(n_units))}),
        populations={"all": np.array([True] * n_units)},
        duration=duration, sampling_rate=sr,
        source=ProvenanceRecord.for_memory("test"),
    )


def test_heatmap_returns_figure_with_one_axes():
    from neurocomplexity.viz.population import figure_population_heatmap
    rec = _make_rec()
    fig = figure_population_heatmap(rec)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) >= 1
    plt.close(fig)


def test_heatmap_image_shape_matches_units_and_bins():
    from neurocomplexity.viz.population import figure_population_heatmap
    rec = _make_rec(n_units=15, duration=5.0)
    fig = figure_population_heatmap(rec, bin_size=0.1)
    images = fig.axes[0].get_images()
    assert len(images) == 1
    arr = images[0].get_array()
    n_rows, n_cols = arr.shape
    assert n_rows == 15
    expected_cols = int(np.floor(5.0 / 0.1))
    assert n_cols == expected_cols
    plt.close(fig)


def test_heatmap_sort_by_peak_time_orders_rows():
    from neurocomplexity.viz.population import figure_population_heatmap
    rec = _make_rec(n_units=8, duration=8.0)
    fig = figure_population_heatmap(rec, bin_size=0.1, sort_by="peak_time")
    arr = fig.axes[0].get_images()[0].get_array()
    peak_bins = np.argmax(arr, axis=1)
    assert np.all(np.diff(peak_bins) >= 0)
    plt.close(fig)


def test_heatmap_sort_by_unit_id_preserves_order():
    from neurocomplexity.viz.population import figure_population_heatmap
    rec = _make_rec(n_units=5)
    fig = figure_population_heatmap(rec, sort_by="unit_id")
    assert len(fig.axes[0].get_images()) == 1
    plt.close(fig)


def test_heatmap_unknown_sort_raises():
    from neurocomplexity.viz.population import figure_population_heatmap
    rec = _make_rec()
    with pytest.raises(ValueError, match="sort_by"):
        figure_population_heatmap(rec, sort_by="invalid")
    plt.close("all")
