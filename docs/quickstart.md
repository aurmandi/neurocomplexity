# Quickstart

This page shows the minimum to go from a recording to a result. For a full
walk-through see the [tutorial notebook](examples/tutorial.ipynb).

## Load a recording

```python
import neurocomplexity as nc
rec = nc.io.from_nwb("path/to/session.nwb")
print(f"{rec.n_spikes} spikes from {rec.n_units} units")
```

NWB support requires the optional `nwb` extra (`pip install
"neurocomplexity[nwb]"`).

### From a Phy curation directory

```python
rec = nc.io.from_phy("path/to/phy_output/")
rec = rec.filter_units(quality=["good"])
```

### From raw Kilosort output (before Phy curation)

Raw Kilosort output has no curator-assigned quality, so attach an automated QC
table (Bombcell / ecephys_spike_sorting / SpikeInterface) and filter on it.
Without this step `neurocomplexity` emits a `QualityControlWarning` at analysis
time because uncurated sorter output is dominated by noise units and MUA.

```python
rec = nc.io.from_kilosort("path/to/kilosort_output/")
rec = nc.io.add_quality(rec, "path/to/bombcell/cluster_metrics.csv")
rec = rec.filter_units(quality="good")
```

### From any SpikeInterface sorter

```python
import spikeinterface.extractors as se
sorting = se.read_phy("path/to/phy_output/")
rec = nc.io.from_spikeinterface(sorting)
```

The SpikeInterface bridge is a soft dependency — install with
`pip install "neurocomplexity[spikeinterface]"`.

## Run an analysis

Every analysis returns a frozen dataclass that carries the numeric output
**and** the parameters used. Analysis functions live under the
``nc.analysis`` namespace.

```python
# Branching ratio
m = nc.analysis.wilting_mr(rec, populations=["all"], bin_size_ms=4)
print(f"m_hat = {m.m:.3f}, R^2 = {m.r_squared:.3f}")

# Criticality exponents (with Sethna consistency)
c = nc.analysis.criticality(rec, populations=["VISp"])
print(f"alpha_s={c.alpha_s:.2f}, alpha_t={c.alpha_t:.2f}, "
      f"gamma_pred={c.gamma_predicted:.2f}, gamma_fit={c.gamma_fit:.2f}")

# Effective connectivity
te = nc.analysis.transfer_entropy(rec, populations=["VISp", "LGd", "CA1"], bin_size_ms=10)

# Geometry
pr = nc.analysis.dimensionality(rec, populations=["VISp"], bin_size_ms=10)
mfd = nc.analysis.manifold(rec, populations=["VISp"], method="pca", dims=2)

# Complexity
mse = nc.analysis.multiscale_entropy(rec, populations=["VISp"])
lmc = nc.analysis.lmc_complexity(rec, populations=["VISp"], bin_size_s=0.05)
```

See [`complexity_measures.md`](complexity_measures.md) for when to use
``multiscale_entropy`` vs ``lmc_complexity``.

## Add inference

**Bootstrap confidence intervals** — block bootstrap over time, per analysis:

```python
from neurocomplexity.inference import bootstrap
ci = bootstrap(m, rec, n=1000, block_seconds=10.0, seed=0)
print(f"95% CI: [{ci.ci_lower:.3f}, {ci.ci_upper:.3f}]")
```

**Surrogate null tests** — Phipson-Smyth-floored p-values with optional
Benjamini-Hochberg FDR across matrices / vectors:

```python
from neurocomplexity.inference import test
null = test(te, rec, surrogate="isi_shuffle", n=500, seed=0,
            alternative="greater", fdr=True)
print(null.p_value_fdr.shape, null.effect_size.shape)
```

For TE we recommend `isi_shuffle`: it destroys cross-unit timing while
preserving each unit's ISI distribution exactly, so a significant TE
cannot be explained by per-unit rate or burstiness alone. `spike_dither`
preserves only approximate rates and is too soft a null for connectivity
inference — the package's calibration suite uses `isi_shuffle` for the
TE Type-I rate test. See `docs/inference.md` for the full
"choose your null" table.

Two-sided tests use the conventional `2 * min(p_greater, p_less)` clipped
at 1, which is robust to skewed null distributions. Available
``alternative`` values: ``"greater"`` (default), ``"less"``, ``"two-sided"``.

Available surrogates:

| Method               | What it preserves                         | When to use |
|----------------------|-------------------------------------------|-------------|
| ``spike_dither``     | Per-unit rate (approximately), count      | Soft rate-only null; suitable for autonomy and exploratory checks |
| ``isi_shuffle``      | Per-unit ISI distribution exactly         | **Recommended for TE / PID / connectivity** (see "choose your null" in `docs/inference.md`) |
| ``interval_shuffle`` | Within-interval ordering, trial structure | Trial-based experiments |

``interval_shuffle`` requires non-overlapping intervals on the recording
(see :func:`~neurocomplexity.io.add_trials`); it will raise ``ValueError``
on overlapping windows to prevent silent corruption.

## Publication figures

Every result dataclass has a per-result figure function in `neurocomplexity.viz`.
All figure functions accept ``palette=`` (one of ``"forest"``, ``"wine"``,
``"sage"``), ``panel_label=``, and ``ax=`` for composite layouts.

```python
from neurocomplexity import viz

fig = viz.figure_criticality(result, palette="forest")
paths = viz.save_publication(fig, "fig1_criticality", tiff_dpi=600)
# -> {"svg": ..., "tiff": ..., "jpg": ...}
```

``save_publication`` always writes SVG (vector) + TIFF (LZW-compressed) + JPG
(quality 95, 600 dpi). Set ``tiff_dpi=1200`` for camera-ready TIFFs. PDF output
is intentionally not supported.

Each analysis result gets its own publication-ready figure. Compose
multi-panel layouts yourself in your manuscript figure-prep tool of choice.

## Run the benchmark suite against the published baseline

```bash
python -m neurocomplexity benchmark --reps 50 -o my_baseline.csv
```

## Warnings the package may emit

| Class                                                         | When |
|---------------------------------------------------------------|------|
| ``nc.warnings.QualityControlWarning``                         | An analysis was run on a recording with no unit-quality column attached. Apply :func:`~neurocomplexity.io.add_quality` or pass a curated NWB file. |
| ``nc.warnings.StationarityWarning``                           | The population rate drifts, is heteroskedastic, or has CV > acceptance. Inspect via ``nc.analysis.stationarity(rec)`` and crop to a stationary epoch. |
| ``nc.warnings.MemoryAllocationWarning``                       | A binning step would allocate more than ~1 GiB of dense matrix. Pass a coarser ``bin_size_ms`` or fewer units. |

All three are subclasses of :class:`UserWarning` and can be filtered or
upgraded to errors via ``warnings.filterwarnings``.

## Progress bars

Silent by default. Enable globally for long-running null tests / bootstraps:

```python
nc.set_progress(True)
```
