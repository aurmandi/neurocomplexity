"""Builder for the comprehensive neurocomplexity tutorial notebook.

Run via:  python tutorial/_build_comprehensive.py

Writes ``tutorial/comprehensive_tutorial.ipynb`` from the structured cell
definitions below. All code cells use synthetic data drawn from the
package's own simulators (no external downloads required) so the
notebook is fully reproducible offline.
"""
from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat import v4

OUT_PATH = Path(__file__).resolve().parent / "comprehensive_tutorial.ipynb"

nb = v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.12",
    },
    "title": "neurocomplexity — comprehensive tutorial",
}

cells: list = []


def md(*lines: str) -> None:
    cells.append(v4.new_markdown_cell("\n".join(lines)))


def code(*lines: str) -> None:
    cells.append(v4.new_code_cell("\n".join(lines)))


# =========================================================================
# Section 0 — Welcome
# =========================================================================
md(
    "# neurocomplexity — comprehensive tutorial",
    "",
    "This notebook is a one-sitting walk-through of the whole package. It "
    "assumes you have written some Python and know what a NumPy array is. "
    "It does **not** assume you have ever opened a Phy directory, run "
    "Kilosort, or used SpikeInterface; those tools are explained here from "
    "scratch before they are used.",
    "",
    "The notebook is built in three layers. The first four sections are "
    "pure background — what extracellular recording produces, how spike "
    "sorting turns raw voltages into a list of timestamped spikes, and "
    "what the three most common output formats actually look like on "
    "disk. The next ten sections introduce the package's data model and "
    "loaders, with synthetic Phy and Kilosort directories built on the "
    "fly so you can run everything offline. The last fourteen sections "
    "cover every analysis the package ships with, every inference tool, "
    "and the visualisation layer, each illustrated on synthetic data with "
    "a known ground truth.",
    "",
    "Run it top-to-bottom on a laptop and it should complete in under ten "
    "minutes. The intuition lives in the prose; the code is there so you "
    "can poke at the numbers.",
    "",
    "---",
)


# =========================================================================
# Section 1 — Background: extracellular recording
# =========================================================================
md(
    "## 1 Background: what is a spike-sorted recording?",
    "",
    "Almost everything `neurocomplexity` does starts from one object: a "
    "list of timestamps, each tagged with the identity of the neuron that "
    "fired the spike. This section explains where those timestamps come "
    "from.",
    "",
    "### 1.1 The raw signal",
    "",
    "When an electrode is implanted close to a population of neurons, "
    "every action potential fired by a nearby cell produces a small "
    "voltage deflection on the electrode. A modern device — a Neuropixels "
    "probe, for example — has 384 such electrodes packed into one shank, "
    "sampled at 30 kHz. The raw recording is therefore a "
    "`(n_samples, 384)` array of int16 voltages, typically tens to "
    "hundreds of gigabytes per session.",
    "",
    "That raw array is not what neuroscientists analyse. Action potentials "
    "are buried in noise and are picked up by several neighbouring "
    "electrodes at once. To get from raw voltage to *which neuron fired "
    "when*, the signal must be **spike sorted**.",
    "",
    "### 1.2 Spike sorting in one paragraph",
    "",
    "Spike sorting has two steps. **Detection** finds every voltage event "
    "sharp enough to be a likely spike. **Clustering** groups events "
    "whose waveforms look alike, on the assumption that events with "
    "similar spatial and temporal waveforms come from the same neuron. "
    "The output of clustering is, for each detected event, a timestamp "
    "(in samples since the start of recording) and a cluster label (an "
    "integer identifying which putative neuron it belongs to).",
    "",
    "### 1.3 Where this package fits in",
    "",
    "`neurocomplexity` does **not** perform spike sorting. It takes the "
    "sorted output and computes complexity, criticality, and "
    "information-flow measures on it. The next three sections explain the "
    "three most common ways that sorted output is stored on disk — "
    "Kilosort, Phy, and SpikeInterface — and how each one maps onto a "
    "`SpikeRecording` in our package.",
)


# =========================================================================
# Section 2 — Kilosort
# =========================================================================
md(
    "## 2 Background: Kilosort",
    "",
    "Kilosort (currently Kilosort 4, *Pachitariu et al. 2024*) is the de "
    "facto standard for spike sorting high-density extracellular "
    "recordings. It is a Python/PyTorch program that takes a raw `.dat` "
    "file plus a probe-geometry file, runs template-matching on the GPU, "
    "and writes a directory of small files describing the sorted output. "
    "A typical Kilosort output directory contains:",
    "",
    "| File | Type | What it stores |",
    "|---|---|---|",
    "| `spike_times.npy` | int64 array | One row per detected spike; values are sample indices into the raw `.dat` |",
    "| `spike_templates.npy` | int32 array | Same length as `spike_times`; the template ID Kilosort matched each spike to |",
    "| `spike_clusters.npy` | int32 array | Same length again; the cluster ID after any automatic or manual merges. **This is what downstream tools should read.** |",
    "| `amplitudes.npy` | float | Per-spike amplitude |",
    "| `templates.npy` | float | The template waveforms themselves |",
    "| `channel_map.npy`, `channel_positions.npy` | int / float | Probe geometry |",
    "| `cluster_KSLabel.tsv` | tab-separated table | One row per cluster: `cluster_id`, `KSLabel` (good or mua, Kilosort's automatic quality judgement) |",
    "| `params.py` | Python source | A small file Kilosort writes that defines `sample_rate`, `n_channels_dat`, `dtype`, etc. |",
    "",
    "Two points to internalise. First, spike times are in *samples*, not "
    "seconds — converting to seconds requires `sample_rate` from "
    "`params.py`, which our loader handles for you. Second, "
    "`cluster_KSLabel.tsv` carries *automatic* labels. Kilosort guesses "
    "`good` or `mua` based on contamination metrics, but it has no idea "
    "whether the cluster is biologically real. That is what Phy is for.",
)


# =========================================================================
# Section 3 — Phy
# =========================================================================
md(
    "## 3 Background: Phy",
    "",
    "Phy (*Rossant et al. 2016*) is the interactive GUI most labs use to "
    "*curate* Kilosort output. A human looks at each cluster's waveform "
    "shape, autocorrelogram, amplitude drift over time, and spatial "
    "footprint, and decides whether the cluster is a real single unit "
    "(`good`), multi-unit noise (`mua`), an artefact (`noise`), or "
    "should be merged or split with a neighbour.",
    "",
    "Phy reads the same Kilosort directory and edits a few files in "
    "place:",
    "",
    "| File | What Phy does to it |",
    "|---|---|",
    "| `spike_clusters.npy` | Rewrites cluster IDs after merges and splits |",
    "| `cluster_info.tsv` | **New file** Phy writes. Contains the curated `group` column (`good`/`mua`/`noise`/`unsorted`) along with `Amplitude`, `ContamPct`, `depth`, `ch` (peak channel), `fr` (firing rate), `n_spikes`, and a few more |",
    "| `cluster_group.tsv` | A minimal alternative to `cluster_info.tsv` containing only `cluster_id, group` |",
    "",
    "So a *curated* Phy directory contains everything Kilosort wrote, "
    "**plus** `cluster_info.tsv`, with the human-corrected quality labels "
    "in its `group` column. That `group` column is what downstream "
    "analyses should filter on — `KSLabel` is the automatic guess, "
    "`group` is the curated truth.",
    "",
    "### Why two loaders",
    "",
    "`neurocomplexity` ships `from_phy` and `from_kilosort` as two "
    "separate functions to make the choice of quality source explicit. "
    "They share most of their code internally; the only difference is "
    "whether labels come from `cluster_info.tsv` (Phy, curated) or "
    "`cluster_KSLabel.tsv` (Kilosort, automatic). When in doubt: use "
    "`from_phy` if your directory has `cluster_info.tsv` — it almost "
    "certainly does, because the standard workflow is Kilosort → Phy. "
    "Use `from_kilosort` only if you have not yet run Phy curation.",
)


# =========================================================================
# Section 4 — SpikeInterface
# =========================================================================
md(
    "## 4 Background: SpikeInterface",
    "",
    "SpikeInterface (*Buccino et al. 2020*) is a Python framework that "
    "wraps every common spike sorter and every common raw-data format "
    "behind two abstract classes. `BaseRecording` is a unified view of "
    "any raw extracellular recording — Neuropixels, Open Ephys, "
    "Blackrock, Plexon, MEArec simulated data, NWB, and others. "
    "`BaseSorting` is a unified view of any sorted output — Kilosort "
    "1-4, MountainSort, IronClust, Tridesclous, manual labels, and "
    "others.",
    "",
    "If your data lives in a format `neurocomplexity` does not natively "
    "support — Open Ephys or Plexon, say — the recommended path is: read "
    "it with SpikeInterface, get a `BaseSorting`, then pass that to "
    "`nc.io.from_spikeinterface(sorting)`.",
    "",
    "Because SpikeInterface is a heavyweight dependency (it pulls in "
    "Neo, probeinterface, MEArec, ...), it is an optional install:",
    "",
    "```bash",
    "pip install 'neurocomplexity[spikeinterface]'",
    "```",
    "",
    "Without it, calling `nc.io.from_spikeinterface(...)` raises a clear "
    "`ImportError` telling you exactly that command.",
)


# =========================================================================
# Section 5 — Setup
# =========================================================================
md(
    "## 5 Setup",
    "",
    "Install the package, then import it. We use NumPy for arrays and "
    "pandas for the per-unit metadata tables.",
    "",
    "```bash",
    "pip install neurocomplexity                      # core",
    "pip install 'neurocomplexity[nwb]'               # adds NWB support",
    "pip install 'neurocomplexity[spikeinterface]'    # adds SpikeInterface bridge",
    "pip install 'neurocomplexity[viz]'               # adds matplotlib for figures",
    "```",
)

code(
    "import warnings",
    "import numpy as np",
    "import pandas as pd",
    "",
    "import neurocomplexity as nc",
    "",
    "print('neurocomplexity', nc.__version__)",
    "print('numpy', np.__version__)",
    "print('pandas', pd.__version__)",
)


# =========================================================================
# Section 6 — The SpikeRecording contract
# =========================================================================
md(
    "## 6 The SpikeRecording data model",
    "",
    "Everything `neurocomplexity` does eventually consumes one object: "
    "`SpikeRecording`. It is a frozen (immutable) dataclass with the "
    "following fields:",
    "",
    "| Field | Type | Meaning |",
    "|---|---|---|",
    "| `spike_times` | float64 array (seconds) | Monotonically non-decreasing |",
    "| `unit_ids` | int64 array (same length as `spike_times`) | Per-spike owner |",
    "| `units` | pandas DataFrame | Per-unit metadata, must have an `id` column |",
    "| `populations` | dict[str, bool array] | Named subsets of units |",
    "| `duration` | positive float (seconds) | Total recording length |",
    "| `sampling_rate` | float \\| None | When known, Hz |",
    "| `source` | `ProvenanceRecord` | Where this recording came from |",
    "| `intervals` | dict[str, DataFrame] | Stimulus or epoch tables |",
    "",
    "Invariants are checked at construction. If you build a recording "
    "with mismatched array shapes, negative spike times, or a population "
    "mask of the wrong length, you get a `RecordingValidationError` at "
    "the point of construction, never silently downstream. This is the "
    "single most important design choice in the package: a "
    "`SpikeRecording`, once constructed, is always valid.",
)


# =========================================================================
# Section 7 — Building a SpikeRecording by hand (from_dict)
# =========================================================================
md(
    "## 7 Build a SpikeRecording by hand",
    "",
    "The simplest loader, `from_dict`, takes a "
    "`{unit_id: spike_times_in_seconds}` mapping plus a duration. We use "
    "it throughout this notebook to construct toy recordings whose "
    "structure we know exactly.",
)

code(
    "rng = np.random.default_rng(0)",
    "",
    "# Three units firing as homogeneous Poisson over 60 s.",
    "spike_trains = {",
    "    0: np.sort(rng.uniform(0, 60, size=300)),  # ~5 Hz",
    "    1: np.sort(rng.uniform(0, 60, size=600)),  # ~10 Hz",
    "    2: np.sort(rng.uniform(0, 60, size=120)),  # ~2 Hz",
    "}",
    "",
    "rec = nc.io.from_dict(spike_trains, duration=60.0)",
    "rec",
)

md(
    "The repr summarises the recording. The fields are accessible "
    "directly:",
)

code(
    "print(f'units: {rec.n_units}')",
    "print(f'spikes: {rec.n_spikes}')",
    "print(f'duration: {rec.duration:.1f} s')",
    "print(f'mean firing rate: {rec.n_spikes / rec.duration / rec.n_units:.2f} Hz/unit')",
    "print()",
    "print('units metadata:')",
    "print(rec.units)",
    "print()",
    "print('populations:', list(rec.populations))",
)


# =========================================================================
# Section 8 — Build a synthetic Phy directory and load it
# =========================================================================
md(
    "## 8 Load a Phy curation directory",
    "",
    "In real life you would point `from_phy` at the output directory of "
    "your spike-sorting pipeline. To make this notebook self-contained, "
    "we *write* a small synthetic Phy directory using the same helper our "
    "test suite uses, then load it back. The helper is a faithful copy of "
    "what's in `tests/_sorter_fixtures.py`:",
)

code(
    "from pathlib import Path",
    "import tempfile",
    "",
    "DEFAULT_SAMPLE_RATE = 30000.0",
    "",
    "def write_sorter_directory(",
    "    directory,",
    "    spike_trains_sec,",
    "    *,",
    "    sample_rate=DEFAULT_SAMPLE_RATE,",
    "    cluster_info=None,",
    "    cluster_group=None,",
    "    cluster_kslabel=None,",
    "):",
    "    \"\"\"Write a Phy/Kilosort-format directory at `directory`.\"\"\"",
    "    directory = Path(directory)",
    "    directory.mkdir(parents=True, exist_ok=True)",
    "    sample_chunks, cluster_chunks = [], []",
    "    for cid, st in spike_trains_sec.items():",
    "        samples = np.round(np.asarray(st, dtype=np.float64) * sample_rate).astype(np.int64)",
    "        sample_chunks.append(samples)",
    "        cluster_chunks.append(np.full(samples.shape, int(cid), dtype=np.int32))",
    "    all_samples = np.concatenate(sample_chunks)",
    "    all_clusters = np.concatenate(cluster_chunks)",
    "    order = np.argsort(all_samples, kind='stable')",
    "    np.save(directory / 'spike_times.npy', all_samples[order])",
    "    np.save(directory / 'spike_clusters.npy', all_clusters[order])",
    "    np.save(directory / 'spike_templates.npy', all_clusters[order])",
    "    (directory / 'params.py').write_text(",
    "        f\"sample_rate = {sample_rate}\\n\"",
    "        \"dat_path = 'continuous.dat'\\n\"",
    "        \"n_channels_dat = 384\\n\"",
    "        \"dtype = 'int16'\\n\"",
    "        \"offset = 0\\n\"",
    "        \"hp_filtered = True\\n\",",
    "        encoding='utf-8',",
    "    )",
    "    if cluster_info is not None:",
    "        cluster_info.to_csv(directory / 'cluster_info.tsv', sep='\\t', index=False)",
    "    if cluster_group is not None:",
    "        cluster_group.to_csv(directory / 'cluster_group.tsv', sep='\\t', index=False)",
    "    if cluster_kslabel is not None:",
    "        cluster_kslabel.to_csv(directory / 'cluster_KSLabel.tsv', sep='\\t', index=False)",
    "    return directory",
)

md(
    "Now build a synthetic Phy directory with five units — three labelled "
    "`good`, one `mua`, one `noise` — and inspect what files ended up on "
    "disk:",
)

code(
    "phy_dir = Path(tempfile.mkdtemp(prefix='nc_phy_')) / 'phy_output'",
    "",
    "trains = {",
    "    10: np.sort(rng.uniform(0, 30, size=400)),",
    "    11: np.sort(rng.uniform(0, 30, size=350)),",
    "    12: np.sort(rng.uniform(0, 30, size=300)),",
    "    13: np.sort(rng.uniform(0, 30, size=600)),  # will be 'mua'",
    "    14: np.sort(rng.uniform(0, 30, size=900)),  # will be 'noise'",
    "}",
    "",
    "# Realistic Phy cluster_info.tsv columns.",
    "cluster_info = pd.DataFrame({",
    "    'cluster_id': [10, 11, 12, 13, 14],",
    "    'group':      ['good', 'good', 'good', 'mua', 'noise'],",
    "    'KSLabel':    ['good', 'good', 'good', 'good', 'mua'],",
    "    'Amplitude':  [78.4, 65.1, 90.2, 30.7, 18.5],",
    "    'ContamPct':  [0.5, 1.2, 0.3, 8.7, 22.1],",
    "    'depth':      [120.0, 220.0, 380.0, 460.0, 530.0],",
    "    'ch':         [12, 28, 51, 64, 73],",
    "    'fr':         [13.3, 11.7, 10.0, 20.0, 30.0],",
    "    'n_spikes':   [400, 350, 300, 600, 900],",
    "})",
    "",
    "write_sorter_directory(phy_dir, trains, cluster_info=cluster_info)",
    "",
    "print('Files written:')",
    "for f in sorted(phy_dir.iterdir()):",
    "    print(f'  {f.name}  ({f.stat().st_size} bytes)')",
)

md("Now load it:")

code(
    "rec_phy = nc.io.from_phy(phy_dir)",
    "print(rec_phy)",
    "print()",
    "print('units table after normalisation:')",
    "print(rec_phy.units)",
)

md(
    "The loader normalises column names: `cluster_id → id`, "
    "`group → quality`, `fr → firing_rate`, `ch → peak_channel`, "
    "`Amplitude → amplitude`, `ContamPct → contam_pct`. All other "
    "columns pass through verbatim. The point is that downstream code "
    "always sees the same column names regardless of how the upstream "
    "file labelled them.",
    "",
    "Drop the non-good units with `filter_units`:",
)

code(
    "rec_good = rec_phy.filter_units(quality=['good'])",
    "print(f'{rec_good.n_units} units after quality filter '",
    "      f'(was {rec_phy.n_units})')",
)


# =========================================================================
# Section 9 — Kilosort directory
# =========================================================================
md(
    "## 9 Load a raw Kilosort directory",
    "",
    "If you have not yet run Phy curation, your directory has "
    "`cluster_KSLabel.tsv` instead of `cluster_info.tsv`. Use "
    "`from_kilosort` in that case:",
)

code(
    "ks_dir = Path(tempfile.mkdtemp(prefix='nc_ks_')) / 'ks_output'",
    "ks_trains = {",
    "    1: np.sort(rng.uniform(0, 20, size=300)),",
    "    2: np.sort(rng.uniform(0, 20, size=500)),",
    "    3: np.sort(rng.uniform(0, 20, size=80)),",
    "}",
    "ks_label = pd.DataFrame({",
    "    'cluster_id': [1, 2, 3],",
    "    'KSLabel':    ['good', 'good', 'mua'],",
    "})",
    "write_sorter_directory(ks_dir, ks_trains, cluster_kslabel=ks_label)",
    "",
    "rec_ks = nc.io.from_kilosort(ks_dir)",
    "print(rec_ks)",
    "print()",
    "print(rec_ks.units)",
)


# =========================================================================
# Section 10 — SpikeInterface bridge
# =========================================================================
md(
    "## 10 SpikeInterface bridge (optional)",
    "",
    "If you have SpikeInterface installed, the bridge looks like this. We "
    "try the import; if the extra is not installed, we skip the cell "
    "rather than crashing.",
)

code(
    "try:",
    "    import spikeinterface  # noqa: F401",
    "    from spikeinterface.core import NumpySorting",
    "    have_si = True",
    "except ImportError:",
    "    have_si = False",
    "    print('SpikeInterface not installed; skipping. Install with: '",
    "          'pip install \"neurocomplexity[spikeinterface]\"')",
    "",
    "if have_si:",
    "    sample_rate = 30000.0",
    "    si_trains = {",
    "        0: (np.array([0.1, 0.5, 1.2]) * sample_rate).astype(np.int64),",
    "        1: (np.array([0.2, 0.9]) * sample_rate).astype(np.int64),",
    "    }",
    "    sorting = NumpySorting.from_unit_dict([si_trains], sampling_frequency=sample_rate)",
    "    rec_si = nc.io.from_spikeinterface(sorting)",
    "    print(rec_si)",
)


# =========================================================================
# Section 11 — NWB
# =========================================================================
md(
    "## 11 Load an NWB file",
    "",
    "Neurodata Without Borders (NWB) is the community standard archive "
    "format. The Allen Brain Observatory Neuropixels sessions are "
    "distributed as `.nwb` files. To run this section you need the NWB "
    "extra:",
    "",
    "```bash",
    "pip install 'neurocomplexity[nwb]'",
    "```",
    "",
    "Then download a session — see the Allen SDK documentation — and "
    "load it with one line:",
    "",
    "```python",
    "rec_nwb = nc.io.from_nwb('path/to/ecephys_session_715093703.nwb')",
    "print(rec_nwb)",
    "```",
    "",
    "Because the file can be many gigabytes, this notebook does not "
    "execute the load; we proceed with synthetic data below.",
)


# =========================================================================
# Section 12 — filter_units, populations, crop
# =========================================================================
md(
    "## 12 Filtering, populations, cropping",
    "",
    "Every transformation on a `SpikeRecording` returns a new "
    "`SpikeRecording`. The original is never mutated. That makes "
    "exploratory analysis safe: you can chain operations and always go "
    "back to the original recording. The three workhorse transformations "
    "are `filter_units` (subset by metadata predicates), "
    "`with_populations` (group units into named sub-populations from a "
    "metadata column), and `crop` (restrict to a time window).",
    "",
    "We pin some metadata onto the recording from §7 to make filtering "
    "interesting:",
)

code(
    "demo = nc.io.from_dict(spike_trains, duration=60.0)",
    "demo.units['quality'] = ['good', 'good', 'mua']",
    "demo.units['brain_area'] = ['V1', 'V1', 'LM']",
    "print(demo.units)",
)

code(
    "# Drop the 'mua' unit.",
    "demo_clean = demo.filter_units(quality=['good'])",
    "print(f'{demo_clean.n_units} units after quality filter '",
    "      f'(was {demo.n_units})')",
)

code(
    "# Define populations from a metadata column.",
    "demo_pops = demo.with_populations(by='brain_area')",
    "print('populations:',",
    "      {k: int(v.sum()) for k, v in demo_pops.populations.items()})",
)

code(
    "# Restrict to a time window.",
    "demo_window = demo.crop(10.0, 30.0)",
    "print(f'cropped {demo.duration:.1f}s -> {demo_window.duration:.1f}s; '",
    "      f'{demo.n_spikes} -> {demo_window.n_spikes} spikes')",
)


# =========================================================================
# Section 13 — Provenance
# =========================================================================
md(
    "## 13 Provenance",
    "",
    "Every `SpikeRecording` carries an immutable `ProvenanceRecord` that "
    "fingerprints where it came from: loader name, source path, a BLAKE2b "
    "hash of the input file(s), the package version that did the load, "
    "and the loader's keyword arguments. This matters for "
    "reproducibility — you can hash a recording against the original file "
    "weeks later and verify that nothing has shifted underneath you, and "
    "the version string lets you tell whether a result was generated by "
    "v1.0 or v1.1 of the package.",
)

code(
    "print(rec_phy.source)",
)


# =========================================================================
# Section 14 — The analyses overview
# =========================================================================
md(
    "## 14 Analyses — the contract",
    "",
    "The package exposes seven analyses, each in its own module under "
    "`neurocomplexity.analysis`, and each returning its own frozen "
    "`*Result` dataclass:",
    "",
    "| Function | What it returns | What it measures |",
    "|---|---|---|",
    "| `wilting_mr` | `BranchingResult` | Sub-sampling-robust branching ratio $\\hat{m}$ |",
    "| `criticality` | `CriticalityResult` | Avalanche size exponent $\\alpha$, duration exponent $\\tau$, $\\kappa$ |",
    "| `shape_collapse` | `ShapeCollapseResult` | Crackling-noise rescaling exponent $\\gamma$ |",
    "| `transfer_entropy` | `TransferEntropyResult` | Directed $\\mathrm{TE}_{X \\to Y}$ matrix across populations |",
    "| `partial_information` | `PIDResult` | Williams-Beer redundancy / unique / synergy decomposition |",
    "| `autonomy` | `AutonomyResult` | VAR-Granger autonomy $p$-value per population |",
    "| `dimensionality` | `DimensionalityResult` | Participation-ratio dimensionality |",
    "",
    "All seven share the same calling convention:",
    "",
    "```python",
    "result = nc.analysis.foo(",
    "    rec,                    # SpikeRecording",
    "    populations=['all'],    # which named populations to use",
    "    bin_size_ms=4,          # discretisation timescale",
    "    ...,                    # estimator-specific arguments",
    ")",
    "```",
    "",
    "Each result dataclass exposes the fitted scalar plus the diagnostic "
    "by-products needed to plot or audit the fit (lag curves, eigenvalue "
    "spectra, avalanche-shape arrays, and so on).",
)


# =========================================================================
# Section 15 — Branching ratio
# =========================================================================
md(
    "## 15 Branching ratio (Wilting-Priesemann)",
    "",
    "The branching ratio $m$ is the average number of new spikes "
    "triggered by each existing spike one bin later. $m \\approx 1$ means "
    "the network sits at the boundary between stable and explosive "
    "activity (critical); $m < 1$ means activity decays (sub-critical); "
    "$m > 1$ means it grows without bound (super-critical). Cortex "
    "appears to live slightly sub-critical.",
    "",
    "The catch is that you only ever record from a tiny fraction of the "
    "local network. Classical avalanche-based estimators are biased "
    "downwards under that kind of sub-sampling. The Wilting-Priesemann "
    "*multi-step regression* estimator (Wilting & Priesemann, 2018) "
    "instead fits the slope of "
    "$r_k = \\mathrm{Cov}(A_{t+k}, A_t) / \\mathrm{Var}(A_t)$ across "
    "lags $k$ in log-space, which is robust to sub-sampling.",
    "",
    "We use the package's branching-network simulator to build a "
    "recording with a known true $m$, then ask the estimator to recover "
    "it:",
)

code(
    "from neurocomplexity.benchmarks.simulators.branching_network import branching_network",
    "from neurocomplexity.analysis import wilting_mr",
    "",
    "rec_bn = branching_network(",
    "    n_units=60, m=0.95, duration_s=120.0, bin_ms=4.0, seed=0,",
    ")",
    "",
    "result = wilting_mr(rec_bn, populations=['all'], bin_size_ms=4.0)",
    "print(f'true m = 0.95, estimated m = {result.m:.3f}')",
    "print(f'fit R^2 = {result.r_squared:.3f}')",
)

md(
    "Expect $\\hat{m}$ within about 0.03 of 0.95 on this single run; the "
    "estimator's error scales with $(1-m)$, so it is tightest near "
    "criticality and noisier in the sub-critical regime. The fit $R^2$ "
    "should be high (> 0.95) — the multi-step regression is a clean "
    "log-linear fit when the underlying process is genuinely near-"
    "critical.",
)


# =========================================================================
# Section 16 — Avalanche exponents
# =========================================================================
md(
    "## 16 Avalanche exponents (criticality)",
    "",
    "A neuronal *avalanche* is a contiguous stretch of bins with at "
    "least one spike, bracketed by empty bins. Critical systems produce "
    "avalanches whose **size** distribution is a power law "
    "$P(s) \\propto s^{-\\alpha}$ with $\\alpha \\approx 1.5$ and whose "
    "**duration** distribution is $P(T) \\propto T^{-\\tau}$ with "
    "$\\tau \\approx 2.0$. These are the mean-field exponents of a "
    "critical branching process and are what experimental papers report "
    "when they argue a recording is critical.",
    "",
    "The naive fit on a linear-bin histogram is upward-biased on "
    "heavy-tailed data because most of the bins live in the tail and are "
    "empty. `neurocomplexity` uses log-binned histograms with "
    "geometric-mean centres instead, which removes the bias. To get a "
    "clean recovery we use the trial-based critical branching simulator "
    "(many independent avalanches seeded from a single spike, propagated "
    "at $m=1$ to extinction):",
)

code(
    "from neurocomplexity.benchmarks.simulators.branching_network import trial_based_avalanches",
    "from neurocomplexity.analysis import criticality",
    "",
    "rec_av = trial_based_avalanches(",
    "    n_units=40, n_trials=3000, m=1.0, bin_ms=4.0, seed=0,",
    ")",
    "",
    "crit = criticality(rec_av, populations=['all'], bin_size_ms=(4.0,))",
    "print(f'alpha (size)     = {crit.alpha_s:.2f}   (expected ~1.5)')",
    "print(f'alpha_t (gamma)  = {crit.alpha_t:.2f}   (gamma scaling exponent)')",
    "print(f'kappa (collapse) = {crit.kappa:.2f}   (~1 at criticality)')",
    "print(f'optimal bin (s)  = {crit.optimal_bin_seconds:.3f}')",
)

md(
    "$\\hat{\\alpha}$ should land within roughly 0.10 of 1.5 at this trial "
    "count. The `alpha_t` field is the $\\gamma$-scaling exponent fit "
    "from $\\langle s \\rangle$ versus $T$, not $\\tau$ directly; the "
    "package fits the duration exponent $\\tau$ via "
    "`fit_alpha(lifetimes_in_bins, xmin=1)` if you want the noisier "
    "tail-based estimate. $\\kappa$ is the Shew et al. (2009) collapse "
    "score — 1.0 at perfect criticality, lower as the system drifts "
    "away — and is the most diagnostic single number if you are trying "
    "to *classify* a recording as critical or not.",
)


# =========================================================================
# Section 17 — Shape collapse
# =========================================================================
md(
    "## 17 Shape collapse",
    "",
    "Beyond the exponents themselves, critical systems exhibit "
    "*universal scaling*: avalanches of different durations $T$, "
    "rescaled by $u = t/T$ and "
    "$\\langle a \\rangle / T^{\\gamma-1}$, collapse onto a single "
    "universal shape. The rescaling exponent $\\gamma$ obeys the "
    "**crackling-noise relation** $\\gamma = (\\tau - 1) / (\\alpha - 1)$ "
    "if the system is genuinely critical (Sethna et al. 2001; Friedman "
    "et al. 2012). The fit $\\gamma$ and the crackling-noise prediction "
    "are independent ways of estimating the same quantity, and their "
    "agreement is one of the most stringent tests of criticality.",
    "",
    "On the same `rec_av` recording from §16:",
)

code(
    "from neurocomplexity.analysis import shape_collapse",
    "",
    "sc = shape_collapse(rec_av, populations=['all'], bin_size_ms=4.0)",
    "# Crackling-noise prediction uses the duration exponent tau; we fit",
    "# it from the lifetime distribution directly:",
    "from neurocomplexity.analysis.criticality import fit_alpha",
    "lifetimes_in_bins = crit.lifetimes / crit.optimal_bin_seconds",
    "tau = fit_alpha(lifetimes_in_bins, xmin=1)",
    "predicted_gamma = (tau - 1) / (crit.alpha_s - 1)",
    "print(f'fit gamma        = {sc.gamma:.2f}')",
    "print(f'tau (duration)   = {tau:.2f}')",
    "print(f'predicted gamma  = {predicted_gamma:.2f}  (from (tau-1)/(alpha-1))')",
    "print(f'collapse residual= {sc.residual:.3g}')",
)

md(
    "Fitted and predicted $\\gamma$ should agree to within roughly 0.1 if "
    "the recording is critical. Disagreement is a useful warning sign "
    "that the system might not be in the critical universality class, "
    "even if the individual exponents look right.",
)


# =========================================================================
# Section 18 — Transfer entropy
# =========================================================================
md(
    "## 18 Transfer entropy",
    "",
    "$\\mathrm{TE}_{X \\to Y}$ is the amount of uncertainty about $Y$'s "
    "next state that is resolved by knowing $X$'s past, *over and above* "
    "what $Y$'s own past already tells you. It is directional — "
    "$\\mathrm{TE}_{X \\to Y}$ is not in general equal to "
    "$\\mathrm{TE}_{Y \\to X}$ — and model-free (Schreiber 2000). The "
    "package uses a binary-symbol histogram estimator with Miller-Madow "
    "bias correction, which is the right default for binned spike "
    "trains.",
    "",
    "We test it on two coupled AR(1) processes with known coupling "
    "strength $c$, thinned through a Poisson rate code into multi-unit "
    "spike populations. The larger $c$, the larger the true TE; the "
    "operationally useful check is that the estimator preserves that "
    "ordering after the spike-encoding loss.",
)

code(
    "from neurocomplexity.benchmarks.simulators.ar_processes import coupled_ar1",
    "from neurocomplexity.benchmarks.cases.info_theory import _ar_to_recording",
    "from neurocomplexity.analysis import transfer_entropy",
    "",
    "for c in [0.0, 0.3, 0.6]:",
    "    x, y, te_true = coupled_ar1(",
    "        c=c, a=0.5, sigma=1.0, n_samples=10_000, seed=0,",
    "    )",
    "    rec_te = _ar_to_recording(",
    "        x, y, base_rate_hz=80.0, modulation=0.9,",
    "        units_per_pop=8, seed=0,",
    "    )",
    "    te_res = transfer_entropy(",
    "        rec_te, populations=['X', 'Y'],",
    "        bin_size_ms=10.0, delay_bins=1,",
    "    )",
    "    te_est = float(te_res.matrix[0, 1])  # X -> Y",
    "    print(f'c = {c:.2f}   true TE = {te_true:.4f}   est TE(X->Y) = {te_est:.4f} nats')",
)

md(
    "Estimated TE grows monotonically with $c$. The absolute scale is "
    "compressed relative to the analytic VAR-process TE because the "
    "Poisson thinning discards some of the linear-Gaussian information, "
    "but the rank ordering is preserved — and that's the property the "
    "package's `info_theory.te_convergence` benchmark validates "
    "quantitatively (Spearman $\\rho \\geq 0.85$ across coupling levels).",
)


# =========================================================================
# Section 19 — PID
# =========================================================================
md(
    "## 19 Partial information decomposition (Williams-Beer)",
    "",
    "Given two source populations $X_1, X_2$ and a target $Y$, PID "
    "splits the joint mutual information $I(Y; X_1, X_2)$ into four "
    "non-negative parts: **redundancy** (what both sources tell you about "
    "$Y$ on their own), **unique-to-$X_1$**, **unique-to-$X_2$**, and "
    "**synergy** (what you only learn by knowing both sources together — "
    "XOR is the canonical example). The package implements the "
    "Williams-Beer $I_{\\min}$ measure (Williams & Beer 2010).",
    "",
    "Five canonical distributions are shipped as simulators so you can "
    "verify the decomposition against ground truth. We loop over three "
    "of them:",
)

code(
    "from neurocomplexity.benchmarks.simulators.pid_distributions import pid_recording",
    "from neurocomplexity.analysis import partial_information",
    "",
    "for name, expected in [",
    "    ('xor',  'pure synergy ~ ln 2'),",
    "    ('and',  'mixed; small R, small S'),",
    "    ('copy', 'pure unique to X1 ~ ln 2'),",
    "]:",
    "    rec_pid = pid_recording(name, n_bins=20_000, bin_ms=10.0, seed=0)",
    "    res = partial_information(",
    "        rec_pid,",
    "        target_pop='target',",
    "        sources=['source_1', 'source_2'],",
    "        bin_size_ms=10.0,",
    "        n_levels=2,",
    "    )",
    "    print(",
    "        f'{name.upper():>4s}:  R={res.redundancy:.3f}  '",
    "        f'U1={res.unique_1:.3f}  U2={res.unique_2:.3f}  S={res.synergy:.3f}    [{expected}]'",
    "    )",
)

md(
    "Expect the XOR row to put almost all the joint MI into synergy, COPY "
    "to put almost all of it into U1, and AND to split a small amount "
    "across redundancy and synergy. If any atom comes out clearly "
    "negative the bias correction has failed — that is the failure mode "
    "to watch for at small `n_bins`.",
)


# =========================================================================
# Section 20 — Autonomy
# =========================================================================
md(
    "## 20 VAR-Granger autonomy",
    "",
    "The autonomy index asks the opposite of TE: how much of the "
    "target's future is *not* predictable from any other population in "
    "the recording. Concretely, `autonomy()` returns, for each "
    "population, the $p$-value of an F-test asking whether external "
    "populations improve the VAR(p) prediction of that population's next "
    "state beyond its own history. **Large $p$** means the population is "
    "statistically autonomous; **small $p$** means external populations "
    "carry information about it.",
)

code(
    "from neurocomplexity.benchmarks.simulators.ar_processes import var1",
    "from neurocomplexity.analysis import autonomy",
    "",
    "# Null: two independent VAR(1) populations -> high p.",
    "A_null = np.array([[0.5, 0.0], [0.0, 0.5]])",
    "Sigma  = np.eye(2)",
    "X0 = var1(A=A_null, Sigma=Sigma, n_samples=2000, seed=0)",
    "rec_null = _ar_to_recording(X0[:, 0], X0[:, 1],",
    "                            base_rate_hz=80.0, modulation=0.9,",
    "                            units_per_pop=8, seed=0)",
    "a_null = autonomy(rec_null, populations=['X', 'Y'], bin_size_ms=10.0)",
    "",
    "# Coupled: Y feeds back into X -> low p for X.",
    "A_cpl = np.array([[0.5, 0.0], [0.3, 0.5]])",
    "X1 = var1(A=A_cpl, Sigma=Sigma, n_samples=2000, seed=0)",
    "rec_cpl = _ar_to_recording(X1[:, 0], X1[:, 1],",
    "                           base_rate_hz=80.0, modulation=0.9,",
    "                           units_per_pop=8, seed=0)",
    "a_cpl = autonomy(rec_cpl, populations=['X', 'Y'], bin_size_ms=10.0)",
    "",
    "print('null   p-values:', {k: f'{v:.3f}' for k, v in a_null.values.items()})",
    "print('coupled p-values:', {k: f'{v:.3f}' for k, v in a_cpl.values.items()})",
)

md(
    "Under the null, both populations' $p$-values should be far above "
    "0.05 — they really are autonomous. Under the coupled VAR, at least "
    "one population should drop below 0.05, reflecting that knowing the "
    "other population's history does help predict its next state. This "
    "is the same statistic the `info_theory.autonomy_calibration` "
    "benchmark validates at Type-I rate $\\leq 0.25$ and power "
    "$\\geq 0.80$ at $c = 0.3$.",
)


# =========================================================================
# Section 21 — Dimensionality
# =========================================================================
md(
    "## 21 Participation-ratio dimensionality",
    "",
    "Given the $N \\times N$ pairwise spike-count correlation matrix, its "
    "eigenvalue spectrum tells you how many *effective* modes carry the "
    "population activity. The participation ratio "
    "$\\mathrm{PR} = (\\sum_i \\lambda_i)^2 / \\sum_i \\lambda_i^2$ is a "
    "single-number summary: $\\mathrm{PR} = 1$ if all units are "
    "perfectly correlated (one mode), $\\mathrm{PR} = N$ if every unit "
    "is independent. PR is preferred over a hard PCA-variance cutoff "
    "because it is differentiable in the eigenvalues and does not "
    "discard information about smaller modes.",
    "",
    "The package ships a rank-$r$ structured-covariance simulator we can "
    "use to verify recovery:",
)

code(
    "from neurocomplexity.benchmarks.simulators.structured_covariance import rank_r_population",
    "from neurocomplexity.analysis import dimensionality",
    "",
    "rec_dim = rank_r_population(",
    "    n_units=50, rank=5, n_bins=6000, bin_ms=10.0, noise=0.05, seed=0,",
    ")",
    "dim = dimensionality(rec_dim, populations=['all'], bin_size_ms=10.0)",
    "print(f'N units = {dim.n_units},  PR = {dim.participation_ratio:.2f}  (true rank = 5)')",
    "print(f'top eigenvalues: {dim.eigenvalues[:8].round(3)}')",
)

md(
    "Expect PR to land within a fraction of a unit of the true rank when "
    "the noise floor is low. As you turn `noise` up, PR drifts toward "
    "$N$ because more of the variance ends up in small but non-zero "
    "tail eigenvalues — exactly the behaviour you want from a soft "
    "dimensionality count.",
)


# =========================================================================
# Section 22 — Surrogates
# =========================================================================
md(
    "## 22 Inference: surrogates",
    "",
    "Every point estimate above is just a number; without a null "
    "distribution you cannot say whether it is significant. "
    "`neurocomplexity.inference.surrogates` provides three null "
    "generators, each preserving a different feature of the data:",
    "",
    "| Surrogate | Preserves | Destroys |",
    "|---|---|---|",
    "| `spike_dither` | mean rate, fine ISI structure | precise timing within ±Δ |",
    "| `isi_shuffle` | mean rate, ISI distribution | spike sequence |",
    "| `interval_shuffle` | rate and bursts at a chosen timescale | long-range structure |",
    "",
    "The most useful sanity check is to recompute a statistic on a "
    "surrogate of the *same* recording and see how it shifts. For a "
    "directed measure like TE, ISI shuffling should drive it toward "
    "zero — the timing relationship between source and target is gone, "
    "even though each unit still fires at its original rate:",
)

code(
    "from neurocomplexity.inference.surrogates import spike_dither, isi_shuffle",
    "",
    "# Build a strongly-coupled recording.",
    "x, y, _ = coupled_ar1(c=0.6, a=0.5, sigma=1.0, n_samples=10_000, seed=0)",
    "rec_strong = _ar_to_recording(x, y, base_rate_hz=80.0, modulation=0.9,",
    "                              units_per_pop=8, seed=0)",
    "te_obs = transfer_entropy(rec_strong, populations=['X', 'Y'], bin_size_ms=10.0)",
    "print(f'observed TE(X->Y)        = {te_obs.matrix[0,1]:.4f} nats')",
    "",
    "# ISI shuffle: destroys timing but keeps rate and ISI distribution.",
    "rec_isi = isi_shuffle(rec_strong, seed=0)",
    "te_isi = transfer_entropy(rec_isi, populations=['X', 'Y'], bin_size_ms=10.0)",
    "print(f'TE(X->Y) on ISI surrogate = {te_isi.matrix[0,1]:.4f} nats')",
    "",
    "# Spike dither at 20 ms: preserves coarse timing, destroys fine.",
    "rec_dith = spike_dither(rec_strong, delta_ms=20.0, seed=0)",
    "te_dith = transfer_entropy(rec_dith, populations=['X', 'Y'], bin_size_ms=10.0)",
    "print(f'TE(X->Y) on dithered      = {te_dith.matrix[0,1]:.4f} nats')",
)

md(
    "The observed TE should be well above both surrogate values. The "
    "ISI-shuffle TE is the strongest null (timing fully destroyed); the "
    "dithered TE sits somewhere in between, because a ±20 ms jitter "
    "blurs precise spike timing but leaves the broad rate envelope "
    "alone. Picking the right surrogate is a substantive choice — it "
    "defines what 'no effect' means for your hypothesis.",
)


# =========================================================================
# Section 23 — Bootstrap CIs
# =========================================================================
md(
    "## 23 Inference: bootstrap confidence intervals",
    "",
    "Block bootstraps resample the recording in contiguous time blocks "
    "(default 5 s), recompute the estimator on each resample, and report "
    "a bias-corrected (BC) percentile interval (Efron 1987, $z_0$ "
    "correction only, no acceleration). Block resampling is essential "
    "for spike data because i.i.d. resampling of individual spikes would "
    "destroy local rate structure. Use this whenever you report a point "
    "estimate.",
)

code(
    "from neurocomplexity.inference import bootstrap",
    "",
    "rec_bn_small = branching_network(",
    "    n_units=60, m=0.95, duration_s=60.0, bin_ms=4.0, seed=0,",
    ")",
    "m_result = wilting_mr(rec_bn_small, populations=['all'], bin_size_ms=4.0)",
    "ci = bootstrap(m_result, rec_bn_small, n=200, block_seconds=5.0, seed=0)",
    "print(f'm_hat = {m_result.m:.3f}    95% CI = [{ci.ci_lower:.3f}, {ci.ci_upper:.3f}]')",
)


# =========================================================================
# Section 24 — Null tests + FDR
# =========================================================================
md(
    "## 24 Inference: null tests and FDR",
    "",
    "For TE and similar pairwise measures the right question is: is "
    "$\\mathrm{TE}(X \\to Y)$ significantly larger than what surrogates "
    "produce? The `test()` function answers that with a Phipson-Smyth-"
    "corrected permutation $p$-value (never exactly zero, even for "
    "perfect separation) and a Benjamini-Hochberg FDR adjustment across "
    "the off-diagonal of the TE matrix.",
)

code(
    "from neurocomplexity.inference import test",
    "",
    "x, y, _ = coupled_ar1(c=0.5, a=0.5, sigma=1.0, n_samples=10_000, seed=0)",
    "rec_te = _ar_to_recording(x, y, base_rate_hz=80.0, modulation=0.9,",
    "                          units_per_pop=8, seed=0)",
    "te = transfer_entropy(rec_te, populations=['X', 'Y'], bin_size_ms=10.0)",
    "inf = test(te, rec_te, surrogate='isi_shuffle', n=100, seed=0, fdr=True)",
    "print(f'observed TE matrix:\\n{np.asarray(te.matrix).round(4)}')",
    "print(f'p-values:\\n{np.asarray(inf.p_value).round(4)}')",
    "print(f'p_FDR:\\n{np.asarray(inf.p_value_fdr).round(4)}')",
)

md(
    "Only the off-diagonal entries are meaningful — a population's TE to "
    "itself is degenerate. With $c = 0.5$ you should see "
    "$X \\to Y$ rejected (small $p$) and $Y \\to X$ retained (large $p$), "
    "consistent with the unidirectional coupling.",
)


# =========================================================================
# Section 25 — Putting it together
# =========================================================================
md(
    "## 25 End-to-end mini-example",
    "",
    "Tie the workflow together on a single synthetic recording: load → "
    "filter → populations → analyse → bootstrap.",
)

code(
    "# 1. Load (here, build by hand for self-containment).",
    "all_trains = {",
    "    i: np.sort(rng.uniform(0, 120, size=300 + 100 * i))",
    "    for i in range(20)",
    "}",
    "rec_e2e = nc.io.from_dict(all_trains, duration=120.0)",
    "rec_e2e.units['quality']    = ['good'] * 20",
    "rec_e2e.units['brain_area'] = ['V1'] * 10 + ['LM'] * 10",
    "",
    "# 2. Filter to good units.",
    "rec_e2e = rec_e2e.filter_units(quality=['good'])",
    "",
    "# 3. Populations from metadata.",
    "rec_e2e = rec_e2e.with_populations(by='brain_area')",
    "",
    "# 4. Branching ratio per population, with bootstrap CIs.",
    "for pop_name in ['V1', 'LM']:",
    "    m_res = wilting_mr(rec_e2e, populations=[pop_name], bin_size_ms=4.0)",
    "    ci = bootstrap(m_res, rec_e2e, n=50, block_seconds=5.0, seed=0)",
    "    print(f'{pop_name:>3s}:  m = {m_res.m:.3f}   '",
    "          f'95% CI = [{ci.ci_lower:.3f}, {ci.ci_upper:.3f}]')",
)


# =========================================================================
# Section 26 — Benchmarks
# =========================================================================
md(
    "## 26 Validating your installation",
    "",
    "Every release of `neurocomplexity` is accompanied by a benchmark "
    "CSV at `results/benchmarks/v<version>.csv`. You can regenerate it "
    "on your own machine and confirm that every estimator recovers its "
    "ground truth within tolerance:",
    "",
    "```bash",
    "python -m neurocomplexity benchmark --reps 50",
    "```",
    "",
    "or in Python:",
)

code(
    "from neurocomplexity.benchmarks import run_all, list_cases",
    "",
    "print('Registered cases:')",
    "for c in list_cases():",
    "    print(f'  {c}')",
    "",
    "# The full run takes a couple of hours at n_reps=200 (see the",
    "# v1.1.0 CSV runtimes). Inside a notebook we run just the cheap",
    "# PID atoms at small reps so the cell finishes in seconds:",
    "fast_cases = [c for c in list_cases() if c.startswith('pid.')]",
    "df = run_all(cases=fast_cases, n_reps=3, seed=0, verbose=False)",
    "print()",
    "print(df[['name', 'observed', 'tolerance', 'passed', 'runtime_s']].to_string(index=False))",
)


# =========================================================================
# Section 27 — Visualisation
# =========================================================================
md(
    "## 27 Visualisation",
    "",
    "If you installed the `[viz]` extra (`pip install "
    "'neurocomplexity[viz]'`), every analysis has a publication-ready "
    "figure function next to it. They return matplotlib `Figure` "
    "objects; the package's `viz._style` module applies Nature-style "
    "rcParams (Arial 7 pt, no top/right spines, editable SVG/PDF). The "
    "`save_publication` helper writes an SVG + PDF + 600 dpi TIFF "
    "triplet in one call, which is what most journals' submission "
    "systems actually want.",
    "",
    "```python",
    "import matplotlib.pyplot as plt",
    "from neurocomplexity.viz import (",
    "    figure_branching, figure_criticality, figure_shape_collapse,",
    "    figure_pid, figure_dimensionality, figure_overview,",
    "    save_publication,",
    ")",
    "",
    "fig = figure_branching(m_result)",
    "save_publication(fig, 'fig_branching')   # writes .svg, .pdf, .tif",
    "```",
    "",
    "`figure_overview` composes branching, criticality, shape collapse, "
    "dimensionality, and PID into one Nature-style double-column layout, "
    "which is what we use for Figure 1 of the methods paper.",
)


# =========================================================================
# Section 28 — Closing
# =========================================================================
md(
    "## 28 Where to go next",
    "",
    "- **Tutorial on real Neuropixels data:** see `tutorial/tutorial.ipynb` "
    "for the NWB-loading worked example on an Allen Brain Observatory "
    "session.",
    "- **API reference:** `docs/api/index.md`.",
    "- **Benchmarks:** `docs/benchmarks.md` documents every case and the "
    "literature reference whose claim it verifies.",
    "- **Issues and pull requests:** "
    "<https://github.com/aurmandi/neurocomplexity>.",
    "",
    "If anything in this tutorial felt confusing, that is more often a "
    "bug in the explanation than in your understanding — please open an "
    "issue.",
)


# =========================================================================
# Emit
# =========================================================================
nb.cells = cells
nbformat.write(nb, OUT_PATH)
print(f"wrote {OUT_PATH}  ({len(cells)} cells)")
