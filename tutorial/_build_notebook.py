"""Build tutorial/tutorial.ipynb from this script.

The notebook is generated rather than hand-edited so the markdown can stay
readable here and the JSON stays valid there. Run me with::

    py -3.12 tutorial/_build_notebook.py
"""
from pathlib import Path
import nbformat as nbf


def md(*lines):
    return nbf.v4.new_markdown_cell("\n".join(lines))


def code(*lines):
    return nbf.v4.new_code_cell("\n".join(lines))


cells = []

# ---------------------------------------------------------------------------
# 0. Title & overview
# ---------------------------------------------------------------------------
cells.append(md(
    "# neurocomplexity — tutorial",
    "",
    "This notebook walks through every public function in `neurocomplexity` "
    "using a real Neuropixels recording from the Allen Brain Observatory "
    "(session `715093703`). It assumes no prior familiarity with the NWB "
    "file format, with the Allen unit-quality conventions, or with the "
    "complex-systems statistics the package computes.",
    "",
    "By the end you will know how to:",
    "",
    "- open an NWB file and inspect what it contains;",
    "- apply unit-quality filtering and cell-type classification;",
    "- define populations and slice the recording to specific time spans;",
    "- run all five analyses in the package and interpret their output;",
    "- render publication-quality figures and use the command-line entry point.",
    "",
    "The cells are written so they can be executed top-to-bottom on any "
    "session NWB file from the Allen Visual Coding cache; only the path in "
    "the first code cell needs to change.",
))

# ---------------------------------------------------------------------------
# 1. NWB primer
# ---------------------------------------------------------------------------
cells.append(md(
    "## 1. What's inside a Neuropixels NWB file",
    "",
    "NWB (Neurodata Without Borders) is a self-describing HDF5 container "
    "for neurophysiology. An Allen Visual Coding session NWB contains, "
    "among other things:",
    "",
    "- A **Units** table — one row per spike-sorted neuron, with sorted "
    "  spike times stored as a ragged array (`spike_times`) plus ~25 "
    "  columns of quality and waveform metrics.",
    "- An **electrodes** table — one row per recording contact on the "
    "  Neuropixels probe(s), with anatomical location (Allen CCF "
    "  acronym), 3D coordinates, and probe-relative geometry.",
    "- A set of **interval tables** under `nwb.intervals` — one DataFrame "
    "  per stimulus block (`spontaneous_presentations`, "
    "  `drifting_gratings_presentations`, "
    "  `natural_movie_one_presentations`, …). Each row is a "
    "  `(start_time, stop_time, stimulus_parameters)` triplet.",
    "- Optional LFP and behavioural streams.",
    "",
    "`neurocomplexity` reads only the parts it needs:",
    "",
    "| What we read | Where it lives in the NWB |",
    "|---|---|",
    "| spike times (sec) | `units['spike_times']` (ragged) |",
    "| unit metadata | every scalar column of `units` |",
    "| brain area | `electrodes['location']` joined via `units['peak_channel_id']`, exposed as `brain_area` |",
    "| interval tables | `nwb.intervals[*]`, exposed as `rec.intervals[*]` |",
    "",
    "Everything else (LFP, raw voltages, image masks) is ignored.",
))

# ---------------------------------------------------------------------------
# 2. Setup
# ---------------------------------------------------------------------------
cells.append(md(
    "## 2. Setup",
    "",
    "Replace the path below with the location of any Allen Visual Coding "
    "session NWB on your machine. If you do not have one, "
    "[`allensdk.brain_observatory.ecephys`](https://allensdk.readthedocs.io/) "
    "will download one into a local cache.",
))

cells.append(code(
    "from pathlib import Path",
    "import numpy as np",
    "import pandas as pd",
    "",
    "import neurocomplexity as nc",
    "print('neurocomplexity', nc.__version__)",
    "",
    "NWB = Path(",
    "    r'C:\\Users\\Sazgar\\OneDrive\\Desktop\\Arman_Dinarvand code sample'",
    "    r'\\neuropixel\\NeuropixelVisCodingData_cache'",
    "    r'\\session_715093703\\session_715093703.nwb'",
    ")",
    "assert NWB.exists(), f'put a real path here: {NWB}'",
))

# ---------------------------------------------------------------------------
# 3. Loading
# ---------------------------------------------------------------------------
cells.append(md(
    "## 3. Loading an NWB file",
    "",
    "`nc.io.from_nwb(path)` returns a `SpikeRecording`, the central data "
    "object of the package. It is a frozen dataclass: every builder method "
    "(`filter_units`, `with_populations`, `crop`, …) returns a *new* "
    "recording rather than mutating in place. That makes it safe to reuse "
    "and share, and it makes the analysis pipeline a chain of pure "
    "transformations.",
    "",
    "The first call is the slow one (the NWB has to be parsed and the "
    "ragged spike arrays concatenated). After that everything is in-memory "
    "and instantaneous.",
))

cells.append(code(
    "rec = nc.io.from_nwb(NWB)",
    "print(rec)",
    "",
    "# What columns did we get from the Units table?",
    "print()",
    "print('units DataFrame columns:')",
    "print(rec.units.columns.tolist())",
))

cells.append(md(
    "Note `populations=[all(...)]` in the repr: by default every unit is "
    "placed in a single population called `'all'`. Real population labels "
    "(brain areas, cell types) are added in the next sections.",
    "",
    "The `units` DataFrame carries the full Allen quality and waveform "
    "panel. The columns most often used for filtering:",
    "",
    "| column | meaning | typical cutoff |",
    "|---|---|---|",
    "| `quality` | Kilosort label, `'good'` or `'noise'` | `== 'good'` |",
    "| `isi_violations` | refractory-period violation rate | `< 0.5` |",
    "| `amplitude_cutoff` | estimated false-negative rate | `< 0.1` |",
    "| `presence_ratio` | fraction of session unit was active | `> 0.9` |",
    "| `firing_rate` | session-mean rate (Hz) | `> 0.1` |",
    "| `waveform_duration` | spike width in ms (waveform feature) | — |",
    "",
    "These are exactly the cutoffs the Allen Institute recommends in their "
    "Visual Coding white paper."
))

# ---------------------------------------------------------------------------
# 4. Inspect / dataframes
# ---------------------------------------------------------------------------
cells.append(code(
    "# A quick numerical look at the unit population",
    "rec.units[['quality', 'firing_rate', 'isi_violations',",
    "           'amplitude_cutoff', 'presence_ratio', 'waveform_duration',",
    "           'brain_area']].head()",
))

cells.append(code(
    "# How many units per brain area?",
    "rec.units['brain_area'].value_counts().head(15)",
))

cells.append(code(
    "# Which interval (stimulus) tables came with this session?",
    "for name, df in rec.intervals.items():",
    "    print(f'  {name:40s}  {len(df):5d} rows')",
))

# ---------------------------------------------------------------------------
# 5. filter_units
# ---------------------------------------------------------------------------
cells.append(md(
    "## 4. Filtering units (`filter_units`)",
    "",
    "`filter_units` accepts either a pandas-style query string or column "
    "keyword arguments. The query form is the right tool when you want to "
    "apply Allen-style range-based QC:",
))

cells.append(code(
    "rec_qc = rec.filter_units(",
    "    \"quality == 'good' and isi_violations < 0.5 \"",
    "    \"and amplitude_cutoff < 0.1 and presence_ratio > 0.9 \"",
    "    \"and firing_rate > 0.1\"",
    ")",
    "print(f'  before: {rec.n_units:5d} units')",
    "print(f'  after : {rec_qc.n_units:5d} units')",
))

cells.append(md(
    "The keyword form is still there for one-liners:",
))

cells.append(code(
    "rec.filter_units(quality=['good']).n_units",
))

# ---------------------------------------------------------------------------
# 6. cell type classification
# ---------------------------------------------------------------------------
cells.append(md(
    "## 5. Cell-type classification",
    "",
    "Spike width separates **narrow-spiking** units (putative fast-spiking "
    "interneurons, mostly inhibitory) from **broad-spiking** units "
    "(putative regular-spiking pyramidal cells, mostly excitatory). The "
    "Allen NWB stores the per-unit spike width in the `waveform_duration` "
    "column (ms). A 0.4 ms cutoff is the standard threshold in the "
    "literature.",
    "",
    "`classify_cell_type` adds an `ei_class` column to the units table. "
    "After that you can filter or define populations by it.",
))

cells.append(code(
    "rec_ei = rec_qc.classify_cell_type(threshold_ms=0.4)",
    "rec_ei.units['ei_class'].value_counts()",
))

cells.append(code(
    "# What does the spike-width distribution actually look like?",
    "import matplotlib.pyplot as plt",
    "",
    "fig, ax = plt.subplots(figsize=(4, 2.5))",
    "ax.hist(rec_ei.units['waveform_duration'].dropna(), bins=40,",
    "        color='#1f6feb', edgecolor='none')",
    "ax.axvline(0.4, ls='--', color='#d6604d', lw=1)",
    "ax.set_xlabel('spike width (ms)')",
    "ax.set_ylabel('# units')",
    "ax.set_title('narrow ←→ broad split at 0.4 ms')",
    "fig.tight_layout()",
))

# ---------------------------------------------------------------------------
# 7. populations
# ---------------------------------------------------------------------------
cells.append(md(
    "## 6. Defining populations",
    "",
    "Populations are named groups of units. Every analysis function in the "
    "package operates on populations rather than raw units, because "
    "population-level statistics are what scale-free / criticality theory "
    "predicts.",
    "",
    "Two ways to define them:",
    "",
    "- `with_populations(by='column_name')` — one population per unique "
    "  value of the column. The natural choice for `brain_area` or "
    "  `ei_class`.",
    "- `with_populations({'mygroup': mask, ...})` — explicit boolean masks "
    "  if you need something custom.",
    "",
    "The `on_unassigned` argument controls what happens to units missing "
    "the column: `'error'` (default), `'drop'` (remove them), or `'other'` "
    "(bucket them).",
))

cells.append(code(
    "rec_pops = rec_qc.with_populations(by='brain_area', on_unassigned='drop')",
    "print(rec_pops)",
))

cells.append(code(
    "# Same recording, but populations defined by E/I class instead",
    "rec_ei_pops = rec_ei.with_populations(by='ei_class', on_unassigned='drop')",
    "print(rec_ei_pops)",
))

# ---------------------------------------------------------------------------
# 8. cropping
# ---------------------------------------------------------------------------
cells.append(md(
    "## 7. Restricting to a time window",
    "",
    "Two complementary tools:",
    "",
    "- `rec.crop(start, end)` — keep a single contiguous interval.",
    "- `rec.crop_to_intervals(name_or_df)` — keep the union of intervals "
    "  listed in an NWB interval table (or any DataFrame with "
    "  `start_time` / `stop_time` columns). The kept intervals are "
    "  concatenated into a contiguous timeline; absolute timestamps are "
    "  discarded.",
    "",
    "The second one is the right way to isolate, e.g., all spontaneous "
    "(grey-screen) periods in an Allen session, which is what scale-free "
    "statistics require to avoid stimulus-locked contamination.",
))

cells.append(code(
    "# Total spontaneous time available in this session:",
    "spont = rec_pops.intervals['spontaneous_presentations']",
    "print(spont.head())",
    "total = (spont['stop_time'] - spont['start_time']).sum()",
    "print(f'{len(spont)} spontaneous blocks, total {total:.1f} s')",
))

cells.append(code(
    "# Restrict to the longest single spontaneous block (5 min)",
    "longest = spont.iloc[(spont['stop_time']",
    "                      - spont['start_time']).idxmax()]",
    "print(f\"longest block: {longest['start_time']:.1f}\"",
    "      f\" - {longest['stop_time']:.1f} s\")",
    "rec_spont = rec_pops.crop(longest['start_time'], longest['stop_time'])",
    "print(rec_spont)",
))

cells.append(code(
    "# Or take the union of ALL spontaneous blocks:",
    "rec_all_spont = rec_pops.crop_to_intervals('spontaneous_presentations')",
    "print(f'concatenated spontaneous timeline: {rec_all_spont.duration:.1f} s')",
))

# ---------------------------------------------------------------------------
# 9. Provenance
# ---------------------------------------------------------------------------
cells.append(md(
    "## 8. Provenance",
    "",
    "Every recording carries a `ProvenanceRecord` with a fingerprint of the "
    "source file. The fingerprint is BLAKE2b over the first 4 MB, the last "
    "4 MB, and the filesize — chosen so it is fast on a multi-gigabyte NWB "
    "but still detects accidental re-downloads or version bumps. "
    "`results.json` from the CLI embeds this record so a figure can always "
    "be traced back to the exact byte-pattern it came from.",
))

cells.append(code(
    "p = rec.source",
    "print('source format :', p.source_format)",
    "print('source path   :', p.source_path)",
    "print('source hash   :', p.source_hash[:16] + '...')",
    "print('package ver   :', p.package_version)",
    "print('loaded at (UTC):', p.loaded_at)",
))

# ---------------------------------------------------------------------------
# 10. Avalanches & criticality
# ---------------------------------------------------------------------------
cells.append(md(
    "## 9. Analysis — criticality (`nc.analysis.criticality`)",
    "",
    "Bins the pooled population activity at several candidate bin sizes, "
    "detects avalanches as runs of consecutively active bins, and fits "
    "power laws to the size and lifetime distributions. The bin size that "
    "maximises the joint log-log linearity of P(s) and P(T) is selected.",
    "",
    "The result is a `CriticalityResult` dataclass exposing:",
    "",
    "- `alpha_s`, `alpha_t` — the two power-law exponents.",
    "- `optimal_bin_seconds` — the bin size that won.",
    "- `branching` — naive branching ratio (descendants per ancestor).",
    "- `kappa` — Shew et al. (2009) deviation-from-power-law statistic; "
    "  `≈ 1` at criticality.",
    "- `sizes`, `lifetimes` — the raw avalanche arrays for plotting.",
    "- `r_squared` — R² of the power-law fit.",
))

cells.append(code(
    "crit = nc.analysis.criticality(rec_spont, populations=['VISp'])",
    "print(f'alpha_s = {crit.alpha_s:.3f}')",
    "print(f'alpha_t = {crit.alpha_t:.3f}')",
    "print(f'kappa   = {crit.kappa:.3f}')",
    "print(f'R^2     = {crit.r_squared:.3f}')",
    "print(f'optimal bin = {crit.optimal_bin_seconds*1e3:.1f} ms')",
    "print(f'n avalanches = {len(crit.sizes)}')",
))

# ---------------------------------------------------------------------------
# 11. Branching ratio (Wilting MR)
# ---------------------------------------------------------------------------
cells.append(md(
    "## 10. Analysis — branching ratio (`nc.analysis.wilting_mr`)",
    "",
    "The Wilting & Priesemann (2018) multi-step regression estimator. It "
    "fits the lagged autocovariance `r_k = Cov(A_t, A_{t+k}) / Var(A_t)` "
    "to an exponential `r_k = b · m^k`. The branching ratio `m` is robust "
    "to subsampling, which the naive avalanche-based estimator is not.",
    "",
    "- `m < 1`: sub-critical, activity dies out.",
    "- `m = 1`: critical.",
    "- `m > 1`: super-critical, activity diverges.",
    "",
    "Awake cortex universally lives in the *reverberating regime*, "
    "`m ≈ 0.95-0.99`.",
))

cells.append(code(
    "br = nc.analysis.wilting_mr(rec_spont, populations=['VISp'],",
    "                             bin_size_ms=4.0, k_max=50)",
    "print(f'm     = {br.m:.3f}')",
    "print(f'R^2   = {br.r_squared:.3f}')",
    "print(f'n bins = {br.n_bins}')",
))

# ---------------------------------------------------------------------------
# 12. Shape collapse
# ---------------------------------------------------------------------------
cells.append(md(
    "## 11. Analysis — avalanche shape collapse (`nc.analysis.shape_collapse`)",
    "",
    "Friedman et al. (2012) prediction: avalanche shapes of different "
    "durations are rescaled copies of a single universal function "
    "`a(t, T) = T^(γ-1) F(t/T)`. The function grid-scans γ to minimise a "
    "scale-invariant residual between rescaled mean shapes, then refines "
    "with a bounded continuous optimiser.",
    "",
    "Result fields: `gamma`, `residual`, `durations_used`, `mean_shapes` "
    "(per-duration), and `rescaled_x` / `rescaled_y` ready to plot.",
))

cells.append(code(
    "sc = nc.analysis.shape_collapse(rec_spont, populations=['VISp'],",
    "                                 bin_size_ms=4.0, max_duration=80)",
    "print(f'gamma    = {sc.gamma:.3f}')",
    "print(f'residual = {sc.residual:.4f}')",
    "print(f'duration classes used = {len(sc.durations_used)}')",
))

# ---------------------------------------------------------------------------
# 13. Dimensionality
# ---------------------------------------------------------------------------
cells.append(md(
    "## 12. Analysis — dimensionality (`nc.analysis.dimensionality`)",
    "",
    "Participation ratio on the per-unit correlation matrix:",
    "$$\\mathrm{PR} = \\frac{(\\sum_i \\lambda_i)^2}{\\sum_i \\lambda_i^2}$$",
    "PR is bounded in `[1, N]`. PR ≈ 1 means one mode dominates "
    "(low-dimensional); PR ≈ N means activity fills the available space "
    "(high-dimensional, asynchronous-irregular).",
))

cells.append(code(
    "dim = nc.analysis.dimensionality(rec_spont, populations=['VISp'],",
    "                                   bin_size_ms=10.0)",
    "print(f'PR     = {dim.participation_ratio:.2f}')",
    "print(f'N units = {dim.n_units}')",
    "print(f'PR / N = {dim.participation_ratio/dim.n_units:.2f}')",
    "print(f'top 5 eigenvalues = {dim.eigenvalues[:5].round(2)}')",
))

# ---------------------------------------------------------------------------
# 14. Transfer entropy
# ---------------------------------------------------------------------------
cells.append(md(
    "## 13. Analysis — transfer entropy (`nc.analysis.transfer_entropy`)",
    "",
    "Binary Schreiber (2000) transfer entropy between every ordered pair "
    "of populations, with the Miller-Madow bias correction "
    "`TE -= (m-1)/(2N)`. Returns a `TransferEntropyResult` whose "
    "`matrix[i, j]` is `TE(j → i)` in nats — i.e., how much information "
    "the past of population `j` carries about the future of population "
    "`i`, beyond `i`'s own past.",
))

cells.append(code(
    "te = nc.analysis.transfer_entropy(",
    "    rec_spont, populations=['VISp', 'LGd', 'CA1'],",
    "    bin_size_ms=5.0,",
    ")",
    "print('TE matrix (rows = receiver, cols = sender), nats:')",
    "te_df = pd.DataFrame(te.matrix, index=te.populations, columns=te.populations)",
    "te_df.round(5)",
))

# ---------------------------------------------------------------------------
# 15. PID
# ---------------------------------------------------------------------------
cells.append(md(
    "## 14. Analysis — partial information decomposition (`nc.analysis.partial_information`)",
    "",
    "Williams-Beer I_min decomposition of `I(target ; (source1, source2))` "
    "into four atoms: **redundancy**, **unique-1**, **unique-2**, "
    "**synergy**. Counts are discretised into `n_levels` quantile-equal "
    "bins per population (default 3 — binary discretisation collapses on "
    "busy populations). Every MI term is Miller-Madow bias corrected.",
    "",
    "Pick the target population and the two sources whose contribution "
    "you want to decompose.",
))

cells.append(code(
    "pid = nc.analysis.partial_information(",
    "    rec_spont, target_pop='VISp', sources=['LGd', 'CA1'],",
    "    bin_size_ms=5.0, n_levels=3,",
    ")",
    "print(f'redundancy = {pid.redundancy:.4f} nats')",
    "print(f'unique LGd = {pid.unique_1:.4f} nats')",
    "print(f'unique CA1 = {pid.unique_2:.4f} nats')",
    "print(f'synergy    = {pid.synergy:.4f} nats')",
    "print(f'total MI   = {pid.total_mi:.4f} nats')",
))

# ---------------------------------------------------------------------------
# 16. Autonomy
# ---------------------------------------------------------------------------
cells.append(md(
    "## 15. Analysis — autonomy (`nc.analysis.autonomy`)",
    "",
    "Granger-style autonomy index — for each population in turn, fit a "
    "VAR with all populations as regressors and an autoregressive model "
    "on that population alone, then F-test whether removing the external "
    "regressors significantly hurts prediction. The returned value per "
    "population is the F-test p-value: high p → externals do not help → "
    "the population is autonomous; low p → it is driven by the others.",
))

cells.append(code(
    "aut = nc.analysis.autonomy(rec_spont,",
    "                            populations=['VISp', 'LGd', 'CA1'],",
    "                            bin_size_ms=10.0)",
    "for pop, p in aut.values.items():",
    "    print(f'  {pop:5s}  autonomy index (F-test p-value) = {p:.3f}')",
    "print(f'lags considered up to {aut.max_lag}')",
))

# ---------------------------------------------------------------------------
# 17. Surrogate test
# ---------------------------------------------------------------------------
cells.append(md(
    "## 16. Null testing — `nc.analysis.surrogate_test`",
    "",
    "Wraps any scalar statistic of a recording in a non-parametric "
    "permutation test. The recording is repeatedly jittered or "
    "ISI-shuffled, the statistic is recomputed on each surrogate, and an "
    "observed-vs-null distribution is returned with a two-sided rank "
    "p-value.",
    "",
    "Pass the test a small function that takes a recording and returns a "
    "single number.",
))

cells.append(code(
    "def stat(r):",
    "    return nc.analysis.criticality(r, populations=['VISp'],",
    "                                    bin_size_ms=[4, 8, 16]).alpha_s",
    "",
    "test = nc.analysis.surrogate_test(",
    "    rec_spont, stat, method='jitter',",
    "    n_shuffles=20, jitter_ms=10.0, seed=0,",
    ")",
    "print(f'observed alpha_s = {test.observed:.3f}')",
    "print(f'null mean        = {np.nanmean(test.null_values):.3f}')",
    "print(f'z score          = {test.z_score:.2f}')",
    "print(f'two-sided p      = {test.p_value:.3f}')",
))

# ---------------------------------------------------------------------------
# 18. Visualisation
# ---------------------------------------------------------------------------
cells.append(md(
    "## 17. Visualisation (`nc.viz`)",
    "",
    "Each analysis has a matching figure function that returns a "
    "Nature-style matplotlib `Figure`: Arial 7 pt, editable PDF/SVG text, "
    "no top/right spines, restrained palette. `viz.figure_overview` "
    "renders all five analyses on a single double-column page.",
))

cells.append(code(
    "fig = nc.viz.figure_criticality(crit); fig",
))

cells.append(code(
    "fig = nc.viz.figure_branching(br); fig",
))

cells.append(code(
    "fig = nc.viz.figure_shape_collapse(sc); fig",
))

cells.append(code(
    "fig = nc.viz.figure_dimensionality(dim); fig",
))

cells.append(code(
    "fig = nc.viz.figure_pid(pid); fig",
))

cells.append(code(
    "fig = nc.viz.figure_overview({",
    "    'criticality':    crit,",
    "    'branching':      br,",
    "    'shape_collapse': sc,",
    "    'dimensionality': dim,",
    "    'pid':            pid,",
    "}, title='session 715093703  —  VISp, spontaneous')",
    "",
    "nc.viz.save_publication(fig, 'tutorial_output/overview',",
    "                        formats=('pdf', 'svg', 'png'))",
    "fig",
))

# ---------------------------------------------------------------------------
# 19. CLI
# ---------------------------------------------------------------------------
cells.append(md(
    "## 18. The command-line interface",
    "",
    "Everything above also collapses to a single shell command. Three "
    "subcommands:",
    "",
    "- `neurocomplexity info <nwb>` — quick description of an NWB "
    "  recording.",
    "- `neurocomplexity analyze <nwb> -o <dir>` — run the whole pipeline, "
    "  emit a `results.json` and Nature-style figures.",
    "- `neurocomplexity figure <results.json> -o <dir>` — re-render "
    "  figures from a cached run, no NWB needed.",
    "",
    "Typical call:",
    "",
    "```bash",
    "neurocomplexity analyze session_715093703.nwb \\",
    "    --start 3765.6 --end 4066.9 \\",
    "    --populations VISp \\",
    "    --target VISp --sources LGd CA1 \\",
    "    --quality good \\",
    "    -o results/session_715093703/",
    "```",
    "",
    "`--start`/`--end` are in seconds on the recording's own timeline. "
    "All quality / bin / discretisation flags can be set from the command "
    "line; run `neurocomplexity analyze --help` for the full list.",
))

# ---------------------------------------------------------------------------
# 19b. Was that result real? Inference
# ---------------------------------------------------------------------------
cells.append(md(
    "## Was that result real? Inference",
    "",
    "Every analysis above returns a point estimate. To make those numbers "
    "interpretable we need null distributions (for TE and PID — is the value "
    "above what we'd expect from preserved firing statistics alone?) and "
    "confidence intervals (for criticality and dimensionality — how precise "
    "is the estimate?).",
    "",
    "`neurocomplexity.inference` provides `test()` for surrogate-based null "
    "tests and `bootstrap()` for confidence intervals.",
))
cells.append(code(
    "from neurocomplexity.inference import test as inf_test, bootstrap as inf_bootstrap",
    "",
    "# Surrogate null test on a transfer-entropy matrix:",
    "te = nc.analysis.transfer_entropy(rec, populations=['VISp'],",
    "                                  bin_size_ms=20, delay_bins=1)",
    "te_inf = inf_test(te, rec, surrogate='isi_shuffle', n=200, seed=0)",
    "print('TE matrix:')",
    "print(te.matrix)",
    "print('p-values (FDR-corrected):')",
    "print(te_inf.p_value_fdr)",
))
cells.append(code(
    "# Bootstrap CI on the branching ratio:",
    "br = nc.analysis.wilting_mr(rec, populations=['VISp'])",
    "m_inf = inf_bootstrap(br, rec, n=200, seed=0, block_seconds=5.0)",
    "print(f'm = {m_inf.observed:.3f} "
    "(95% CI [{m_inf.ci_lower:.3f}, {m_inf.ci_upper:.3f}])')",
))

# ---------------------------------------------------------------------------
# 20. Recap
# ---------------------------------------------------------------------------
cells.append(md(
    "## 19. Recap",
    "",
    "The full chain from raw NWB to manuscript figure, in seven lines:",
    "",
    "```python",
    "rec = (nc.io.from_nwb(NWB)",
    "         .filter_units(\"quality == 'good' and isi_violations < 0.5 \"",
    "                       \"and amplitude_cutoff < 0.1\")",
    "         .with_populations(by='brain_area', on_unassigned='drop')",
    "         .crop_to_intervals('spontaneous_presentations'))",
    "",
    "crit = nc.analysis.criticality(rec, populations=['VISp'])",
    "br   = nc.analysis.wilting_mr(rec, populations=['VISp'])",
    "sc   = nc.analysis.shape_collapse(rec, populations=['VISp'])",
    "dim  = nc.analysis.dimensionality(rec, populations=['VISp'])",
    "pid  = nc.analysis.partial_information(rec, target_pop='VISp',",
    "                                          sources=['LGd', 'CA1'])",
    "",
    "nc.viz.save_publication(",
    "    nc.viz.figure_overview({",
    "        'criticality': crit, 'branching': br,",
    "        'shape_collapse': sc, 'dimensionality': dim, 'pid': pid,",
    "    }), 'overview')",
    "```",
    "",
    "That's the whole package.",
))


# ---------------------------------------------------------------------------
# Validation: benchmark suite
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
    "## 20. Validation: how do we know the numbers are right?\n\n"
    "Every analysis in `neurocomplexity` is validated against synthetic data\n"
    "with closed-form or simulator-derived ground truth. The eleven\n"
    "benchmark cases live in `neurocomplexity.benchmarks`; full description\n"
    "in [`docs/benchmarks.md`](../docs/benchmarks.md)."
))
cells.append(nbf.v4.new_code_cell(
    "from neurocomplexity.benchmarks import run_all\n"
    "df = run_all(cases=['pid.atoms_xor', 'pid.atoms_rdn'],\n"
    "             n_reps=3, seed=0, verbose=False)\n"
    "df"
))


# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.12",
        "pygments_lexer": "ipython3",
        "mimetype": "text/x-python",
        "file_extension": ".py",
    },
}

out = Path(__file__).parent / "tutorial.ipynb"
nbf.write(nb, out)
print(f"wrote {out}  ({len(cells)} cells)")
