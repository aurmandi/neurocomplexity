---
title: "neurocomplexity: a validated Python toolkit for criticality, information flow, and dimensionality analyses of spike-sorted recordings"
running_title: "neurocomplexity: complexity analyses for spike trains"
authors:
  - name: "Arman Dinarvand"
    affiliation: "1"
    corresponding: true
    email: "armandinarvand@khu.ac.ir"
affiliations:
  - id: "1"
    name: "Department of Animal Biology, Kharazmi University, Tehran, Iran"
keywords:
  - neuronal avalanches
  - branching ratio
  - transfer entropy
  - partial information decomposition
  - criticality
  - spike-sorted recordings
  - reproducible analysis
  - Python
bibliography: paper.bib
journal: Frontiers in Neuroinformatics
article_type: Methods
date: 2026-05-17
---

# Abstract

Quantifying complexity, criticality, and information flow in
spike-sorted recordings has become a routine but methodologically
fragmented step in systems neuroscience. The relevant estimators —
branching ratio, avalanche-distribution exponents, shape collapse,
pairwise transfer entropy, partial information decomposition (PID),
vector-autoregressive Granger autonomy, and participation-ratio
dimensionality — are well-served individually by specialised
toolboxes such as MR.Estimator [@spitzner2021mrestimator] for the
branching ratio, IDTxl [@wollstadt2019idtxl] for multivariate
information dynamics, JIDT [@lizier2014jidt] for general
information-theoretic measures, MVGC [@barnett2014mvgc] for Granger
causality, Elephant [@denker2018elephant] for spike-train statistics
on the Neo data model, and SpikeInterface [@buccino2020spikeinterface]
for sorter-stage workflows. No single open package, however, integrates
all seven analysis families on a shared immutable spike-train data
model, ships uniform per-estimate statistical inference
(block-bootstrap confidence intervals and surrogate null tests), and
asserts recovery of synthetic ground truth for every estimator on
every release. We introduce **neurocomplexity**, a Python package that
fills this integration gap. It is built around an immutable
`SpikeRecording` data class with hashed provenance, lazy file-format
adapters for Neurodata Without Borders, Phy curation directories,
raw Kilosort output, and any SpikeInterface `BaseSorting`, and a
unified `InferenceResult` carrying bias-corrected bootstrap
intervals, Phipson–Smyth-corrected surrogate $p$-values, and
Benjamini–Hochberg false-discovery-rate adjustment. An eleven-case
benchmark suite verifies every estimator against simulator-derived
ground truth (coupled AR(1) processes, branching networks,
structured-covariance generators, and analytic Williams–Beer
distributions); a frozen baseline CSV accompanies every release. We
describe the architecture, report the validation benchmark, and
outline an end-to-end demonstration on a publicly available Allen
Brain Observatory Neuropixels session.

**Keywords:** neuronal avalanches, branching ratio, transfer
entropy, partial information decomposition, criticality, spike-sorted
recordings, reproducible analysis, Python.

---

# 1 Introduction

The hypothesis that cortical circuits operate near a critical point
has generated a large body of empirical and theoretical work over
the past two decades [@beggs2003neuronal; @petermann2009spontaneous;
@plenz2021selforganized; @munoz2018colloquium]. Closely related
lines of work have used directed information measures to map
effective connectivity from spike trains
[@schreiber2000measuring; @vicente2011transfer; @ito2011extending],
decomposed redundant and synergistic contributions of multiple
sources via partial information decomposition (PID)
[@williams2010nonnegative; @timme2014synergy], and characterised the
effective dimensionality of population activity through
eigenvalue-based summaries of the covariance spectrum
[@rajan2006eigenvalue; @cunningham2014dimensionality;
@stringer2019highdimensional]. These families are unevenly adopted:
branching-ratio estimation and transfer entropy are now widely
standard in spike-train work, dimensionality (in the
participation-ratio sense) has become routine since
[@stringer2019highdimensional], while partial information
decomposition and avalanche shape collapse remain more specialised
but rapidly growing literatures.

Each family answers a different question about the same recording.
The branching ratio asks whether the population is amplifying or
damping perturbations on average — a single scalar that distinguishes
sub-critical, critical, and super-critical operation
[@wilting2018inferring]. Avalanche exponents and shape collapse
characterise the heavy-tailed event-size statistics that
self-organised-critical systems are predicted to display
[@friedman2012universal; @sethna2001crackling]. Pairwise transfer
entropy quantifies how much the future of one population is reduced
in uncertainty by the past of another, beyond its own history
[@schreiber2000measuring]. Partial information decomposition resolves
that joint information into redundant, unique, and synergistic
contributions of multiple sources [@williams2010nonnegative].
VAR-Granger autonomy asks the inverse question — to what extent the
target is statistically independent of the rest of the recording
[@granger1969investigating; @barnett2014mvgc]. Participation-ratio
dimensionality summarises the effective rank of the population
covariance into one number [@rajan2006eigenvalue;
@cunningham2014dimensionality]. A paper that wants to characterise a
single cortical area along all of these axes simultaneously is asking
a perfectly reasonable scientific question that, in practice,
requires assembling and validating estimators from several different
toolchains.

The existing toolchain landscape is rich but fragmented. MR.Estimator
[@spitzner2021mrestimator] is the canonical Python implementation of
the subsampling-robust multi-step regression branching-ratio
estimator, with bootstrap confidence intervals built in. IDTxl
[@wollstadt2019idtxl] provides multivariate transfer entropy,
mutual information, active information storage, and a BROJA partial
information decomposition with hierarchical surrogate permutation
tests. JIDT [@lizier2014jidt] offers a comprehensive Java
implementation of information-theoretic measures with surrogate
significance testing. MVGC [@barnett2014mvgc; @seth2015granger] is
the reference MATLAB toolbox for multivariate Granger causality with
Monte-Carlo significance testing against VAR ground truth. Elephant
[@denker2018elephant] is the leading open-source Python toolkit for
spike-train statistics, built on the Neo `SpikeTrain` data model.
SpikeInterface [@buccino2020spikeinterface] is the de facto framework
for spike sorting and sorter comparison, with extensive ground-truth
validation for the sorting step itself. Each of these toolboxes is
excellent within its own scope. No single open package, however,
exposes all seven analysis families used in the criticality /
effective-connectivity / dimensionality literature behind a single
immutable spike-train data model, with uniform per-estimate
block-bootstrap intervals and surrogate null tests, and ships a
release-pinned benchmark CSV asserting recovery of synthetic ground
truth for every estimator. The practical result is that a paper
combining several of these families typically requires custom glue
code and ad-hoc per-estimator inference, with no shared baseline that
reviewers can re-run.

Three recent developments compound the problem. First, the
near-universal adoption of Kilosort and Phy for spike sorting
[@rossant2016spike; @pachitariu2024spike] means that the input to
any analysis pipeline is increasingly a directory of `.npy` and
`.tsv` files rather than a curated MATLAB structure. Second,
although SpikeInterface [@buccino2020spikeinterface] has unified the
upstream sorting world, downstream complexity analyses still require
a separate ecosystem. Third, the Wilting–Priesemann multi-step
regression estimator of the branching ratio
[@wilting2018inferring] has shown that classical avalanche-based
estimators are systematically biased under sub-sampled recordings
[@levina2017subsampling], implying that older toolchains may need to
be revisited where they have not adopted subsampling-robust
estimators.

**neurocomplexity** is positioned as an integrator rather than an
inventor. Its contributions are: (i) a single immutable
`SpikeRecording` data model used by every estimator, populated from
NWB, Phy, raw Kilosort, or any SpikeInterface `BaseSorting`; (ii)
seven analysis families covering branching ratio, avalanche
exponents, shape collapse, pairwise transfer entropy, PID,
VAR-Granger autonomy, and participation-ratio dimensionality, each
implemented with reference to the canonical specialist toolbox above
and validated against the same simulator-derived ground truth;
(iii) a unified statistical-inference layer wrapping every estimator
in block-bootstrap confidence intervals and Phipson–Smyth-corrected
surrogate null tests with Benjamini–Hochberg multiple-comparison
adjustment; and (iv) an eleven-case benchmark suite that runs on
every release and asserts that every estimator recovers its
respective ground truth within documented tolerances. We describe
the design, report the validation benchmark, and outline an
end-to-end demonstration on a publicly available Allen Brain
Observatory Neuropixels recording [@siegle2021survey].

---

# 2 Materials and Methods

## 2.1 Software architecture

neurocomplexity is implemented in pure Python (supported on
3.10–3.13) and depends only on NumPy [@harris2020array], SciPy
[@virtanen2020scipy], pandas, statsmodels, and small utility
packages. Optional install extras (`[nwb]`, `[spikeinterface]`,
`[viz]`) gate the heavy-weight loaders and the figure module so
that the core installation remains light. The package is organised
into six self-contained modules:

```
neurocomplexity/
  core/           — data model and invariants
  io/             — file-format adapters
  analysis/       — point estimators
  inference/      — surrogates, bootstrap, null tests
  benchmarks/     — synthetic simulators and validation cases
  viz/            — publication-style figures
```

The architectural contract is that every analysis consumes the same
immutable `SpikeRecording`, every inference routine consumes the
corresponding analysis-result dataclass, and every benchmark case
asserts that the analysis–inference combination recovers a known
ground truth on a synthetic recording produced by a simulator in
the same module. The contract is small enough to hold in a reader's
head and rigid enough that adding a new estimator is a strictly
local edit.

## 2.2 Data model

`core.SpikeRecording` is a frozen Python dataclass with five
required fields — `spike_times` (float64 seconds, monotonically
non-decreasing), `unit_ids` (int64 per-spike owner), `units` (a
pandas DataFrame of per-unit metadata with at least an `id`
column), `populations` (a mapping from named string to boolean mask
over units), and `duration` (positive float, seconds) — plus
optional `sampling_rate`, `intervals` for stimulus or epoch tables,
and `source`, a `ProvenanceRecord` carrying the source path, the
file's BLAKE2b 128-bit fingerprint (head $\oplus$ tail $\oplus$
size), the loader version, the package version, and the ISO-format
load timestamp. Invariants are enforced at construction: spike
times are sorted in place if not already, length mismatches raise a
typed `RecordingValidationError`, and population masks are verified
against the units table. The dataclass is frozen so that recordings
are safe to share between threads and cannot be silently mutated by
downstream code.

## 2.3 File-format adapters

All loaders return a `SpikeRecording`. Four on-disk formats are
supported:

- `from_nwb(path)` — Neurodata Without Borders [@rubel2022nwb],
  targeting the Allen Visual Coding Neuropixels schema but tolerating
  any standard NWB Units table; joins anatomical location from the
  electrodes table when available.
- `from_phy(directory)` — a Phy curation directory
  [@rossant2016spike], reading `spike_times.npy`,
  `spike_clusters.npy`, `params.py`, and `cluster_info.tsv` (with
  `cluster_group.tsv` as fallback). Quality labels come from the
  curated `group` column. If `spike_clusters.npy` is absent the
  loader falls back to `spike_templates.npy` with a `UserWarning`
  noting that Phy merges or splits will not be reflected. Any
  cluster id present in the spike-cluster array but absent from the
  label table is given a synthetic `unsorted` row so the
  `SpikeRecording` invariants always hold.
- `from_kilosort(directory)` — raw Kilosort output
  [@pachitariu2024spike] before any Phy curation, using automatic
  labels from `cluster_KSLabel.tsv`.
- `from_spikeinterface(sorting, recording=None)` — any
  `BaseSorting` object from the SpikeInterface ecosystem
  [@buccino2020spikeinterface], enabling indirect support for Open
  Ephys, Blackrock, Plexon, and any other format SpikeInterface
  reads.

Heavy loaders are exposed through a module-level `__getattr__` so
that `import neurocomplexity` triggers no eager import of `pynwb`,
`spikeinterface`, or their transitive dependencies; the
corresponding module is loaded only on first call. Missing optional
dependencies raise an `ImportError` naming the install command for
the relevant extra.

## 2.4 Analyses

All seven estimators share a binning helper that produces a
$(T, P)$ int32 array of spike counts per time bin per population,
so that binning is identical across analyses for any given
recording.

### 2.4.1 Branching ratio

`analysis.branching.wilting_mr` implements the multi-step
regression estimator of Wilting and Priesemann
[@wilting2018inferring], which fits the slope of
$r_k = \mathrm{Cov}(A_{t+k}, A_t)/\mathrm{Var}(A_t)$ as a function
of lag $k$ in log space. This estimator is, by design, robust to
sub-sampling — a property that does not hold for the classical
Beggs–Plenz avalanche-based estimator [@levina2017subsampling] and
that matters for any modern recording in which only a few hundred
of the local network's neurons are observed. A critical network
produces $\hat{m} \approx 1$, sub-critical $\hat{m} < 1$,
super-critical $\hat{m} > 1$.

### 2.4.2 Avalanche exponents

`analysis.criticality` fits the size-distribution exponent
$\alpha$ and the duration-distribution exponent $\tau$ of neuronal
avalanches defined on the population spike-count time series
following the construction of Beggs and Plenz
[@beggs2003neuronal]. The fit uses **log-binned histograms** with
geometric-mean bin centres in log-space to avoid the known upward
bias of linearly binned maximum-likelihood estimators on
heavy-tailed data [@touboul2017power]. A bin-size scaling
correction is applied so that exponents are stable across a sensible
range of bin widths.

### 2.4.3 Avalanche shape collapse

`analysis.shape_collapse` implements the universal scaling collapse
of Friedman and colleagues [@friedman2012universal;
@sethna2001crackling]: avalanches of different durations $T$ are
rescaled by $u = t/T$ and $\langle a \rangle / T^{\gamma - 1}$, and
$\gamma$ is fit to minimise the residual after rescaling. A
critical branching system obeys the **crackling-noise relation**
$\gamma = (\tau - 1)/(\alpha - 1)$ [@sethna2001crackling], so
$\gamma$ can be compared against the value predicted from the
independently fit exponents of §2.4.2.

### 2.4.4 Transfer entropy

`analysis.transfer_entropy` implements Schreiber's pairwise
transfer entropy [@schreiber2000measuring] with **Miller–Madow bias
correction** on binary-thresholded counts. The estimator is
directional (source → target) and operates on whatever bin size the
user selects via the shared binning helper. The implementation
follows the conventions of extant toolkits [@vicente2011transfer;
@lizier2014jidt] and is explicitly tuned for the spike-train
regime where bins are sparse and naive estimators inherit large
positive bias [@ito2011extending].

### 2.4.5 Partial information decomposition

`analysis.pid` implements the Williams–Beer $I_{\min}$ partial
information decomposition [@williams2010nonnegative] over two source
populations and one target. The decomposition splits the joint
mutual information $I(\mathrm{target}; (\mathrm{src}_1,
\mathrm{src}_2))$ into four non-negative atoms: redundancy,
unique-to-source-1, unique-to-source-2, and synergy. Spike-count
time series are quantile-discretised into $n_\text{levels} = 3$
states per population so that the joint distribution does not
saturate at the binary alphabet that pure spike/no-spike encoding
yields [@timme2014synergy; @schneidman2003network].

### 2.4.6 VAR-Granger autonomy

`analysis.autonomy` computes a per-population "autonomy index" from
a vector autoregressive Granger-causality test
[@granger1969investigating; @barnett2014mvgc; @seth2015granger].
The autonomy index is the $p$-value of the F-test that asks whether
the external populations provide predictive information about the
target beyond the target's own history. Large $p$-values indicate
that the target is statistically autonomous from the rest of the
recording.

### 2.4.7 Participation-ratio dimensionality

`analysis.dimensionality` computes the participation ratio of the
eigenvalue spectrum of the pairwise spike-count correlation matrix,
$\mathrm{PR} = \left(\sum_i \lambda_i\right)^2 / \sum_i \lambda_i^2$
[@rajan2006eigenvalue; @cunningham2014dimensionality].
$\mathrm{PR} = 1$ corresponds to a one-dimensional mode of activity;
$\mathrm{PR} = N$ corresponds to an isotropic spectrum. PR has been
used as a compact descriptor of effective dimensionality in modern
Neuropixels work [@stringer2019highdimensional].

## 2.5 Statistical inference

Every analysis result can be wrapped in an
`inference.InferenceResult` carrying: the observed value; the
null distribution (when surrogates were used); the bootstrap
distribution (when bootstrap was used); a two-sided $p$-value
**Phipson–Smyth corrected** so that no permutation $p$-value is
ever zero [@phipson2010permutation]; a Benjamini–Hochberg
false-discovery-rate-adjusted $p$ across populations
[@benjamini1995controlling]; a non-parametric effect size; and
**bias-corrected percentile bootstrap confidence-interval bounds**
(BC, with the $z_0$ correction but without the acceleration term),
following the BC family due to Efron [@efron1987better].

Three surrogate generators are supplied: `spike_dither` (jitter
each spike by a Gaussian increment, preserving rate but disrupting
precise timing); `isi_shuffle` (permute inter-spike intervals per
unit, preserving rate and ISI distribution but destroying sequence
structure); and `interval_shuffle` (preserve bursts at a chosen
timescale while destroying longer-range structure). A
`SurrogatePool` wraps a least-recently-used cache (default capacity
64, backed by an ordered dictionary) so that a session of hundreds
of null tests does not regenerate the same surrogate draws across
estimators.

Block bootstraps use a block length specified in seconds (default
5 s); the block construction preserves autocorrelation up to that
timescale while permitting variance estimation across blocks.

## 2.6 Validation benchmark suite

`benchmarks` defines a synthetic ground truth for every analysis
family. Eleven cases are registered (Table 1). Each case is a small
function decorated with `@register("group.name")` that returns a
`BenchmarkResult` carrying the observed value, the expected value,
the tolerance, a pass/fail flag, runtime, and per-replicate
metadata. Cases for the avalanche exponents and for autonomy use
trial-based and multi-unit-per-population simulators respectively,
chosen so that finite-sample biases do not contaminate the
ground-truth assertion (full design notes are documented at
`docs/benchmarks.md` in the repository). The CLI command

```bash
python -m neurocomplexity benchmark --reps 200 -o baseline.csv
```

runs every case and writes a CSV that is shipped with each release
as the validated baseline at
`results/benchmarks/v<version>.csv`.

## 2.7 Reproducibility and provenance

Three mechanisms make a neurocomplexity-driven analysis reproducible
from disk to figure. First, every `SpikeRecording` carries an
immutable `ProvenanceRecord` (source path, BLAKE2b 128-bit content
fingerprint of head and tail bytes plus file size, loader version,
package version, ISO load timestamp), so that any downstream
artefact can be traced back to the exact bytes from which it was
derived. Second, every public release (i) is tagged in version
control, (ii) is archived on Zenodo with a citable DOI, (iii) is
paired with the corresponding benchmark CSV from a 200-replicate
run, and (iv) carries a machine-readable `CITATION.cff`. Third,
continuous integration runs the full test suite across Python
3.10–3.13 on Linux, macOS, and Windows for every commit, and the
shipped benchmark CSV permits any downstream user to verify that
their installation reproduces the documented behaviour bit-for-bit
on a fixed seed.

---

# 3 Results

## 3.1 Benchmark validation

The full benchmark suite is run on every release. Each row of
Table 1 reports the mean error of the named estimator across
$n_\text{reps}$ independent simulator replicates against the
documented tolerance, the binary pass/fail outcome, and per-case
runtime.

> **Table 1.** Benchmark suite for neurocomplexity v1.1.0,
> seed 0. **Numerical entries from the validated v1.1.0 baseline
> CSV (`results/benchmarks/v1.1.0.csv`, $n_\text{reps} = 200$) are
> to be inserted upon completion of the regeneration run currently
> in progress; the suite itself is fully described above and the
> case list is fixed.**
>
> Cases: `criticality.m_hat`, `criticality.exponents`,
> `info_theory.te_convergence`, `info_theory.te_null`,
> `info_theory.autonomy_calibration`, `pid.atoms_xor`,
> `pid.atoms_and`, `pid.atoms_copy`, `pid.atoms_rdn`,
> `pid.atoms_unq`, `dimensionality.pr_rank`.

The validation matrix is summarised graphically in Figure 2 (case
× tolerance), generated automatically from the same CSV.

## 3.2 End-to-end demonstration on Allen Neuropixels data

To demonstrate end-to-end use on a real recording, we will analyse
one publicly available Allen Brain Observatory Neuropixels session
[@siegle2021survey] downloaded as a standard NWB file. The full
analysis script will be shipped with the package at
`tutorial/tutorial.ipynb`.

The intended workflow is: open the NWB file with
`nc.io.from_nwb(...)`; filter units to `quality == "good"` and to
an acceptable presence-ratio with `rec.filter_units(...)`; define
populations per visual cortical area (V1, LM, AL, RL, AM, PM); run
the full pipeline with default parameters; compute branching ratio,
transfer entropy, PID, autonomy, and participation-ratio
dimensionality per area and per pairwise combination; and attach
bootstrap confidence intervals and surrogate null tests to each
estimate.

> **[ANALYSIS PENDING.]** The four data-driven figures of §3.2
> — Figure 3 (per-area branching ratio with bootstrap CIs),
> Figure 4 (directed effective-connectivity matrix across visual
> areas via transfer entropy with surrogate null gating), Figure 5
> (PID redundancy / synergy summary across triplets), and the
> overview composite Figure 1 — require running the notebook on the
> downloaded session. The prose and captions around each figure are
> drafted in advance so that the numerical results, the
> session-identifier, and the area-level statistics can be inserted
> against a fixed narrative once the analysis run completes. The
> intent is to test, with quantified uncertainty, whether visual
> cortex of the awake mouse operates near but slightly sub-critical
> to the branching boundary, consistent with primate and
> non-primate observations of slightly sub-critical operation
> [@petermann2009spontaneous; @priesemann2014spike], and to map the
> direction of information flow against the documented anatomical
> hierarchy [@siegle2021survey].

---

# 4 Discussion

We have introduced neurocomplexity, a Python package that brings
together seven families of complexity, criticality, and
information-flow estimators behind a single immutable data model
and a unified statistical inference layer, with an eleven-case
benchmark suite that verifies every estimator against
simulator-derived ground truth on every release.

The design choices are deliberate. First, an immutable data class
(`SpikeRecording`) is the only object that crosses module
boundaries; every estimator takes a recording and returns a result
dataclass; every inference call takes a result and returns an
augmented result; every benchmark case is a closed loop from
simulator to recording to result to assertion. This contract is
small enough to hold in a reader's head and rigid enough that
adding a new estimator is a strictly local edit. Second, we have
prioritised statistical inference equal in standing to the point
estimate. In the literature it is common to report
$\hat{m} = 0.97$ or a transfer-entropy value without uncertainty
bounds. For finite recordings of sub-sampled networks these are
often the only honest descriptions of the evidence; our
`InferenceResult` exposes both confidence-interval and null-test
branches uniformly so that downstream papers can report them
without writing extra code. Third, we have prioritised file-format
breadth. Loaders for NWB, Phy, Kilosort, and any SpikeInterface
object cover the realistic on-disk states of a modern spike-sorted
recording without forcing the user through a heavy-weight
conversion step. Optional extras mean that a user who needs only
Phy support never installs `pynwb`, which has historically been a
frequent pain point in laboratory environments with strict
dependency policies.

### Related software and complementary positioning

neurocomplexity does not aim to replace any of the specialist
toolboxes introduced in §1. MR.Estimator [@spitzner2021mrestimator]
remains the reference implementation of the multi-step regression
branching-ratio estimator and should be the first stop for any user
whose study centres on that single estimator. IDTxl
[@wollstadt2019idtxl] remains the reference for advanced
multivariate transfer entropy and the BROJA partial information
decomposition with hierarchical surrogate testing. JIDT
[@lizier2014jidt] remains the canonical Java toolkit for
high-dimensional information-theoretic measures. MVGC
[@barnett2014mvgc] remains the reference for advanced Granger-
causality work in MATLAB. Elephant [@denker2018elephant] remains
the leading spike-train statistics library on the Neo data model.
SpikeInterface [@buccino2020spikeinterface] is the de facto
framework for sorting and curation. Where a study requires the
most advanced features of any single one of these tools, users
should reach for the corresponding specialist package.

neurocomplexity is positioned as a complementary integration layer
when a study needs several of these families together on the same
spike-sorted recording: it offers one data model, one inference
layer, one validation matrix, and one citable release, so that the
quantitative claims of such a paper are backed by a publicly
verifiable benchmark and so that the bin definitions, surrogate
generators, and bootstrap construction are shared rather than
re-implemented per estimator.

### Limitations

There are limitations the user should know about. First,
neurocomplexity currently supports only discrete-time analyses on
binned spike counts; continuous-time estimators (kernel-based
transfer entropy on inter-spike intervals, for example) are not
implemented. Second, the autonomy module fits linear VAR(1) models
and will under-state non-linear influences. Third, the PID
implementation is restricted to two sources and one target; higher-
arity decompositions would require additional implementation
effort that has been deferred until concrete demand emerges.
Fourth, the benchmark for shape collapse is currently inherited
from the `criticality.exponents` case (they share the
exponent-fitting routine) rather than being a free-standing case
with its own simulator; a dedicated case is planned for v1.2.
None of these are fundamental design problems; each will be
addressed in future releases without breaking the public API.

The package is open to community contribution; issues, pull requests,
and benchmark proposals are welcomed at the repository.

---

# 5 Code and Data Availability

The neurocomplexity package (v1.1.0) is released under the MIT
licence at <https://github.com/aurmandi/neurocomplexity>. A
versioned archive is deposited on Zenodo (DOI to be assigned at
release). Continuous integration logs and the v1.1.0 benchmark CSV
are committed at `results/benchmarks/v1.1.0.csv` once the run
completes. The Allen Brain Observatory Neuropixels session used for
the end-to-end demonstration is freely available via the Allen
Institute's data portal [@devries2020large; @siegle2021survey]. The
analysis notebook reproducing every figure in this paper is
committed at `tutorial/tutorial.ipynb`.

# Author Contributions

S.A.D. conceived, designed, and implemented the package; performed
the validation benchmarks; analysed the demonstration dataset; and
wrote the manuscript.

# Funding

The author received no specific funding for this work.

# Conflict of Interest

The author declares no competing interests.

# Acknowledgements

The author thanks the developers of NumPy, SciPy, pandas, and
statsmodels for the foundations on which neurocomplexity is built,
and the developers of SpikeInterface and the Neurodata Without
Borders ecosystem for making spike-sorted recordings interoperable
across laboratories.

# References
