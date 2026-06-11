# Steinmetz funnel reconciliation + surrogate-null reuse (2026-06-10)

## Canonical decision (2026-06-11)

The manuscript exists in two drafts that had diverged on the §3.3 selection
criterion. **paper.tex is canonical** (it builds paper.pdf and is the more
rigorous draft); paper.md is non-canonical and slated for regeneration from
paper.tex. The canonical criterion is **σ-free**: the branching ratio is
sub-sampling-biased (it correlates with unit count, Pearson r≈+0.40) and is
therefore reported only as a negative control, **not** used to gate selection.

## B4 — the canonical σ-free funnel (64 / 7 / 57 / 37)

Reproduced from `datasets/steinmetz_results.csv` with the `steinmetz_table3_ci.py`
TRACT set and thresholds (N≥30, sethna_delta≤0.10, R²≥0.85; no σ gate):

| Stage | Count |
|------|------:|
| Populations with ≥30 units and a default-stationary 200-s epoch | **64** |
| − fibre tracts (a priori non-neural) | 7 |
| Grey-matter populations | **57** |
| — fail crackling-noise consistency (Δγ>10% or R²<0.85) | 20 |
| — **pass (crackling-consistent) → Table 3** | **37** |

Check: 7 + 20 + 37 = 64 ✓ ; 20 + 37 = 57 grey ✓. Over the 37:
α_s = 1.42 ± 0.09, α_t = 1.55 ± 0.10, crackling deviation 4.1% ± 2.4% — matches
paper.tex §3.3 exactly.

### Rejected alternative (the σ-gated 27)

An earlier paper.md draft additionally required σ ∈ [0.85, 1.15], yielding a
27-population subset (all 27 ⊂ the 37) and a different funnel framed as
89/55/11/27. That funnel was both non-canonical and arithmetically inconsistent
(55 + 11 + 27 = 93 ≠ 89, because the "55 σ-off" spanned 46 grey + 9 fibre that
the "11 fibre" then double-counted). It is **superseded** by the σ-free 37 above.
Do not use the 89/55/11/27 funnel.

### Surrogate negative control (already in paper.tex §3.3)

Applying the identical crackling criterion to surrogates: 65% of real
grey-matter populations pass, versus 22.5% ± 1.3% of ISI shuffles and
6.3% ± 1.4% of rate-matched Poisson (from `datasets/steinmetz_null_summary.csv`).
The Wilting–Priesemann branching ratio does **not** separate the conditions
(m̂ ≈ 0.93 data, ≈1.00 Poisson, ≈0.92 shuffle), confirming it is uninformative
as a criterion. This is the discriminative evidence; the within-set exponent
consistency is a descriptor of the retained 37, not independent proof.

## Per-population BCa CIs (B2)

`datasets/steinmetz_table3_ci.py` is σ-free and attaches 95% BCa intervals
(n=2000 avalanche resamples) for the **37** crackling-consistent populations to
`datasets/steinmetz_results.csv` (tau/alpha/gfit lo–hi). The intervals are
narrow (τ half-width ≈ 0.006) — sampling precision only — and an order of
magnitude below the estimator's ±0.24 synthetic-truth accuracy floor (Table 1).

## B1 — surrogate-null reuse (no rerun)

`datasets/steinmetz_null_summary.csv` (real 0.649 / isi 0.225 / poisson 0.063
crackling-pass) is reused unchanged: the rigor-hardening code edits (autonomy
OLS F-test, adaptive binning option, gamma_fit bootstrap) do not touch the
criticality crackling-pass path that produced these numbers.

## Release

The hardening pass ships under the existing **1.1.0** release (no version bump);
the 12-case benchmark suite overwrites `results/benchmarks/v1.1.0.csv` in place.
