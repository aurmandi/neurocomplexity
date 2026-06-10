# Steinmetz funnel reconciliation + surrogate-null reuse (2026-06-10)

## B4 — funnel 89/55/11/27 is arithmetically inconsistent

Paper §3.3 currently presents the population funnel as:

> Of the **89** populations with a stationary 200-s epoch, **55** are clearly
> sub-/super-critical (σ ∉ [0.85, 1.15]) and a further **11** are fibre tracts
> excluded a priori; only the remaining **27** grey-matter populations meet the
> criticality criteria.

This does not partition the 89: **55 + 11 + 27 = 93 ≠ 89.**

Root cause (verified against `datasets/steinmetz_results.csv`, 89 rows):

- The **55** = *all* populations with σ ∉ [0.85, 1.15], spanning both tissue
  classes: **46 grey-matter + 9 fibre**.
- The **11** fibre tracts (`ar, ccb, ccg, dhc, em, fi, fp, fr, int, or`;
  10 unique acronyms, 11 rows) overlap the 55 — **9** of them are also σ-off,
  so those 9 are counted twice.
- The **5** grey-matter populations that are σ-in-band but fail the shape /
  crackling-noise criteria are silently dropped.

Net arithmetic error = +4 = (9 double-counted) − (5 omitted).

### Correct disjoint funnel (exclude tissue axis first, then σ, then shape)

| Stage | Count |
|------|------:|
| Stationary 200-s epoch | **89** |
| − fibre tracts (a priori non-neural) | 11 |
| Grey-matter populations | **78** |
| — sub-/super-critical (σ ∉ [0.85, 1.15]) | 46 |
| — σ-in-band but fail shape/CSN criteria | 5 |
| — **pass criticality (Table 3)** | **27** |

Check: 11 + 46 + 5 + 27 = 89 ✓ ; 46 + 5 + 27 = 78 grey ✓.

Cross-tab (branching σ vs tissue, from CSV):

```
              σ-off   σ-in   total
fibre            9      2      11
grey            46     32      78
total           55     34      89
grey σ-in pass = 27 ; grey σ-in fail = 5
```

**Action for C1 (deferred manuscript edit):** rewrite the §3.3 funnel to the
disjoint form above. Recommended wording: exclude the 11 fibre tracts first
(78 grey-matter), then report 46 sub-/super-critical + 5 in-band shape failures
+ 27 passing. Do not present "55" and "11" as additive stages.

## B1 — surrogate-null reuse (no rerun)

`datasets/steinmetz_null_summary.csv` (real 0.649 / isi_shuffle 0.225 /
poisson 0.063 crackling-pass; n_pop = 57) is **reused unchanged**. Rationale:
the rigor-hardening code edits (autonomy OLS F-test, adaptive binning option,
gamma_fit in the avalanche bootstrap) do **not** touch the criticality
crackling-pass path that produced these surrogate numbers — the default 4-ms
single-bin criticality pipeline is byte-for-byte unchanged. Re-running the
multi-hour surrogate sweep would reproduce the same figures within Monte-Carlo
noise, so it is deferred, not repeated.

> Note: the surrogate summary's `n_pop = 57` is a *different* denominator from
> the §3.3 funnel's 89 (it counts the criticality-screened populations entering
> the surrogate comparison, not all stationary epochs). C1/C2 should state which
> denominator each percentage uses to avoid a second apparent inconsistency.
