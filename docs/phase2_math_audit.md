# Phase 2 — Mathematical Correctness Audit

Estimator-by-estimator audit of every numerical primitive in `neurocomplexity`.
Each section: canonical formula from the primary reference, code citation,
toy-case verification, verdict.

Status legend: ✅ pass · ⚠ note (correct but needs caveat) · ❌ bug.

| § | Estimator | File | Verdict |
|---|---|---|---|
| 1 | Participation ratio | `analysis/dimensionality.py:52-58` | ✅ |
| 2 | LMC complexity `C = H · D` | `analysis/complexity.py:88-137` | ⚠ |
| 3 | Phipson-Smyth +1 p-value | `inference/null_test.py:19-64` | ✅ |
| 4 | Benjamini-Hochberg FDR | `inference/null_test.py:95-107` | ✅ |
| 5 | Effect size (z-score) | `inference/null_test.py:67-92` | ✅ |
| 6 | Sample entropy (SampEn) | `analysis/mse.py` | — pending |
| 7 | MSE coarse-graining | `analysis/mse.py` | — pending |
| 8 | Schreiber TE | `analysis/transfer_entropy.py` | — pending |
| 9 | Williams-Beer I_min PID | `analysis/pid.py` | — pending |
| 10 | Critical branching-process simulator (NEW) | — | — pending |
| 11 | Wilting-Priesemann MR | `analysis/branching.py` | — pending |
| 12 | Sethna gamma identity | `analysis/criticality.py` | — pending |

---

## § 1 — Participation ratio (Cunningham & Yu 2014; Gao & Ganguli 2015) — ✅

### Canonical form

For eigenvalues `λ₁, …, λ_N` of an empirical correlation (or covariance) matrix:

```
PR = (Σ λᵢ)² / Σ λᵢ²        bounded in [1, N]
```

- `PR = N` when all `λᵢ` equal (isotropic / full-rank uniform spectrum)
- `PR = 1` when a single mode dominates

### Code citation

`neurocomplexity/analysis/dimensionality.py:52-58`

```python
def _participation_ratio(eig: np.ndarray) -> float:
    eig = np.asarray(eig, dtype=np.float64)
    eig = eig[eig > 0]
    if eig.size == 0:
        return float("nan")
    s = eig.sum()
    return float((s * s) / np.sum(eig * eig))
```

### Verification

- Identity-correlation toy (N = 5, 20, 100): all `λᵢ = 1` → PR = (N·1)²/(N·1²) = N. **Pass** (Phase 1 test 6).
- One-dominant-mode toy (λ = [10, 1e-6, …]): PR ≈ 1. **Pass** (Phase 1 test 7).
- Hypothesis property: PR ∈ [1, n] for random positive eigenvalues. **Pass** (Phase 1 test 21).

### Notes

The function operates on **correlation-matrix** eigenvalues (per-unit z-scoring
applied before `np.cov`, see lines 109–111). This is current best practice
(Stringer et al. 2019 *Nat Neurosci*) — without per-unit normalisation, a single
high-firing-rate unit can artificially collapse the dimensionality estimate.
The docstring explicitly mentions "correlation matrix" — no documentation gap.

### Verdict

**Pass.** Implementation matches canonical form. No discrepancy.

---

## § 2 — LMC complexity `C = H · D` (López-Ruiz, Mancini, Calbet 1995) — ⚠

### Canonical form

For a discrete probability distribution `p = (p₁, …, p_N)` over `N` states:

```
H = -Σ pᵢ ln pᵢ                  (Shannon entropy, nats)
H_normalised = H / ln(N)         ∈ [0, 1]
D = Σ (pᵢ - 1/N)²                (Euclidean² distance from uniform)
C = H_normalised · D             (statistical complexity)
```

Asymptotic extremes:

- Uniform `pᵢ = 1/N` → `H = 1`, `D = 0`, `C = 0`.
- Delta `p₁ = 1, p_{i≠1} = 0` → `H = 0`, `D = (1 - 1/N)² + (N-1)/N²`, `C = 0`.

Modern normalisation (`H/ln N`) is from Anteneodo & Plastino 1996, Martin et al.
2006; widely accepted as the canonical post-1995 form.

### Code citation

`neurocomplexity/analysis/complexity.py:88-137`

```python
def _shannon_entropy_counts(counts: np.ndarray) -> float:
    # ...
    H = -np.sum(p[nz] * np.log(p[nz]))
    return float(H / np.log(N))

def _lmc_disequilibrium(counts: np.ndarray) -> float:
    # ...
    return float(np.sum((p - 1.0 / N) ** 2))

def _hdc_from_count_series(series: np.ndarray) -> tuple[float, float, float, int]:
    series = np.asarray(series, dtype=np.int64)
    if series.size == 0:
        return 0.0, 0.0, 0.0, 0
    max_count = int(series.max())
    edges = np.arange(max_count + 2)
    counts, _ = np.histogram(series, bins=edges)
    H = _shannon_entropy_counts(counts)
    D = _lmc_disequilibrium(counts)
    return H, D, float(H * D), int(counts.size)
```

### Verification

- Uniform 100-bin toy: `C ≈ 0`. **Pass** (Phase 1 test 9).
- Delta toy: `C ≈ 0`. **Pass** (Phase 1 test 10).
- Per-population non-negativity: **Pass** (Phase 1 test 8).

### Caveat (the ⚠)

**State-space size `N` is data-dependent** — `N = max(counts) + 1` per population
(`_hdc_from_count_series` line 131). This means:

- A population with peak count 5 in any single bin has `N = 6` discrete states.
- A population with peak count 50 has `N = 51` discrete states.
- Their `H_normalised` values are *not directly comparable* because the
  normalisation base differs.

This is **documented** (`n_states_per_pop` field; docstring lines 56–58), but it
is a foot-gun. A user comparing C across populations of different mean firing
rate will silently be comparing two different normalisations.

### Recommendation for Phase 4 (reviewer panel)

Either: (a) accept the design as-is and add a prominent warning to the
docstring / `docs/complexity_measures.md` page; (b) add an optional
`n_states` argument that fixes the state space across populations (e.g., to
`global_max + 1`); (c) deprecate per-population state space.

### Verdict

**Pass with caveat.** Formula correct, normalisation correct, extremes correct.
Cross-population comparability is a known design trade-off.

---

## § 3 — Phipson-Smyth +1 p-value (Phipson & Smyth 2010) — ✅

### Canonical form

For a permutation null with `n` resamples and `b = #{null ≥ observed}` (greater
tail):

```
p̂ = (1 + b) / (1 + n)               minimum 1/(1+n), never zero
```

For two-sided: `p̂_two_sided = min(1, 2 · min(p̂_greater, p̂_less))`. This is
robust to skewed null distributions; the earlier `|null - μ| ≥ |obs - μ|`
formulation under-powers under asymmetric nulls and is no longer recommended.

### Code citation

`neurocomplexity/inference/null_test.py:19-64`

```python
def _p_greater():
    ...
    return (1.0 + ge) / (1.0 + n)

def _p_less():
    ...
    return (1.0 + le) / (1.0 + n)

if alternative == "two-sided":
    pg = _p_greater()
    pl = _p_less()
    return np.minimum(1.0, 2.0 * np.minimum(pg, pl))
```

### Verification

- Floor invariant `p > 0` even for extreme observed: **Pass** (Phase 1 test 17).
- Array case `p ∈ (0, 1]`: **Pass** (Phase 1 test 18).
- Hypothesis property: `pvalue_from_null` ∈ (0, 1] across all alternatives:
  **Pass** (Phase 1 test 20).

### Verdict

**Pass.** Matches Phipson & Smyth (2010) Section 2 exactly.

---

## § 4 — Benjamini-Hochberg FDR (Benjamini & Hochberg 1995) — ✅

### Canonical form

Given `m` p-values, sort ascending `p₍₁₎ ≤ p₍₂₎ ≤ … ≤ p₍ₘ₎`. The step-up
adjusted q-value at rank `k` is:

```
q̃₍ₖ₎ = p₍ₖ₎ · m / k
q₍ₖ₎ = min{ q̃₍ⱼ₎ : j ≥ k }       (monotone non-decreasing in k)
```

with each `q` clipped to `[0, 1]`.

### Code citation

`neurocomplexity/inference/null_test.py:95-107`

```python
def fdr_bh(p):
    p = np.asarray(p, dtype=float)
    shape = p.shape
    flat = p.ravel()
    m = flat.size
    order = np.argsort(flat)
    ranked = flat[order]
    q = ranked * m / (np.arange(m) + 1)          # q̃₍ₖ₎ = p₍ₖ₎ · m / k
    q = np.minimum.accumulate(q[::-1])[::-1]      # right-to-left running min
    out = np.empty_like(flat)
    out[order] = np.clip(q, 0.0, 1.0)
    return out.reshape(shape)
```

The right-to-left running min `np.minimum.accumulate(q[::-1])[::-1]` computes,
for each `k`, `min{q̃₍ⱼ₎ : j ≥ k}` — the monotone step-up adjustment.

### Verification

- FDR `≥` raw p elementwise: **Pass** (Phase 1 test 19).

### Notes

- NaN handling: NaN p-values pass through unchanged (NaN compares false in
  `np.minimum.accumulate`); current behaviour is silent pass-through. Acceptable
  for now; flag for the Phase 4 reviewer panel whether to raise instead.
- Family definition is **global across the flattened array**. For a TE matrix
  this means FDR is computed across all `N²` directed pairs simultaneously.
  This is the conservative and correct choice; documented behaviour.

### Verdict

**Pass.** Matches Benjamini & Hochberg (1995) procedure.

---

## § 5 — Effect size (z-score against null) — ✅

### Canonical form

```
z = (observed - mean(null)) / sd(null)         sd uses ddof = 1
```

NaN where `sd = 0` (rather than `±inf`).

### Code citation

`neurocomplexity/inference/null_test.py:67-92`

```python
def effect_size(observed, null):
    null = np.asarray(null)
    mu = np.nanmean(null, axis=0)
    sd = np.nanstd(null, axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (observed - mu) / sd
    return np.where(sd > 0, z, np.nan)
```

### Verdict

**Pass.** `ddof = 1` (sample SD) — correct for finite permutation samples.
NaN-safe along the resample axis.

---

---

## § 6 — Sample entropy (Richman & Moorman 2000) — ✅

### Canonical form

For a 1-D series `x` of length `N`, template length `m`, tolerance `r`,
common-range `K = N − m`:

```
B = #{(i, j) : 0 ≤ i < j < K, ‖x[i:i+m]  − x[j:j+m]‖_∞   ≤ r}   (no self-matches)
A = #{(i, j) : 0 ≤ i < j < K, ‖x[i:i+m+1] − x[j:j+m+1]‖_∞ ≤ r}
SampEn(m, r, N) = −ln(A / B),    NaN if A = 0 or B = 0
```

The **common range** `K = N − m` (both `m`- and `(m+1)`-templates iterate the
same `K` starting indices) is the canonical Richman & Moorman convention,
distinguishing SampEn from approximate entropy (ApEn).

### Code citation

`neurocomplexity/analysis/mse.py:75-105`

```python
def _sample_entropy(x: np.ndarray, m: int, r: float) -> float:
    N = x.size
    if N < m + 2: return float("nan")
    K = N - m
    if K < 2:    return float("nan")
    from numpy.lib.stride_tricks import sliding_window_view

    def _count_matches(length: int) -> int:
        windows = sliding_window_view(x, length)[:K]   # common range K
        count = 0
        for i in range(K - 1):
            d = np.max(np.abs(windows[i + 1:] - windows[i]), axis=1)  # Chebyshev
            count += int(np.count_nonzero(d <= r))                    # j > i
        return count

    B = _count_matches(m)
    A = _count_matches(m + 1)
    if B == 0 or A == 0: return float("nan")
    return float(-np.log(A / B))
```

- `K = N − m` ✓
- `[:K]` slice on both window lengths enforces common range ✓
- Chebyshev (`np.max(np.abs(...))`) ✓
- `j = i + 1 .. K − 1` excludes self-matches ✓
- `−ln(A/B)` ✓
- NaN on degenerate match counts ✓

### Verification

- Identity with direct `_sample_entropy` call at scale 1: **Pass** (Phase 1
  test 11).

### Verdict

**Pass.** Matches Richman & Moorman (2000) Eqs. 2–6 exactly.

---

## § 7 — MSE coarse-graining + tolerance default (Costa, Goldberger, Peng 2002) — ⚠

### Canonical form

Coarse-graining at scale `τ` (Costa 2002 Eq. 1):

```
y_j^(τ) = (1/τ) · Σ_{i = (j−1)τ + 1}^{jτ} x_i,     j = 1, …, ⌊N / τ⌋
```

Tolerance `r = 0.15 · σ_X` where `σ_X` is the SD of the **original**
(uncoarse-grained) series.

### Code citation

`neurocomplexity/analysis/mse.py:64-72` (coarse-grain) and
`neurocomplexity/analysis/mse.py:211` (tolerance):

```python
def _coarse_grain(x: np.ndarray, scale: int) -> np.ndarray:
    n = x.size // scale
    if n == 0: return np.empty(0, dtype=np.float64)
    return x[:n * scale].reshape(n, scale).mean(axis=1)

# in multiscale_entropy(...):
r = r_factor * float(series.std(ddof=0))   # SD of the original, not coarse-grained
```

Coarse-graining matches Costa 2002 Eq. 1 exactly. Tolerance uses the original
series SD (verified by inspection of the loop structure: `r` is computed once
per population, before the scale loop).

### Discrepancy (the ⚠)

Default `r_factor = 0.2`. The docstring (line 144) reads:

> ``r_factor : Tolerance factor for sample entropy (default 0.2 — Costa 2002).``

Costa, Goldberger, Peng (2002 *PRL* 89:068102) §III explicitly used **`r = 0.15 · σ`**, not 0.2.
The value 0.2 originates from Pincus (1991, ApEn) and is commonly cited but is
**not** the Costa MSE default.

Severity: not a bug — `0.2` is a defensible literature value, and MSE results
move smoothly in `r`. But the docstring **misattributes the default**.

### Fix recommendation (for Phase 4 reviewer panel to decide)

Either:

- **(a) Change the default to 0.15** to match Costa canonical, accept the
  user-visible reproducibility change (CHANGELOG note required).
- **(b) Keep `0.2` but correct the citation** to read e.g.
  `"default 0.2 (Pincus 1991 ApEn convention); Costa 2002 MSE used 0.15"`,
  and add a paragraph to `docs/complexity_measures.md` explaining the
  trade-off.

I lean toward (b) for v1.x and (a) for v2.0 if any breaking change is being
made anyway.

### Verdict

**Pass with documentation discrepancy.** Math is correct; default value is
defensible but citation is misleading.

---

## § 8 — Schreiber TE on binary spike trains + Miller-Madow (Schreiber 2000) — ⚠

### Canonical form

```
TE(X → Y) = Σ_{y_{t+1}, y_t, x_t}  p(y_{t+1}, y_t, x_t)
                                  · log[ p(y_{t+1} | y_t, x_t) / p(y_{t+1} | y_t) ]
```

For binary X, Y with history `k = l = 1`: 8-state joint distribution over
`(y_future, y_past, x_past) ∈ {0, 1}³`. In nats when log = ln. Non-negative.
TE(X → X) = 0 trivially.

Miller-Madow plug-in entropy bias (Miller 1955):

```
H_MM = H_plug + (m − 1) / (2N),     m = number of nonzero outcomes
```

For TE, the simplest practical bias correction subtracts `(m_joint − 1)/(2N)`
once at the end, where `m_joint` is the number of populated 8-cells. This is
the convention used by `pyinform` and most binary-TE implementations; it is a
**simplified** version of the principled correction
`(m_{xy} − 1)(m_{xy} − m_x − m_y + 1)/(2N)` (Roulston 1999).

### Code citation

`neurocomplexity/analysis/transfer_entropy.py:58-98`

```python
def _binary_schreiber_te(source_ts, target_ts, delay=1) -> float:
    y = (target_ts > 0).astype(np.int8)
    x = (source_ts > 0).astype(np.int8)
    y_future = y[delay:]; y_past = y[:-delay]; x_past = x[:-delay]
    N = len(y_future)

    idx = 4 * y_future + 2 * y_past + x_past
    counts = np.bincount(idx, minlength=8).astype(np.float64)
    p_joint = counts / N

    te = 0.0
    for yf, yp, xp in itertools.product((0,1), repeat=3):
        i = 4*yf + 2*yp + xp
        if counts[i] == 0: continue
        p_yyx = p_joint[i]                              # p(y_f, y_p, x_p)
        p_yx  = p_joint[4*0+2*yp+xp] + p_joint[4*1+2*yp+xp]  # p(y_p, x_p)
        p_yy  = p_joint[4*yf+2*yp+0] + p_joint[4*yf+2*yp+1]  # p(y_f, y_p)
        p_y   = Σ over (y_f, x_p) of p_joint[...]            # p(y_p)
        cond_full = p_yyx / p_yx     # p(y_f | y_p, x_p)
        cond_red  = p_yy / p_y       # p(y_f | y_p)
        te += p_yyx * np.log(cond_full / cond_red)

    m = int(np.sum(counts > 0))
    te -= (m - 1) / (2.0 * N)
    return max(0.0, float(te))
```

All marginalisations correct: `p(y_p, x_p)`, `p(y_f, y_p)`, `p(y_p)` each sum
the joint over the unmissing axis ✓. Result in nats ✓. Floors at 0 ✓.

### Discrepancy (the ⚠)

The Miller-Madow correction `(m_joint − 1) / (2N)` is the **simplified
practitioner's** version, not the principled Roulston correction for
conditional MI. For modest sample sizes (N ≳ 10⁴, m ≈ 5–8 of 8 cells filled)
the difference is small (~0.01 nats); for very small N or sparsely populated
joints it can matter.

### Verification

- TE matrix diagonal = 0: **Pass** (Phase 1 test 14).
- Non-negative: **Pass** (Phase 1 test 15).
- Row=source convention on a deterministic driver→follower: **Pass** (Phase 1
  test 16, `TE[a→b] ≫ TE[b→a]`).

### Recommendation (Phase 4 reviewer panel)

Either accept the simplified MM (and document the choice), or upgrade to the
Roulston formula. Add the Kraskov k-NN estimator as a non-binary alternative
for future versions (the code already has the hook: `estimator="binary"` is
the only implemented option, line 117).

### Verdict

**Pass with documentation note.** Math matches Schreiber (2000). MM correction
is a defensible simplification.

---

## § 9 — Williams-Beer I_min PID (Williams & Beer 2010) — ⚠

### Canonical form

Specific information about target outcome `y` from source `S_k`:

```
I(Y = y ; S_k) = Σ_s p(s | y) · log[ p(s | y) / p(s) ]
```

I_min redundancy:

```
R(Y ; S_1, S_2) = Σ_y p(y) · min_k I(Y = y ; S_k)
```

Bivariate-PID atoms (Williams-Beer lattice):

```
R = I_min(Y ; S_1, S_2)
U_1 = I(Y ; S_1) − R
U_2 = I(Y ; S_2) − R
S = I(Y ; (S_1, S_2)) − U_1 − U_2 − R
```

All four atoms `≥ 0` and `R + U_1 + U_2 + S = I(Y ; (S_1, S_2))`.

### Code citation

`neurocomplexity/analysis/pid.py:135-168` — specific info and `_redundancy_imin`:

```python
def _specific_info(target_val, src_axis, joint) -> float:
    j = joint / joint.sum()
    p_y = j.sum(axis=(1, 2))
    if p_y[target_val] == 0: return 0.0
    other = 3 - src_axis     # 1<->2
    p_sy = j.sum(axis=other)             # (L_y, L_s)
    p_s = p_sy.sum(axis=0)               # (L_s,)
    out = 0.0
    p_yv = p_y[target_val]
    for sv in range(p_sy.shape[1]):
        p_sv_yv = p_sy[target_val, sv]
        if p_sv_yv == 0: continue
        p_sv = p_s[sv]
        if p_sv == 0: continue
        p_s_given_y = p_sv_yv / p_yv
        out += p_s_given_y * (np.log(1.0 / p_sv) - np.log(1.0 / p_s_given_y))
    return float(out)

def _redundancy_imin(joint) -> float:
    p_y = (joint / joint.sum()).sum(axis=(1, 2))
    r = 0.0
    for yv in range(p_y.size):
        if p_y[yv] == 0: continue
        i1 = _specific_info(yv, 1, joint)
        i2 = _specific_info(yv, 2, joint)
        r += p_y[yv] * min(i1, i2)        # I_min definition
    return float(max(r, 0.0))
```

Atom computation (lines 246-253):

```python
redundancy = _redundancy_imin(joint)
redundancy = min(redundancy, i_y_s1, i_y_s2)         # floor protection vs. MM bias
unique_1 = max(0.0, i_y_s1 - redundancy)
unique_2 = max(0.0, i_y_s2 - redundancy)
synergy  = max(0.0, total_mi - redundancy - unique_1 - unique_2)
```

All MI terms use `_mi_mm` (Miller-Madow corrected). Quantile-equal
discretisation to `n_levels = 3` by default avoids binary-saturation
collapse on busy populations.

### Verification

- All four atoms ≥ 0: **Pass** (Phase 1 test 12).
- Sum equals total MI: **Pass** (Phase 1 test 13).
- Canonical-case benchmarks (`benchmarks/cases/pid.py`): XOR (S = ln 2), AND
  (R ≈ 0.216, S ≈ 0.347), RDN (R = ln 2), UNQ (U_1 = ln 2) all pass within
  tolerance 0.10 nats.

### Discrepancies (the ⚠)

Two structural concerns, neither is a bug:

1. **Benchmark tolerance is loose.** `benchmarks/cases/pid.py` accepts a
   per-rep max error of 0.10 nats. For AND, expected S = 0.347 nats; an error
   of 0.10 is ~29 % relative. With 20,000 bins, a tighter tolerance (~0.03)
   should be achievable. Recommend tightening in Phase 4.
2. **I_min is known to over-estimate redundancy.** Williams-Beer's
   `min(I(y; s_1), I(y; s_2))` averaged over `y` is non-negative but does not
   distinguish *same-content* redundancy from *same-magnitude* coincidence.
   Bertschinger et al. (2014) proposed `I_BROJA` to fix this. We use I_min as
   the historical / default choice; this should be documented as a known
   limitation.

### Recommendation (Phase 4)

- Tighten the PID benchmark tolerance.
- Add a paragraph to `docs/` (probably `docs/complexity_measures.md` or a new
  `docs/information_decomposition.md`) explaining I_min's redundancy
  overestimation and pointing readers at I_BROJA / I_PM for future versions.

### Verdict

**Pass.** Matches Williams & Beer (2010) `I_min` PID lattice. Known
historical-choice limitations to document.

---

## § 10–12 — Criticality cluster (pending)

Still to audit: critical branching-process simulator (NEW fixture), Wilting MR
multi-step regression, Sethna gamma identity. Each requires building the
simulator first; tracked as the next Phase 2 task.

Updates appended below this line.

