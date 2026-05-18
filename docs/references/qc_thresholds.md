# QC Threshold Reference

`add_quality` infers a categorical `quality` column when the source format does
not provide one explicitly. The thresholds below match Allen Institute defaults.

## Bombcell

`quality` is taken directly from Bombcell's `unitType`:

| `unitType` | `quality` |
|---|---|
| 1 | `good` |
| 2 | `mua` |
| 0 | `noise` |

## ecephys_spike_sorting

`quality` is inferred per unit:

| Condition | Resulting `quality` |
|---|---|
| `firing_rate < 0.1` Hz | `noise` |
| `isi_viol < 0.5` **AND** `amplitude_cutoff < 0.1` **AND** `presence_ratio > 0.9` | `good` |
| otherwise | `mua` |

## SpikeInterface

Same thresholds as ecephys, applied to SI column names
(`isi_violations_ratio`, `amplitude_cutoff`, `presence_ratio`, `firing_rate`).

## Source of thresholds

These thresholds reflect the most commonly cited defaults in the Neuropixels
analysis literature (Siegle et al. 2021; IBL Brain Wide Map). They are not
universal — labs whose pipelines use different cutoffs should pass their QC
table through `add_quality` with `format=` explicit, then refine selection
with `rec.filter_units(...)` using their own criteria.

## Tested versions

- Bombcell: column schema stable since v1.2.
- ecephys_spike_sorting: column schema stable since v0.2.
- SpikeInterface: column schema stable since v0.97.
