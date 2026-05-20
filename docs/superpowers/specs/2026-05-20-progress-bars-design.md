# Progress Bars — Design Spec

**Date:** 2026-05-20
**Sub-project:** tqdm progress (#6 from architectural critique)
**Status:** Approved by user 2026-05-20

## Goal

Show progress bars on long-running loops so interactive users (notebooks, REPL) get immediate feedback, without polluting log output in scripts/CI. Single global opt-in switch.

## Non-Goals

- ETA accuracy guarantees beyond what `tqdm` provides out of the box.
- Per-call kwarg overrides (deliberately rejected to keep API surface minimal).
- Auto-detection of TTY/Jupyter (deliberately rejected — explicit opt-in only).

## Approach

### Module: `neurocomplexity/_progress.py`

```python
from __future__ import annotations
from typing import Iterable, Optional

_enabled: bool = False

def set_progress(enabled: bool) -> None:
    """Enable or disable progress bars for all subsequent long-running loops."""
    global _enabled
    _enabled = bool(enabled)

def progress_iter(iterable: Iterable, *, total: Optional[int] = None,
                  desc: Optional[str] = None) -> Iterable:
    """Wrap iterable with tqdm if progress is enabled; pass through otherwise."""
    if not _enabled:
        return iterable
    from tqdm.auto import tqdm
    return tqdm(iterable, total=total, desc=desc, leave=False)
```

Exported as `nc.set_progress` from `neurocomplexity/__init__.py`. The `progress_iter` helper is private (`_progress.progress_iter`).

`leave=False` means bars vanish on completion, keeping notebook output clean when many analyses run in series.

### Dependency

Add `tqdm>=4.65` to the **base** install requires (it's ~80kB, pure Python, no transitive deps worth worrying about). Update `requirements.txt` and `pyproject.toml`.

## Insertion sites

| File | Loop | `desc` | `total` |
|------|------|--------|---------|
| `inference/null_test.py` | `for i in range(n_resamples): ... pool.draw(i) ...` | `"null replicates"` | `n_resamples` |
| `inference/bootstrap.py` | `for b in range(n_resamples): ...` | `"bootstrap"` | `n_resamples` |
| `analysis/transfer_entropy.py` | pairwise `(src, tgt)` loop | `"TE matrix"` | `n_pairs` |
| `analysis/autonomy.py` | pairwise loop | `"autonomy"` | `n_pairs` |
| `analysis/pid.py` | triples loop | `"PID atoms"` | `n_triples` |

For nested loops (TE pairwise inside null_test), the inner bar still uses `leave=False`, so the outer bar redraws correctly.

## Tests (`tests/test_progress.py`)

| Test | Behaviour |
|------|-----------|
| `test_default_disabled` | After fresh import, `_progress._enabled is False`; `progress_iter([1,2,3])` returns the list unchanged (same object). |
| `test_set_progress_enables_tqdm_wrap` | `nc.set_progress(True)`; `progress_iter([1,2,3])` returns a `tqdm` instance (`hasattr(..., 'update')`). |
| `test_set_progress_disable_restores_passthrough` | Enable then disable; iterable returned unchanged. |
| `test_iteration_count_correct_when_enabled` | Enable; consume `progress_iter(range(5))` into a list — values intact. |
| `test_set_progress_idempotent` | Calling `set_progress(True)` twice does not raise or duplicate state. |

Per-site smoke tests (in existing test files) confirm the loops still complete identically with progress on vs off — the analysis must produce identical numeric results.

## API surface

```python
import neurocomplexity as nc
nc.set_progress(True)   # session-wide
# ... run analyses, see bars ...
nc.set_progress(False)  # quiet again
```

That's it. Zero per-call kwargs added anywhere.

## File layout

```
neurocomplexity/_progress.py            # new
neurocomplexity/__init__.py             # export set_progress
neurocomplexity/inference/null_test.py  # wrap replicate loop
neurocomplexity/inference/bootstrap.py  # wrap replicate loop
neurocomplexity/analysis/transfer_entropy.py  # wrap pairwise
neurocomplexity/analysis/autonomy.py    # wrap pairwise
neurocomplexity/analysis/pid.py         # wrap triples
requirements.txt                        # + tqdm>=4.65
pyproject.toml                          # + tqdm>=4.65 in dependencies
tests/test_progress.py                  # new
```
