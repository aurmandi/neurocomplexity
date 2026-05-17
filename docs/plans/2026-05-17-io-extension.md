# IO Extension Implementation Plan (v1.1.0)

**Goal:** Add `from_phy`, `from_kilosort`, and `from_spikeinterface` loaders to `neurocomplexity.io`, plus tests, docs, and packaging support.

**Architecture:** Two new file-based loaders (`phy.py`, `kilosort.py`) thin-wrap a shared internal helper (`_sorter_output.py`) that handles `params.py` parsing, `.npy` loading, and TSV normalization. A separate `spikeinterface.py` lazy-imports SpikeInterface and accepts any `BaseSorting`. All four heavy loaders are lazy-imported via `io/__init__.py` `__getattr__` so the package stays light.

**Tech Stack:** numpy, pandas, pytest. `spikeinterface` and `pynwb` are optional extras with `pytest.importorskip` gates in tests.

---

## File map

| File | Status | Responsibility |
|---|---|---|
| `neurocomplexity/io/__init__.py` | modify | Lazy `__getattr__` exposing all four loaders |
| `neurocomplexity/io/_sorter_output.py` | create | Shared sorter-directory helper |
| `neurocomplexity/io/phy.py` | create | `from_phy` wrapper |
| `neurocomplexity/io/kilosort.py` | create | `from_kilosort` wrapper |
| `neurocomplexity/io/spikeinterface.py` | create | `from_spikeinterface` adapter |
| `tests/test_io_phy.py` | create | Phy fixture + roundtrip tests |
| `tests/test_io_kilosort.py` | create | Kilosort fixture + roundtrip tests |
| `tests/test_io_spikeinterface.py` | create | SI adapter tests (gated) |
| `tests/_sorter_fixtures.py` | create | Shared synthetic-sorter-dir builder |
| `pyproject.toml` | modify | Add `[spikeinterface]` extra; bump version to 1.1.0 |
| `neurocomplexity/_version.py` | modify | Bump to 1.1.0 |
| `docs/quickstart.md` | modify | Real Phy / Kilosort snippets replacing the v1.1 placeholder |
| `docs/installation.md` | modify | Add `[spikeinterface]` extra to extras table |
| `docs/io.md` | create | Loader reference + decision flowchart |
| `docs/index.md` | modify | Add `io` to toctree |
| `CHANGELOG.md` | modify | v1.1.0 section |

---

## Task 1 — `[spikeinterface]` extra and lazy `__getattr__` scaffold

**Files:**
- Modify: `pyproject.toml`
- Modify: `neurocomplexity/io/__init__.py`

- [ ] **Step 1: Add the extra**

In `pyproject.toml`, in `[project.optional-dependencies]`, insert after the `nwb = ...` line:

```toml
spikeinterface = ["spikeinterface>=0.100"]
```

- [ ] **Step 2: Extend lazy `__getattr__`**

Replace the body of `neurocomplexity/io/__init__.py` with:

```python
"""I/O loaders that materialise a ``SpikeRecording`` from disk or memory.

Heavy loaders are lazy-imported so installing without the corresponding
extra does not pull in pynwb / spikeinterface / hdmf. Pure-NumPy loaders
import eagerly.
"""
from __future__ import annotations

from neurocomplexity.io.dict_loader import from_dict

__all__ = [
    "from_dict",
    "from_nwb",
    "from_phy",
    "from_kilosort",
    "from_spikeinterface",
]


def __getattr__(name):
    if name == "from_nwb":
        from neurocomplexity.io.nwb import from_nwb as _f
        return _f
    if name == "from_phy":
        from neurocomplexity.io.phy import from_phy as _f
        return _f
    if name == "from_kilosort":
        from neurocomplexity.io.kilosort import from_kilosort as _f
        return _f
    if name == "from_spikeinterface":
        from neurocomplexity.io.spikeinterface import from_spikeinterface as _f
        return _f
    raise AttributeError(f"module 'neurocomplexity.io' has no attribute {name!r}")
```

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml neurocomplexity/io/__init__.py
git commit -m "io: scaffold lazy loaders + add spikeinterface extra"
```

---

## Task 2 — Shared synthetic-sorter-dir fixture

**Files:**
- Create: `tests/_sorter_fixtures.py`

- [ ] **Step 1: Write the fixture helper**

Content of `tests/_sorter_fixtures.py`:

```python
"""Helpers that write a synthetic Phy/Kilosort sorter output directory."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

DEFAULT_SAMPLE_RATE = 30000.0


def write_sorter_directory(
    directory: Path,
    spike_trains_sec: Mapping[int, np.ndarray],
    *,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
    cluster_info: pd.DataFrame | None = None,
    cluster_group: pd.DataFrame | None = None,
    cluster_kslabel: pd.DataFrame | None = None,
    omit_spike_clusters: bool = False,
) -> Path:
    """Write a Phy/Kilosort-format directory at ``directory``.

    ``spike_trains_sec`` is a {cluster_id: spike_times_seconds} mapping. The
    helper converts to integer samples, builds the per-spike cluster array,
    sorts by sample index, and writes the .npy files. Any of the TSV
    arguments that are not None are written verbatim.
    """
    directory.mkdir(parents=True, exist_ok=True)

    sample_chunks: list[np.ndarray] = []
    cluster_chunks: list[np.ndarray] = []
    for cid, st in spike_trains_sec.items():
        samples = np.round(np.asarray(st, dtype=np.float64) * sample_rate).astype(np.int64)
        sample_chunks.append(samples)
        cluster_chunks.append(np.full(samples.shape, int(cid), dtype=np.int32))

    if sample_chunks:
        all_samples = np.concatenate(sample_chunks)
        all_clusters = np.concatenate(cluster_chunks)
        order = np.argsort(all_samples, kind="stable")
        all_samples = all_samples[order]
        all_clusters = all_clusters[order]
    else:
        all_samples = np.empty(0, dtype=np.int64)
        all_clusters = np.empty(0, dtype=np.int32)

    np.save(directory / "spike_times.npy", all_samples)
    if not omit_spike_clusters:
        np.save(directory / "spike_clusters.npy", all_clusters)
    # spike_templates is always written; Phy keeps it around even after merges.
    np.save(directory / "spike_templates.npy", all_clusters.astype(np.int32))

    (directory / "params.py").write_text(
        f"dat_path = 'continuous.dat'\n"
        f"n_channels_dat = 384\n"
        f"dtype = 'int16'\n"
        f"offset = 0\n"
        f"sample_rate = {sample_rate}\n"
        f"hp_filtered = True\n",
        encoding="utf-8",
    )

    if cluster_info is not None:
        cluster_info.to_csv(directory / "cluster_info.tsv", sep="\t", index=False)
    if cluster_group is not None:
        cluster_group.to_csv(directory / "cluster_group.tsv", sep="\t", index=False)
    if cluster_kslabel is not None:
        cluster_kslabel.to_csv(directory / "cluster_KSLabel.tsv", sep="\t", index=False)

    return directory
```

- [ ] **Step 2: Commit**

```bash
git add tests/_sorter_fixtures.py
git commit -m "tests: synthetic sorter-output fixture helper"
```

---

## Task 3 — `_sorter_output` helper (TDD)

**Files:**
- Create: `neurocomplexity/io/_sorter_output.py`
- Create: `tests/test_io_phy.py` (first test goes here; from_phy comes in Task 4)

- [ ] **Step 1: Write the failing roundtrip test**

Create `tests/test_io_phy.py`:

```python
"""Tests for from_phy (and the shared _sorter_output helper through it)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import neurocomplexity as nc
from tests._sorter_fixtures import write_sorter_directory, DEFAULT_SAMPLE_RATE


def _phy_cluster_info(unit_ids, groups):
    return pd.DataFrame({
        "cluster_id": list(unit_ids),
        "group": list(groups),
        "KSLabel": ["good"] * len(unit_ids),
        "Amplitude": np.linspace(50.0, 100.0, len(unit_ids)),
        "ContamPct": np.linspace(0.1, 5.0, len(unit_ids)),
        "depth": np.linspace(100.0, 500.0, len(unit_ids)),
        "ch": list(range(len(unit_ids))),
        "fr": np.linspace(1.0, 10.0, len(unit_ids)),
        "n_spikes": [5, 4, 6],
        "sh": [0] * len(unit_ids),
    })


def test_from_phy_roundtrip(tmp_path):
    trains = {
        0: np.array([0.10, 0.42, 1.07, 2.55, 3.91]),
        1: np.array([0.05, 0.93, 2.10, 4.80]),
        2: np.array([0.30, 0.70, 1.50, 2.20, 3.00, 4.10]),
    }
    info = _phy_cluster_info([0, 1, 2], ["good", "mua", "good"])
    write_sorter_directory(tmp_path, trains, cluster_info=info)

    rec = nc.io.from_phy(tmp_path)

    assert rec.n_units == 3
    assert rec.n_spikes == sum(len(v) for v in trains.values())
    assert np.all(np.diff(rec.spike_times) >= 0)

    for uid, st in trains.items():
        recovered = np.sort(rec.spike_times[rec.unit_ids == uid])
        np.testing.assert_allclose(recovered, st, atol=1.0 / DEFAULT_SAMPLE_RATE)

    # Column normalization: cluster_id -> id, group -> quality, fr -> firing_rate,
    # ch -> peak_channel, Amplitude -> amplitude, ContamPct -> contam_pct.
    cols = set(rec.units.columns)
    assert {"id", "quality", "firing_rate", "depth",
            "peak_channel", "amplitude", "contam_pct", "n_spikes"} <= cols

    assert list(rec.units.sort_values("id")["quality"]) == ["good", "mua", "good"]
    assert rec.source.source_format == "phy"
    assert rec.populations["all"].sum() == 3
```

- [ ] **Step 2: Run test, confirm it fails**

Run: `python -m pytest tests/test_io_phy.py::test_from_phy_roundtrip -v`

Expected: `ModuleNotFoundError: No module named 'neurocomplexity.io.phy'` (the lazy `__getattr__` returns a function that imports a not-yet-existing module).

- [ ] **Step 3: Implement `_sorter_output.py`**

Create `neurocomplexity/io/_sorter_output.py`:

```python
"""Shared loader for Phy / Kilosort sorter output directories."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal, Mapping

import numpy as np
import pandas as pd

from neurocomplexity.core.exceptions import RecordingValidationError
from neurocomplexity.core.provenance import ProvenanceRecord
from neurocomplexity.core.recording import SpikeRecording

LabelSource = Literal["phy", "kilosort"]

# Column normalization map. Applied after read_csv.
_RENAME = {
    "cluster_id": "id",
    "group": "quality",
    "KSLabel": "quality",
    "fr": "firing_rate",
    "ch": "peak_channel",
    "Amplitude": "amplitude",
    "ContamPct": "contam_pct",
}


def _parse_params(path: Path) -> dict:
    """Execute params.py in an empty namespace and return its globals.

    Phy/Kilosort write params.py as a small Python file. Matches the
    convention used by SpikeInterface and Phy itself. Trust assumption: the
    sorter directory is owned by the user running the loader.
    """
    if not path.exists():
        raise RecordingValidationError(f"missing params.py at {path}")
    ns: dict = {}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), ns)
    if "sample_rate" not in ns or not isinstance(ns["sample_rate"], (int, float)):
        raise RecordingValidationError(
            f"params.py at {path} missing numeric sample_rate"
        )
    return ns


def _load_label_table(directory: Path, label_source: LabelSource) -> pd.DataFrame:
    """Load the cluster label table for the requested sorter mode."""
    if label_source == "phy":
        info_path = directory / "cluster_info.tsv"
        group_path = directory / "cluster_group.tsv"
        if info_path.exists():
            df = pd.read_csv(info_path, sep="\t")
        elif group_path.exists():
            df = pd.read_csv(group_path, sep="\t")
        else:
            raise RecordingValidationError(
                f"Phy directory {directory} has neither cluster_info.tsv "
                f"nor cluster_group.tsv"
            )
    elif label_source == "kilosort":
        ks_path = directory / "cluster_KSLabel.tsv"
        if not ks_path.exists():
            raise RecordingValidationError(
                f"Kilosort directory {directory} missing cluster_KSLabel.tsv"
            )
        df = pd.read_csv(ks_path, sep="\t")
    else:
        raise ValueError(f"unknown label_source {label_source!r}")

    return df.rename(columns=_RENAME)


def _load_spike_clusters(directory: Path) -> np.ndarray:
    """Prefer spike_clusters.npy (Phy merges applied); fall back to spike_templates.npy."""
    sc = directory / "spike_clusters.npy"
    if sc.exists():
        return np.load(sc).astype(np.int64)
    st = directory / "spike_templates.npy"
    if st.exists():
        warnings.warn(
            f"{directory}: spike_clusters.npy missing; using spike_templates.npy. "
            "Any Phy merges or splits will not be reflected.",
            UserWarning, stacklevel=3,
        )
        return np.load(st).astype(np.int64)
    raise RecordingValidationError(
        f"{directory} has neither spike_clusters.npy nor spike_templates.npy"
    )


def _load_sorter_output(
    directory: Path,
    *,
    label_source: LabelSource,
    duration: float | None,
    populations: Mapping[str, np.ndarray] | None,
) -> SpikeRecording:
    directory = Path(directory)
    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(f"sorter directory not found: {directory}")

    params = _parse_params(directory / "params.py")
    sample_rate = float(params["sample_rate"])

    samples_path = directory / "spike_times.npy"
    if not samples_path.exists():
        raise RecordingValidationError(f"missing spike_times.npy at {samples_path}")
    samples = np.load(samples_path).astype(np.int64).ravel()
    if samples.size == 0:
        raise RecordingValidationError(f"{directory} contains no spikes")

    clusters = _load_spike_clusters(directory).ravel()
    if clusters.shape != samples.shape:
        raise RecordingValidationError(
            f"spike_times shape {samples.shape} != clusters shape {clusters.shape}"
        )

    order = np.argsort(samples, kind="stable")
    samples = samples[order]
    clusters = clusters[order]
    spike_times = samples.astype(np.float64) / sample_rate

    units_df = _load_label_table(directory, label_source)
    if "id" not in units_df.columns:
        raise RecordingValidationError(
            f"label table for {directory} has no cluster_id/id column"
        )
    units_df["id"] = units_df["id"].astype(np.int64)

    # Add synthetic rows for any cluster present in spike_clusters but absent
    # in the label table (preserves SpikeRecording invariant).
    known_ids = set(units_df["id"].tolist())
    extra_ids = sorted(set(int(c) for c in np.unique(clusters)) - known_ids)
    if extra_ids:
        extras = pd.DataFrame({"id": extra_ids, "quality": ["unsorted"] * len(extra_ids)})
        units_df = pd.concat([units_df, extras], ignore_index=True)

    if "quality" not in units_df.columns:
        units_df["quality"] = "unsorted"
    units_df["quality"] = units_df["quality"].fillna("unsorted")

    if duration is None:
        duration = float(spike_times.max()) + 1.0
    duration = float(duration)

    if populations is None:
        populations = {"all": np.ones(len(units_df), dtype=bool)}

    provenance = ProvenanceRecord.for_file(
        directory / "params.py", source_format=label_source,
    )

    return SpikeRecording(
        spike_times=spike_times,
        unit_ids=clusters,
        units=units_df.reset_index(drop=True),
        populations=populations,
        duration=duration,
        sampling_rate=sample_rate,
        source=provenance,
    )
```

- [ ] **Step 4: Run test again — still fails (no `from_phy` yet)**

Run: `python -m pytest tests/test_io_phy.py::test_from_phy_roundtrip -v`

Expected: still fails with `ModuleNotFoundError: No module named 'neurocomplexity.io.phy'`. That is fixed in Task 4.

- [ ] **Step 5: Commit**

```bash
git add neurocomplexity/io/_sorter_output.py tests/test_io_phy.py
git commit -m "io: shared Phy/Kilosort loader helper + first roundtrip test"
```

---

## Task 4 — `from_phy`

**Files:**
- Create: `neurocomplexity/io/phy.py`

- [ ] **Step 1: Implement `from_phy`**

Create `neurocomplexity/io/phy.py`:

```python
"""Loader for Phy curation directories.

Phy is the standard interactive curation GUI applied on top of Kilosort
output. After curation, Phy writes ``cluster_info.tsv`` with the final
``group`` column (good / mua / noise / unsorted). This loader prefers
that file and falls back to ``cluster_group.tsv`` if the user has only
the minimal export.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from neurocomplexity.core.recording import SpikeRecording
from neurocomplexity.io._sorter_output import _load_sorter_output


def from_phy(
    directory: str | Path,
    *,
    duration: float | None = None,
    populations: Mapping[str, np.ndarray] | None = None,
) -> SpikeRecording:
    """Build a SpikeRecording from a Phy curation directory.

    Parameters
    ----------
    directory
        Path containing ``spike_times.npy``, ``spike_clusters.npy``,
        ``params.py``, and ``cluster_info.tsv`` (or ``cluster_group.tsv``).
    duration
        Override the recording duration in seconds. Default is
        ``max(spike_times) + 1.0``.
    populations
        Override the default ``{"all": ones}`` population mask.

    Notes
    -----
    No quality filtering is applied at load time; call
    ``rec.filter_units(quality=['good'])`` downstream to drop MUA/noise.
    """
    return _load_sorter_output(
        Path(directory),
        label_source="phy",
        duration=duration,
        populations=populations,
    )
```

- [ ] **Step 2: Run the Task-3 test — now passes**

Run: `python -m pytest tests/test_io_phy.py::test_from_phy_roundtrip -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add neurocomplexity/io/phy.py
git commit -m "io: from_phy loader"
```

---

## Task 5 — `from_kilosort`

**Files:**
- Create: `neurocomplexity/io/kilosort.py`
- Create: `tests/test_io_kilosort.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_io_kilosort.py`:

```python
"""Tests for from_kilosort."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import neurocomplexity as nc
from tests._sorter_fixtures import write_sorter_directory, DEFAULT_SAMPLE_RATE


def test_from_kilosort_roundtrip(tmp_path):
    trains = {
        7: np.array([0.20, 0.55, 1.00]),
        8: np.array([0.10, 0.90, 1.80, 2.40]),
    }
    ks_label = pd.DataFrame({
        "cluster_id": [7, 8],
        "KSLabel": ["good", "mua"],
    })
    write_sorter_directory(tmp_path, trains, cluster_kslabel=ks_label)

    rec = nc.io.from_kilosort(tmp_path)

    assert rec.n_units == 2
    assert rec.n_spikes == sum(len(v) for v in trains.values())
    assert set(rec.units["id"]) == {7, 8}
    assert dict(zip(rec.units["id"], rec.units["quality"])) == {7: "good", 8: "mua"}
    assert rec.source.source_format == "kilosort"
    assert rec.sampling_rate == DEFAULT_SAMPLE_RATE


def test_from_kilosort_missing_kslabel_raises(tmp_path):
    trains = {0: np.array([0.1, 0.2])}
    write_sorter_directory(tmp_path, trains)  # no cluster_kslabel
    with pytest.raises(Exception, match="cluster_KSLabel.tsv"):
        nc.io.from_kilosort(tmp_path)
```

- [ ] **Step 2: Run test, confirm failure**

Run: `python -m pytest tests/test_io_kilosort.py -v`
Expected: `ModuleNotFoundError: No module named 'neurocomplexity.io.kilosort'`.

- [ ] **Step 3: Implement `from_kilosort`**

Create `neurocomplexity/io/kilosort.py`:

```python
"""Loader for raw Kilosort output directories (no Phy curation)."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from neurocomplexity.core.recording import SpikeRecording
from neurocomplexity.io._sorter_output import _load_sorter_output


def from_kilosort(
    directory: str | Path,
    *,
    duration: float | None = None,
    populations: Mapping[str, np.ndarray] | None = None,
) -> SpikeRecording:
    """Build a SpikeRecording from a raw Kilosort output directory.

    Reads automatic quality labels from ``cluster_KSLabel.tsv``. If you
    have already run Phy curation on this directory use ``from_phy``
    instead so the curated ``group`` column is used.
    """
    return _load_sorter_output(
        Path(directory),
        label_source="kilosort",
        duration=duration,
        populations=populations,
    )
```

- [ ] **Step 4: Run test — passes**

Run: `python -m pytest tests/test_io_kilosort.py -v`
Expected: PASS for both tests.

- [ ] **Step 5: Commit**

```bash
git add neurocomplexity/io/kilosort.py tests/test_io_kilosort.py
git commit -m "io: from_kilosort loader and roundtrip test"
```

---

## Task 6 — Edge-case tests for Phy fallbacks

**Files:**
- Modify: `tests/test_io_phy.py`

- [ ] **Step 1: Add fallback tests**

Append to `tests/test_io_phy.py`:

```python
def test_from_phy_uses_cluster_group_fallback(tmp_path):
    """When cluster_info.tsv is absent, cluster_group.tsv is used."""
    trains = {0: np.array([0.1, 0.3]), 1: np.array([0.2])}
    group = pd.DataFrame({"cluster_id": [0, 1], "group": ["good", "noise"]})
    write_sorter_directory(tmp_path, trains, cluster_group=group)

    rec = nc.io.from_phy(tmp_path)
    assert dict(zip(rec.units["id"], rec.units["quality"])) == {0: "good", 1: "noise"}


def test_from_phy_missing_spike_clusters_falls_back_to_templates(tmp_path):
    trains = {0: np.array([0.1]), 1: np.array([0.2])}
    info = _phy_cluster_info([0, 1, 2], ["good", "good", "good"]).iloc[:2]
    write_sorter_directory(tmp_path, trains, cluster_info=info,
                            omit_spike_clusters=True)
    with pytest.warns(UserWarning, match="spike_templates"):
        rec = nc.io.from_phy(tmp_path)
    assert rec.n_spikes == 2


def test_from_phy_unknown_cluster_id_gets_unsorted_row(tmp_path):
    trains = {0: np.array([0.1]), 99: np.array([0.2])}  # 99 not in TSV
    info = _phy_cluster_info([0], ["good"])
    write_sorter_directory(tmp_path, trains, cluster_info=info)
    rec = nc.io.from_phy(tmp_path)
    assert set(rec.units["id"]) == {0, 99}
    assert rec.units.set_index("id").loc[99, "quality"] == "unsorted"


def test_from_phy_duration_override(tmp_path):
    trains = {0: np.array([0.1, 0.5])}
    info = _phy_cluster_info([0], ["good"])
    write_sorter_directory(tmp_path, trains, cluster_info=info)
    rec = nc.io.from_phy(tmp_path, duration=10.0)
    assert rec.duration == 10.0


def test_from_phy_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        nc.io.from_phy(tmp_path / "does_not_exist")


def test_from_phy_no_label_tables_raises(tmp_path):
    trains = {0: np.array([0.1])}
    write_sorter_directory(tmp_path, trains)  # no TSVs
    with pytest.raises(Exception, match="cluster_info|cluster_group"):
        nc.io.from_phy(tmp_path)
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest tests/test_io_phy.py -v`
Expected: all six new tests + the original roundtrip PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_io_phy.py
git commit -m "tests: edge cases for from_phy (fallbacks, missing files, overrides)"
```

---

## Task 7 — `from_spikeinterface`

**Files:**
- Create: `neurocomplexity/io/spikeinterface.py`
- Create: `tests/test_io_spikeinterface.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_io_spikeinterface.py`:

```python
"""Tests for from_spikeinterface (gated on the [spikeinterface] extra)."""
from __future__ import annotations

import sys

import numpy as np
import pytest

si = pytest.importorskip("spikeinterface")
from spikeinterface.core import NumpySorting  # noqa: E402

import neurocomplexity as nc  # noqa: E402


def test_from_spikeinterface_roundtrip():
    sample_rate = 30000.0
    trains = {
        0: np.array([0.1, 0.5, 1.2]),
        1: np.array([0.2, 0.9]),
    }
    spike_times_samples = {
        uid: (st * sample_rate).astype(np.int64) for uid, st in trains.items()
    }
    sorting = NumpySorting.from_unit_dict(
        [spike_times_samples], sampling_frequency=sample_rate,
    )

    rec = nc.io.from_spikeinterface(sorting)

    assert rec.n_units == 2
    assert rec.n_spikes == sum(len(v) for v in trains.values())
    assert rec.sampling_rate == sample_rate
    for uid, st in trains.items():
        recovered = np.sort(rec.spike_times[rec.unit_ids == uid])
        np.testing.assert_allclose(recovered, st, atol=1.0 / sample_rate)
    assert rec.source.source_format == "spikeinterface"


def test_from_spikeinterface_missing_pkg_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "spikeinterface", None)
    # Force re-import of the loader module to hit the lazy import path.
    sys.modules.pop("neurocomplexity.io.spikeinterface", None)
    from neurocomplexity.io import spikeinterface as adapter
    with pytest.raises(ImportError, match=r"neurocomplexity\[spikeinterface\]"):
        adapter.from_spikeinterface(object())
```

- [ ] **Step 2: Run test, confirm fail**

Run: `python -m pytest tests/test_io_spikeinterface.py -v`
Expected: `ModuleNotFoundError: No module named 'neurocomplexity.io.spikeinterface'` (or test skips if SI not installed — both acceptable; we ship the module either way).

- [ ] **Step 3: Implement the adapter**

Create `neurocomplexity/io/spikeinterface.py`:

```python
"""Bridge to the SpikeInterface ecosystem.

Accepts any ``spikeinterface.BaseSorting`` and, optionally, a paired
``BaseRecording`` for duration and channel metadata. SpikeInterface is a
soft dependency — install with ``pip install 'neurocomplexity[spikeinterface]'``.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from neurocomplexity.core.provenance import ProvenanceRecord
from neurocomplexity.core.recording import SpikeRecording


def from_spikeinterface(
    sorting,
    *,
    recording=None,
    duration: float | None = None,
    populations: Mapping[str, np.ndarray] | None = None,
) -> SpikeRecording:
    """Build a SpikeRecording from a spikeinterface BaseSorting.

    Parameters
    ----------
    sorting
        A ``spikeinterface.BaseSorting`` instance.
    recording
        Optional ``spikeinterface.BaseRecording``; used for ``duration``
        and any per-channel properties (e.g. ``brain_area``).
    duration
        Override the recording duration in seconds.
    populations
        Override the default ``{"all": ones}`` population mask.
    """
    try:
        import spikeinterface  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "from_spikeinterface requires spikeinterface. "
            "Install with: pip install 'neurocomplexity[spikeinterface]'"
        ) from exc

    sample_rate = float(sorting.sampling_frequency)
    unit_ids = list(sorting.unit_ids)

    spike_chunks: list[np.ndarray] = []
    owner_chunks: list[np.ndarray] = []
    for uid in unit_ids:
        st = sorting.get_unit_spike_train(uid, return_times=True)
        st = np.asarray(st, dtype=np.float64)
        spike_chunks.append(st)
        owner_chunks.append(np.full(st.shape, int(uid), dtype=np.int64))

    if spike_chunks:
        spike_times = np.concatenate(spike_chunks)
        owners = np.concatenate(owner_chunks)
        order = np.argsort(spike_times, kind="stable")
        spike_times = spike_times[order]
        owners = owners[order]
    else:
        spike_times = np.empty(0, dtype=np.float64)
        owners = np.empty(0, dtype=np.int64)

    # Build units DataFrame from any sorter properties.
    units_dict: dict = {"id": [int(u) for u in unit_ids]}
    for key in sorting.get_property_keys():
        try:
            vals = sorting.get_property(key)
        except Exception:
            continue
        if vals is None or len(vals) != len(unit_ids):
            continue
        units_dict[str(key)] = list(vals)
    units_df = pd.DataFrame(units_dict)
    if "quality" not in units_df.columns:
        units_df["quality"] = "unsorted"

    # Duration priority: explicit kwarg > recording.get_duration() > spike-times pad.
    if duration is None:
        if recording is not None:
            try:
                duration = float(recording.get_duration())
            except Exception:
                duration = None
    if duration is None:
        duration = float(spike_times.max()) + 1.0 if spike_times.size else 1.0
    duration = float(duration)

    if populations is None:
        populations = {"all": np.ones(len(units_df), dtype=bool)}

    provenance = ProvenanceRecord.for_memory(
        "spikeinterface", hint=type(sorting).__name__,
    )

    return SpikeRecording(
        spike_times=spike_times,
        unit_ids=owners,
        units=units_df,
        populations=populations,
        duration=duration,
        sampling_rate=sample_rate,
        source=provenance,
    )
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_io_spikeinterface.py -v`
Expected: roundtrip PASS if SI installed (else SKIPPED); missing-pkg test PASS.

- [ ] **Step 5: Commit**

```bash
git add neurocomplexity/io/spikeinterface.py tests/test_io_spikeinterface.py
git commit -m "io: from_spikeinterface adapter (soft dependency)"
```

---

## Task 8 — Documentation updates

**Files:**
- Modify: `docs/quickstart.md`
- Modify: `docs/installation.md`
- Create: `docs/io.md`
- Modify: `docs/index.md`

- [ ] **Step 1: Update `docs/quickstart.md`**

Replace the existing "Load a recording" section (NWB block plus the v1.1 admonition) with:

````markdown
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

```python
rec = nc.io.from_kilosort("path/to/kilosort_output/")
rec = rec.filter_units(quality=["good"])
```

### From any SpikeInterface sorter

```python
import spikeinterface.extractors as se
sorting = se.read_phy("path/to/phy_output/")
rec = nc.io.from_spikeinterface(sorting)
```

The SpikeInterface bridge is a soft dependency — install with
`pip install "neurocomplexity[spikeinterface]"`.
````

- [ ] **Step 2: Update `docs/installation.md`**

In the "Optional dependencies" section, append to the install block:

```bash
pip install "neurocomplexity[spikeinterface]"  # adds spikeinterface
```

And extend the trailing prose to mention that `from_spikeinterface` and `from_nwb` both raise clear `ImportError`s when the extras are missing.

- [ ] **Step 3: Create `docs/io.md`**

```markdown
# I/O loaders

`neurocomplexity.io` materialises a `SpikeRecording` from disk or memory.
All file/external loaders are lazy-imported so the package stays light
when you only need one format.

## Decision flow

```
have an NWB file?            → from_nwb
have a Phy curation dir?     → from_phy
have raw Kilosort output?    → from_kilosort
have any SpikeInterface       → from_spikeinterface
sorter object?
have spike trains in memory? → from_dict
```

## `from_phy(directory, *, duration=None, populations=None)`

Reads `spike_times.npy`, `spike_clusters.npy`, `params.py`, and
`cluster_info.tsv` (falling back to `cluster_group.tsv`). Quality labels
come from the curated `group` column; no filtering is applied at load
time — call `rec.filter_units(quality=['good'])` downstream.

Columns normalized into `rec.units`: `id`, `quality`, `firing_rate`,
`peak_channel`, `depth`, `amplitude`, `contam_pct`, `n_spikes`. Any
other columns from `cluster_info.tsv` are passed through verbatim.

## `from_kilosort(directory, *, duration=None, populations=None)`

Same directory layout as `from_phy`, but quality labels come from
`cluster_KSLabel.tsv` (automatic) instead of `cluster_info.tsv`
(curated).

## `from_spikeinterface(sorting, *, recording=None, duration=None, populations=None)`

Accepts any `spikeinterface.BaseSorting`. Pull duration and channel
metadata from an optional paired `recording`. SpikeInterface is the
recommended path for formats not natively supported here (Open Ephys,
Blackrock, Plexon, MEArec, NEO-readable, ...).

## Security note on `params.py`

Phy writes `params.py` as executable Python. `from_phy` and
`from_kilosort` execute it in an isolated namespace, matching the
behaviour of Phy itself and SpikeInterface. Treat sorter directories
the same way you treat any other code you would `python -m` against —
do not run `from_phy` on directories pulled from untrusted sources.
```

- [ ] **Step 4: Add `io` to the toctree in `docs/index.md`**

In the "User guide" toctree, insert `io` between `quickstart` and `examples/tutorial`:

```
installation
quickstart
io
examples/tutorial
benchmarks
inference
```

- [ ] **Step 5: Commit**

```bash
git add docs/quickstart.md docs/installation.md docs/io.md docs/index.md
git commit -m "docs: Phy/Kilosort/SpikeInterface loader docs + decision flowchart"
```

---

## Task 9 — Version bump + CHANGELOG

**Files:**
- Modify: `neurocomplexity/_version.py`
- Modify: `pyproject.toml`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Bump version**

`neurocomplexity/_version.py`:

```python
__version__ = "1.1.0"
```

`pyproject.toml` `[project]` block: change `version = "1.0.0"` to `version = "1.1.0"`.

- [ ] **Step 2: Add CHANGELOG section**

Prepend to `CHANGELOG.md` (above the v1.0.0 section):

```markdown
## v1.1.0 — 2026-05-17

### Added
- `nc.io.from_phy(directory)` — load a Phy curation directory.
- `nc.io.from_kilosort(directory)` — load raw Kilosort output.
- `nc.io.from_spikeinterface(sorting, recording=None)` — bridge to any
  `spikeinterface.BaseSorting`.
- `[spikeinterface]` optional install extra.
- New `docs/io.md` loader reference page.

### Changed
- `neurocomplexity.io.__init__` now lazy-imports every heavy loader via
  `__getattr__`; importing the top-level package no longer touches
  pynwb or spikeinterface.
```

- [ ] **Step 3: Run the full test suite one last time**

Run: `python -m pytest tests/ -q --ignore=tests/test_inference_calibration.py`
Expected: every test PASS (or SKIP for SI tests if extra not installed).

- [ ] **Step 4: Commit**

```bash
git add neurocomplexity/_version.py pyproject.toml CHANGELOG.md
git commit -m "release: v1.1.0 — Phy, Kilosort, and SpikeInterface loaders"
```

---

## Self-review notes

- **Spec coverage:** Tasks 3-7 cover every loader, the helper, the column normalization map, the params.py parser, the fallback chain (`cluster_info` → `cluster_group`, `spike_clusters` → `spike_templates`), missing-extras error paths, and the synthetic-cluster row guarantee. Task 8 covers all four doc files in the spec. Task 9 covers the version bump and CHANGELOG.
- **Type consistency:** `_load_sorter_output` signature, the `_RENAME` map, and the four loader signatures match the spec verbatim.
- **No placeholders:** every code block is complete; every `pytest` command lists the exact module to run and the expected outcome.
