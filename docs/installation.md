# Installation

## From PyPI (recommended)

```bash
pip install neurocomplexity
```

## From conda-forge

```bash
conda install -c conda-forge neurocomplexity
```

## From source (development)

```bash
git clone https://github.com/aurmandi/neurocomplexity.git
cd neurocomplexity
pip install -e ".[dev,docs]"
```

## Supported Python versions

3.10, 3.11, 3.12. Tested on Linux, macOS, and Windows in CI.

## Optional dependencies

- `pynwb` is a runtime dependency; the NWB loader is the primary entry point.
- The Phy and Kilosort loaders require no additional pip packages beyond
  numpy / pandas.
