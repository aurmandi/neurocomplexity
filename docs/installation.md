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

NWB support is opt-in to keep the core install lightweight (no `pynwb` /
`h5py` / `hdmf` unless you ask for them):

```bash
pip install "neurocomplexity[nwb]"              # adds pynwb + h5py
pip install "neurocomplexity[spikeinterface]"   # adds spikeinterface
pip install "neurocomplexity[viz]"              # adds matplotlib (and plotly/dash)
pip install "neurocomplexity[dev]"              # pytest, ruff, coverage
pip install "neurocomplexity[nwb,spikeinterface,viz,dev]"   # everything
```

Calling `nc.io.from_nwb(...)` or `nc.io.from_spikeinterface(...)` without
the matching extra raises a clear `ImportError` pointing at the right
install command.
