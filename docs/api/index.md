# API reference

This page documents the **stable public surface** covered by SemVer. See
[`docs/api_stability.md`](../api_stability.md) for the contract,
deprecation policy, and the list of *experimental* re-exports
(`ContinuousSignal`, `ProvenanceRecord`, `set_progress`,
`estimate_bin_spikes_bytes`, `neurocomplexity.warnings` private state)
that are intentionally **not** in the SemVer-covered surface.

## Core data types

```{eval-rst}
.. automodule:: neurocomplexity.core.recording
   :members:
   :show-inheritance:
```

## Loaders

```{eval-rst}
.. automodule:: neurocomplexity.io.nwb
   :members:

.. automodule:: neurocomplexity.io.dict_loader
   :members:
```

## Analyses

### Criticality

```{eval-rst}
.. automodule:: neurocomplexity.analysis.criticality
   :members:

.. automodule:: neurocomplexity.analysis.branching
   :members:

.. automodule:: neurocomplexity.analysis.shape_collapse
   :members:
```

### Information theory

```{eval-rst}
.. automodule:: neurocomplexity.analysis.transfer_entropy
   :members:

.. automodule:: neurocomplexity.analysis.pid
   :members:

.. automodule:: neurocomplexity.analysis.autonomy
   :members:
```

### Dimensionality

```{eval-rst}
.. automodule:: neurocomplexity.analysis.dimensionality
   :members:
```

### Surrogates

```{eval-rst}
.. automodule:: neurocomplexity.analysis.surrogates
   :members:
```

## Inference

```{eval-rst}
.. automodule:: neurocomplexity.inference.surrogates
   :members:

.. automodule:: neurocomplexity.inference.bootstrap
   :members:

.. automodule:: neurocomplexity.inference.null_test
   :members:

.. automodule:: neurocomplexity.inference.results
   :members:
```

## Benchmarks

```{eval-rst}
.. automodule:: neurocomplexity.benchmarks.runner
   :members:
```

## Experimental re-exports

These are importable from the top-level namespace but are **not** covered
by SemVer. See [`docs/api_stability.md`](../api_stability.md) for the
rationale.

```{eval-rst}
.. automodule:: neurocomplexity.core.continuous
   :members:

.. automodule:: neurocomplexity.core.provenance
   :members:

.. automodule:: neurocomplexity._progress
   :members:

.. automodule:: neurocomplexity.analysis._binning
   :members: estimate_bin_spikes_bytes

.. automodule:: neurocomplexity.warnings
   :members:
```

## Visualisation

```{eval-rst}
.. automodule:: neurocomplexity.viz
   :members:
```
