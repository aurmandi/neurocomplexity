# Contributing to neurocomplexity

Thanks for considering a contribution. The package is small, so onboarding is short.

## Development setup

```bash
git clone https://github.com/aurmandi/neurocomplexity.git
cd neurocomplexity
python -m venv .venv
. .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Running tests

```bash
# Fast suite (CI gate)
pytest tests/ --ignore=tests/test_inference_calibration.py -q

# Full suite including the slow calibration and benchmark cases
pytest tests/ -q
```

## Code style

- Run `ruff check .` before opening a PR.
- Public functions need docstrings; we do not enforce a specific style.
- New analyses should ship with a benchmark case in
  `neurocomplexity/benchmarks/cases/` and a row in `docs/benchmarks.md`.

## Reporting bugs / proposing features

Use the GitHub issue templates. Bug reports must include the version and a
minimal reproducer.

## Pull-request checklist

- Tests added or updated
- `pytest` passes locally
- `ruff check .` is clean
- `CHANGELOG.md` has an entry under the next-release header
