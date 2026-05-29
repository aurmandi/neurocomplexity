"""Global progress-bar toggle.

Disabled by default so the library stays silent in scripts and CI. Enable with
``nc.set_progress(True)`` for an interactive session; every long-running loop
in ``inference`` and ``analysis`` will then show a transient ``tqdm`` bar.
"""
from __future__ import annotations

from collections.abc import Iterable

_enabled: bool = False


def set_progress(enabled: bool) -> None:
    """Enable or disable progress bars globally for this Python process.

    Affects every long-running loop in :mod:`neurocomplexity.inference`
    and :mod:`neurocomplexity.analysis` that wraps its iterator with
    :func:`progress_iter`. Off by default so the library stays silent in
    scripts and CI.

    Parameters
    ----------
    enabled
        ``True`` to show ``tqdm`` bars, ``False`` to suppress them.
    """
    global _enabled
    _enabled = bool(enabled)


def progress_iter(iterable: Iterable, *, total: int | None = None,
                  desc: str | None = None):
    """Wrap iterable with tqdm if progress is enabled, otherwise pass through.

    Bars use ``leave=False`` so they vanish on completion, keeping notebook
    output uncluttered when many analyses run in series.
    """
    if not _enabled:
        return iterable
    from tqdm.auto import tqdm
    return tqdm(iterable, total=total, desc=desc, leave=False)
