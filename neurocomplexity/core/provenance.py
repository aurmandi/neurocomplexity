from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from neurocomplexity._version import __version__


@dataclass(frozen=True)
class ProvenanceRecord:
    source_path: str
    source_format: str
    source_hash: str
    loader_version: str
    package_version: str
    loaded_at: str

    @classmethod
    def for_file(cls, path: str | os.PathLike, source_format: str) -> "ProvenanceRecord":
        path = str(Path(path).resolve())
        return cls(
            source_path=path,
            source_format=source_format,
            source_hash=hash_file(path),
            loader_version=__version__,
            package_version=__version__,
            loaded_at=datetime.now(timezone.utc).isoformat(),
        )

    @classmethod
    def for_memory(cls, source_format: str, hint: str = "") -> "ProvenanceRecord":
        return cls(
            source_path=f"<memory:{hint}>" if hint else "<memory>",
            source_format=source_format,
            source_hash="",
            loader_version=__version__,
            package_version=__version__,
            loaded_at=datetime.now(timezone.utc).isoformat(),
        )


def hash_file(path: str | os.PathLike, head_bytes: int = 4 * 1024 * 1024,
               tail_bytes: int = 4 * 1024 * 1024) -> str:
    """BLAKE2b hash of (head || tail || filesize). Cheap fingerprint, not crypto-secure."""
    path = Path(path)
    size = path.stat().st_size
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as f:
        head = f.read(min(head_bytes, size))
        h.update(head)
        if size > head_bytes + tail_bytes:
            f.seek(-tail_bytes, os.SEEK_END)
            h.update(f.read(tail_bytes))
        elif size > head_bytes:
            h.update(f.read())
    h.update(size.to_bytes(8, "little"))
    return h.hexdigest()
