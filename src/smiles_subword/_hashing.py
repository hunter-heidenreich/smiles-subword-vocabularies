"""Shared SHA256 helper for provenance-tracking call sites.

Stage manifests, run manifests, and shard verification all stream-hash files
the same way: 1 MiB chunks, lowercase hex. Private module; callers re-export
under their own names where that preserves an existing public surface.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

_SHA_READ_CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    """Return the lowercase-hex SHA256 of ``path``, streamed in 1 MiB chunks."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_SHA_READ_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase-hex SHA256 of in-memory ``data``.

    The in-memory counterpart to :func:`sha256_file`, for call sites that have
    already materialized the bytes (a built manifest, a small artifact).
    """
    return hashlib.sha256(data).hexdigest()


__all__ = ["sha256_bytes", "sha256_file"]
