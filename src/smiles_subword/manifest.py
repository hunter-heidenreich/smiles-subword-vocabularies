"""Provenance shapes and readers/writers for the project's MANIFEST.yaml files.

Manifest provenance lives here at two granularities:

- The top-level ``data/MANIFEST.yaml`` ledger of externally-sourced artifacts —
  :class:`ManifestEntry` plus :func:`load_manifest_entry` /
  :func:`record_manifest_entry` (one entry per corpus: URL, SHA256, size,
  ingest date).
- The per-shard records the ingest and preprocess stages write into their own
  stage ``MANIFEST.yaml`` files — :class:`ShardInfo` / :func:`shard_dicts`.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from smiles_subword.paths import DATA_DIR

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

try:
    import fcntl

    _HAS_FLOCK = True
except ImportError:
    _HAS_FLOCK = False

__all__ = [
    "ManifestEntry",
    "ShardInfo",
    "load_manifest_entry",
    "record_manifest_entry",
    "shard_dicts",
]


# --- Top-level data/MANIFEST.yaml ledger -------------------------------------


class ManifestEntry(BaseModel):
    """One externally-sourced artifact tracked in `data/MANIFEST.yaml`.

    `provenance` is an optional structured block for derived artifacts that
    need machine-recoverable lineage beyond the free-text `notes` — e.g. the
    ZINC-22 corpus records its enumerated tranche set there. It is
    omitted from the serialized entry when `None`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    path: Path
    source_url: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    ingest_date: date
    notes: str | None = None
    provenance: dict[str, Any] | None = None


def load_manifest_entry(
    manifest_id: str,
    manifest_path: Path | None = None,
) -> ManifestEntry:
    """Look up one artifact by id in `data/MANIFEST.yaml`.

    Raises:
        KeyError: if `manifest_id` is not present in the manifest.
        FileNotFoundError: if the manifest file does not exist.
    """
    path = manifest_path or (DATA_DIR / "MANIFEST.yaml")
    with path.open() as fh:
        payload = yaml.safe_load(fh)
    for raw in payload.get("artifacts", []):
        if raw.get("id") == manifest_id:
            return ManifestEntry.model_validate(raw)
    raise KeyError(f"manifest_id {manifest_id!r} not found in {path}")


def record_manifest_entry(
    entry: ManifestEntry,
    manifest_path: Path | None = None,
) -> None:
    """Append `entry` to `data/MANIFEST.yaml`, idempotent on (id, sha256).

    For record-on-first-observation sources: the first download writes its
    observed SHA here; later loads verify against it. The read-modify-write cycle
    is serialized by an advisory `flock` on a sidecar `<manifest_path>.lock`
    (POSIX only — no-op without `fcntl`), and the replace is atomic (tmp + fsync
    + rename), so concurrent runs cannot clobber each other and a crash leaves
    the previous manifest intact.

    Raises:
        ValueError: if `entry.id` is already present with a different SHA
            (upstream artifact changed, or the ID was reused); reconcile
            manually before rerunning.
    """
    path = manifest_path or (DATA_DIR / "MANIFEST.yaml")
    with _locked(path):
        with path.open() as fh:
            payload = yaml.safe_load(fh) or {}
        artifacts = payload.setdefault("artifacts", [])
        for raw in artifacts:
            if raw.get("id") != entry.id:
                continue
            existing = ManifestEntry.model_validate(raw)
            if existing.sha256 == entry.sha256:
                return
            raise ValueError(
                f"manifest_id {entry.id!r}: recorded sha256 {existing.sha256} "
                f"conflicts with new {entry.sha256}"
            )
        artifacts.append(
            entry.model_dump(mode="json", exclude_none=True),
        )
        _atomic_write_yaml(path, payload)


@contextlib.contextmanager
def _locked(path: Path) -> Iterator[None]:
    """Hold an exclusive advisory `flock` on `<path>.lock` for the block.

    No-op on platforms without `fcntl.flock` (Windows).
    """
    if not _HAS_FLOCK:
        yield
        return
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _atomic_write_yaml(path: Path, payload: object) -> None:
    """Write `payload` to `path` atomically (tmp + fsync + replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False)
        fh.flush()
        os.fsync(fh.fileno())
    tmp_path.replace(path)


# --- Per-shard stage-manifest provenance -------------------------------------


@dataclass(frozen=True)
class ShardInfo:
    """Per-shard provenance recorded in a stage `MANIFEST.yaml`.

    One entry per written Parquet shard: its file name, content hash, and
    row/byte counts. Shared by the ingest and preprocess stages.
    """

    file: str
    sha256: str
    n_rows: int
    n_bytes: int


def shard_dicts(shards: Iterable[ShardInfo]) -> list[dict[str, object]]:
    """Render shards into the manifest YAML's per-shard dict shape."""
    return [
        {
            "file": s.file,
            "sha256": s.sha256,
            "n_rows": s.n_rows,
            "n_bytes": s.n_bytes,
        }
        for s in shards
    ]
