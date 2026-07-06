"""Shared low-level I/O helpers for preprocess stages."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from smiles_subword._hashing import sha256_file
from smiles_subword._io import atomic_output_dir
from smiles_subword._shards import ShardWriter as ShardWriter
from smiles_subword._time import utc_now_naive_seconds
from smiles_subword.manifest import shard_dicts as shard_dicts

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from datetime import datetime
    from pathlib import Path


def verify_shard_sha256(shards: Iterable[tuple[Path, str]]) -> None:
    """Hash each shard and raise if any disagrees with the recorded digest."""
    for path, expected in shards:
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"sha256 mismatch for {path}: expected {expected}, got {actual}"
            )


def coerce_to_schema(batch: pa.RecordBatch, schema: pa.Schema) -> pa.RecordBatch:
    """Project `batch` onto `schema`: select the schema's fields by name and cast
    each to its type (dropping any extra columns). A no-op if already equal.
    """
    if batch.schema.equals(schema):
        return batch
    arrays = [batch.column(field.name).cast(field.type) for field in schema]
    return pa.RecordBatch.from_arrays(arrays, schema=schema)


def read_source_manifest(input_dir: Path) -> tuple[bytes, dict[str, Any]]:
    """Return (raw bytes, parsed dict) of `input_dir/MANIFEST.yaml`.

    Raw bytes are kept around so callers can hash them for provenance.
    """
    raw = (input_dir / "MANIFEST.yaml").read_bytes()
    return raw, yaml.safe_load(raw)


def read_and_verify_source_manifest(
    input_dir: Path, *, verify: bool
) -> tuple[bytes, dict[str, Any]]:
    """Read `input_dir/MANIFEST.yaml`, optionally verifying its shard SHAs.

    Returns `(raw bytes, parsed dict)` like :func:`read_source_manifest`. When
    `verify` is True, every shard the manifest lists is hashed and checked
    against its recorded digest (raising on mismatch) first.
    """
    raw, manifest = read_source_manifest(input_dir)
    if verify:
        verify_shard_sha256(
            (input_dir / e["file"], e["sha256"]) for e in manifest.get("shards", [])
        )
    return raw, manifest


def list_input_shards(input_dir: Path) -> list[Path]:
    """Return the sorted `*.parquet` shards under `input_dir`.

    Globs every shard regardless of upstream prefix (`conformant_v1-*`,
    `canon_dedup_v1-*`, ...); the preprocess stages key off the directory, not
    the shard name.
    """
    return sorted(input_dir.glob("*.parquet"))


def count_parquet_rows(shards: Iterable[Path]) -> int:
    """Total row count across `shards` (sum of each file's metadata `num_rows`)."""
    return sum(pq.ParquetFile(s).metadata.num_rows for s in shards)


def write_manifest(directory: Path, payload: dict[str, object]) -> None:
    """Write `payload` to `directory/MANIFEST.yaml` as block YAML (key order kept).

    The plain write is correct: stage writers call this inside the `stage_run`
    staging dir that is renamed into place atomically as a whole.
    """
    (directory / "MANIFEST.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))


@contextmanager
def stage_run(output_dir: Path) -> Iterator[tuple[Path, datetime]]:
    """Manage staging-dir + atomic-rename around one stage driver call.

    Yields `(staging_dir, started_ts)`. Via
    :func:`smiles_subword._io.atomic_output_dir` with ``keep_on_error=True``: on
    normal exit the staging dir replaces `output_dir`; on exception it is left in
    place for inspection (the rename never happens).
    """
    started_ts = utc_now_naive_seconds()
    with atomic_output_dir(output_dir, keep_on_error=True) as staging_dir:
        yield staging_dir, started_ts
