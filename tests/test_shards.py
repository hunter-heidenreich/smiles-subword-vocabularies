"""Unit tests for the shared byte-budgeted Parquet `ShardWriter`.

`ShardWriter` is exercised transitively by every ingest/preprocess stage, but
its standalone contract — lazy open, empty-batch no-op, idempotent close,
byte-budget rollover, and the per-shard `ShardInfo` provenance (sha256, row and
byte counts) — is worth pinning directly rather than inferring through a full
stage run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq

from smiles_subword._hashing import sha256_file
from smiles_subword._shards import ShardWriter

if TYPE_CHECKING:
    from pathlib import Path

_SCHEMA = pa.schema([("smiles", pa.string())])


def _batch(*values: str) -> pa.RecordBatch:
    return pa.record_batch([pa.array(list(values))], schema=_SCHEMA)


def _make_writer(
    staging_dir: Path, *, target_bytes: int, prefix: str = "raw_v1"
) -> ShardWriter:
    return ShardWriter(
        staging_dir,
        schema=_SCHEMA,
        shard_prefix=prefix,
        target_bytes=target_bytes,
    )


def test_collects_into_single_shard_under_budget(tmp_path: Path) -> None:
    writer = _make_writer(tmp_path, target_bytes=10**9)
    writer.write_batch(_batch("C", "CC"))
    writer.write_batch(_batch("CCC"))
    writer.close_current()

    assert len(writer.shards) == 1
    assert writer.shards[0].n_rows == 3
    table = pq.read_table(tmp_path / writer.shards[0].file)
    assert table.column("smiles").to_pylist() == ["C", "CC", "CCC"]


def test_rolls_over_when_shard_crosses_budget(tmp_path: Path) -> None:
    # target_bytes=1 forces every written batch past the budget, so each
    # non-empty write lands in its own shard.
    writer = _make_writer(tmp_path, target_bytes=1)
    for i in range(3):
        writer.write_batch(_batch(f"C{i}"))
    writer.close_current()

    assert len(writer.shards) == 3
    assert [s.n_rows for s in writer.shards] == [1, 1, 1]


def test_empty_batch_does_not_open_a_shard(tmp_path: Path) -> None:
    writer = _make_writer(tmp_path, target_bytes=10**9)
    writer.write_batch(_batch())
    writer.close_current()

    assert writer.shards == []
    assert list(tmp_path.glob("*.parquet")) == []


def test_close_current_without_open_shard_is_noop(tmp_path: Path) -> None:
    writer = _make_writer(tmp_path, target_bytes=10**9)
    writer.close_current()
    assert writer.shards == []


def test_close_current_is_idempotent(tmp_path: Path) -> None:
    writer = _make_writer(tmp_path, target_bytes=10**9)
    writer.write_batch(_batch("C"))
    writer.close_current()
    writer.close_current()  # second close must not duplicate the ShardInfo
    assert len(writer.shards) == 1


def test_shard_files_are_zero_padded_and_sequential(tmp_path: Path) -> None:
    writer = _make_writer(tmp_path, target_bytes=1, prefix="raw_v1")
    for i in range(3):
        writer.write_batch(_batch(f"C{i}"))
    writer.close_current()

    assert [s.file for s in writer.shards] == [
        "raw_v1-00000.parquet",
        "raw_v1-00001.parquet",
        "raw_v1-00002.parquet",
    ]


def test_shardinfo_records_provenance(tmp_path: Path) -> None:
    writer = _make_writer(tmp_path, target_bytes=10**9)
    writer.write_batch(_batch("C", "CC"))
    writer.close_current()

    info = writer.shards[0]
    path = tmp_path / info.file
    assert info.sha256 == sha256_file(path)
    assert info.n_bytes == path.stat().st_size
    assert info.n_rows == 2
