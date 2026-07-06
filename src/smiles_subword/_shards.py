"""Shared byte-budgeted Parquet shard writer for the ingest and preprocess stages.

Both emit `raw_v1`-style Parquet shards into a staging dir under the same
contract: roll over on a byte budget, record a `ShardInfo` per shard.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pyarrow.parquet as pq

from smiles_subword._hashing import sha256_file
from smiles_subword.manifest import ShardInfo

if TYPE_CHECKING:
    from pathlib import Path

    import pyarrow as pa


class ShardWriter:
    """Byte-budgeted Parquet shard writer.

    Opens a new shard when the current one crosses `target_bytes`; records a
    `ShardInfo` (file, sha256, row/byte counts) per closed shard.

    The size check is approximate: `ParquetWriter` buffers a row group in memory,
    so `st_size` lags until it flushes — a shard can overshoot `target_bytes`
    before rolling over. Treat `target_bytes` as a soft floor, not an exact cut
    point.
    """

    def __init__(
        self,
        staging_dir: Path,
        *,
        schema: pa.Schema,
        shard_prefix: str,
        target_bytes: int,
        compression: Literal["zstd", "snappy", "gzip"] = "zstd",
        compression_level: int = 3,
    ) -> None:
        self._staging_dir = staging_dir
        self._schema = schema
        self._shard_prefix = shard_prefix
        self._target_bytes = target_bytes
        self._compression = compression
        self._compression_level = compression_level
        self._writer: pq.ParquetWriter | None = None
        self._current_path: Path | None = None
        self._current_rows = 0
        self.shards: list[ShardInfo] = []

    def write_batch(self, batch: pa.RecordBatch) -> None:
        if batch.num_rows == 0:
            return
        if self._writer is None:
            self._open_new_shard()
        assert self._writer is not None
        assert self._current_path is not None
        self._writer.write_batch(batch)
        self._current_rows += batch.num_rows
        if self._current_path.stat().st_size >= self._target_bytes:
            self.close_current()

    def close_current(self) -> None:
        if self._writer is None or self._current_path is None:
            return
        self._writer.close()
        size = self._current_path.stat().st_size
        self.shards.append(
            ShardInfo(
                file=self._current_path.name,
                sha256=sha256_file(self._current_path),
                n_rows=self._current_rows,
                n_bytes=size,
            )
        )
        self._writer = None
        self._current_path = None
        self._current_rows = 0

    def _open_new_shard(self) -> None:
        self._current_path = (
            self._staging_dir / f"{self._shard_prefix}-{len(self.shards):05d}.parquet"
        )
        self._writer = pq.ParquetWriter(
            self._current_path,
            self._schema,
            compression=self._compression,
            compression_level=self._compression_level,
        )
