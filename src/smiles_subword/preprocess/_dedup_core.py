"""Core for the DuckDB window-dedup.

The byte-deterministic dedup query — keep the lexicographically smallest
``source_id`` per ``smiles``, emit rows ordered by ``smiles`` — and the
hash-bucket partitioner, shared by the ``canon_dedup`` pipeline
(``raw_v1`` -> ``canon_dedup_v1``).

Output columns and Arrow schema are parameters; the dedup key is not: the input
must carry a ``smiles`` column (the dedup key) and a ``source_id`` column (the
tie-break key).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from smiles_subword._sql import quote_ident
from smiles_subword.preprocess._io import coerce_to_schema

if TYPE_CHECKING:
    from pathlib import Path

    import duckdb

    from smiles_subword.preprocess._io import ShardWriter

_MEMORY_LIMIT_RE = re.compile(r"^\d+(\.\d+)?\s*[KMGT]?B?$", re.IGNORECASE)

BUCKET_KEY_LABEL = "smiles_prefix_2"


def apply_duckdb_pragmas(
    con: duckdb.DuckDBPyConnection,
    *,
    threads: int | None,
    memory_limit: str | None,
) -> None:
    """Apply optional thread / memory-limit PRAGMAs to a DuckDB connection.

    Raises:
        ValueError: if `memory_limit` is set to a malformed value.
    """
    if threads is not None:
        con.execute(f"PRAGMA threads = {int(threads)}")
    if memory_limit is not None:
        if not _MEMORY_LIMIT_RE.match(memory_limit):
            raise ValueError(f"malformed duckdb_memory_limit: {memory_limit!r}")
        con.execute(f"SET memory_limit = '{memory_limit}'")


def dedup_to_writer(
    con: duckdb.DuckDBPyConnection,
    *,
    parquet_source: str,
    output_columns: tuple[str, ...],
    writer: ShardWriter,
    out_schema: pa.Schema,
    rows_per_batch: int,
) -> None:
    """Window-dedup `parquet_source` into `writer`.

    Keeps the row with the lexicographically smallest `source_id` per `smiles`,
    breaking further ties by the remaining emitted columns so the surviving row
    is uniquely determined even when `(smiles, source_id)` is not unique, and
    emits rows ordered by `smiles` — byte-identical under re-runs with a
    deterministic shard writer. `parquet_source` may be a glob or a single file.
    """
    # Column identifiers are interpolated into SQL (not ?-params), so validate +
    # quote each via the shared guard.
    select_cols = ", ".join(quote_ident(c) for c in output_columns)
    # Total tie-break: `source_id` first (keep the smallest), then every other
    # emitted column, so the surviving row is unique even when `(smiles,
    # source_id)` is not — byte-determinism without relying on upstream uniqueness.
    tie_break = ", ".join(
        quote_ident(c)
        for c in ("source_id", *(c for c in output_columns if c != "source_id"))
    )
    result = con.execute(
        f"""
        SELECT {select_cols}
        FROM (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY smiles ORDER BY {tie_break}
                   ) AS _rn
            FROM read_parquet(?)
        )
        WHERE _rn = 1
        ORDER BY smiles
        """,
        [parquet_source],
    )
    reader = result.to_arrow_reader(batch_size=rows_per_batch)
    for batch in reader:
        writer.write_batch(coerce_to_schema(batch, out_schema))


def bucket_key(smiles: str) -> str:
    """Hex-encoded first 2 UTF-8 bytes of `smiles`, zero-padded to 4 chars.

    Hex preserves byte ordering, so sorting bucket keys reproduces the
    underlying SMILES-prefix ordering — required for byte-equivalence with
    single-pass dedup. Canonical SMILES are ASCII so the result is always
    exactly 4 lowercase hex chars (filename-safe across platforms).
    """
    prefix = smiles.encode("utf-8")[:2]
    if len(prefix) < 2:
        prefix = prefix + b"\x00" * (2 - len(prefix))
    return prefix.hex()


def partition_into_buckets(
    input_shards: list[Path],
    bucket_dir: Path,
    *,
    rows_per_batch: int,
    out_schema: pa.Schema,
    parquet_compression: str,
    parquet_compression_level: int,
) -> dict[str, Path]:
    """Partition `input_shards` into per-SMILES-prefix bucket Parquet files.

    Duplicates always collide in the same bucket, so each bucket is deduped
    independently. Returns a mapping of bucket key to its file path.

    One `ParquetWriter` per distinct 2-byte prefix stays open for the whole
    pass, so peak memory scales with the number of distinct prefixes — fine
    while canonical-SMILES prefix cardinality stays small (low hundreds);
    revisit if a corpus broadens that alphabet.
    """
    bucket_paths: dict[str, Path] = {}
    bucket_writers: dict[str, pq.ParquetWriter] = {}
    try:
        for shard in input_shards:
            pf = pq.ParquetFile(shard)
            for batch in pf.iter_batches(batch_size=rows_per_batch):
                if batch.num_rows == 0:
                    continue
                keys = pa.array(
                    [bucket_key(s.as_py()) for s in batch.column("smiles")],
                    type=pa.string(),
                )
                for key in pc.unique(keys).to_pylist():  # pyright: ignore[reportAttributeAccessIssue]
                    sub = batch.filter(pc.equal(keys, pa.scalar(key)))  # pyright: ignore[reportAttributeAccessIssue]
                    writer = bucket_writers.get(key)
                    if writer is None:
                        path = bucket_dir / f"bucket_{key}.parquet"
                        bucket_paths[key] = path
                        writer = pq.ParquetWriter(
                            path,
                            out_schema,
                            compression=parquet_compression,
                            compression_level=parquet_compression_level,
                        )
                        bucket_writers[key] = writer
                    writer.write_batch(coerce_to_schema(sub, out_schema))
    finally:
        for w in bucket_writers.values():
            w.close()
    return bucket_paths
