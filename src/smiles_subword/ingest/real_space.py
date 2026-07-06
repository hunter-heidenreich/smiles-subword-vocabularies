"""Stage 0 REAL-Space ingest: enumerated Enamine REAL .cxsmiles -> raw_v1 shards.

Globs the pre-staged Enamine REAL `.cxsmiles` set under `cfg.raw_dir`
(acquisition-agnostic) and streams every file, in sorted order, into
byte-budgeted Parquet shards conforming to the raw_v1 schema:

    source_id: string, smiles: string, source: string, ingest_ts: timestamp[us]

The SMILES column is captured verbatim: any CXSMILES ` |...|` extension block
(space-delimited, hence the quote-disabled reader) passes through for
`canon_dedup_v1` to parse or reject; a NULL/missing SMILES is coalesced to `''`
(the schema's `smiles` is non-nullable).

Provenance: no single content-addressed input, so the manifest pins every file
by SHA256 in a `source_files` array plus an aggregate `input_sha256`.
Determinism: files read one at a time, sorted, through a single-threaded DuckDB
connection, so reruns produce byte-identical shards (modulo `ingest_ts`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from smiles_subword._hashing import sha256_bytes, sha256_file
from smiles_subword.ingest._common import (
    ingest_timestamp,
    relative_to_repo,
    run_single_file_ingest,
    stream_csv_path,
    write_stage_manifest,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from datetime import datetime
    from pathlib import Path

    import pyarrow as pa

    from smiles_subword.config import RealSpaceCorpusConfig
    from smiles_subword.ingest.types import IngestResult
    from smiles_subword.manifest import ShardInfo

__all__ = ["ingest"]


def ingest(cfg: RealSpaceCorpusConfig) -> IngestResult:
    """Glob `cfg.raw_dir` for `cfg.glob` files and stream them into raw_v1 shards.

    Writes shards into a sibling `.tmp` directory, then atomic-renames into
    place. Always emits a per-stage `MANIFEST.yaml` listing every shard plus the
    per-file `source_files` provenance block.

    Args:
        cfg: validated REAL-Space corpus config.

    Raises:
        FileNotFoundError: if `cfg.glob` matches no file under `cfg.raw_dir`.
    """
    files = sorted(cfg.raw_dir.glob(cfg.glob))
    if not files:
        raise FileNotFoundError(
            f"glob {cfg.glob!r} matched no files under {cfg.raw_dir}"
        )

    ingest_ts = ingest_timestamp()
    rows_per_file: dict[Path, int] = dict.fromkeys(files, 0)
    file_shas = {path: sha256_file(path) for path in files}
    agg_lines = sorted(f"{relative_to_repo(path)}:{file_shas[path]}" for path in files)
    input_sha256 = sha256_bytes("\n".join(agg_lines).encode())

    def _stream(c: RealSpaceCorpusConfig, ts: datetime) -> Iterator[pa.RecordBatch]:
        """Yield raw_v1 batches for every file, in order, file by file.

        Tallies each file's emitted row count into `rows_per_file` as a side
        effect; `_write_manifest` reads it once `write_shards` has drained this.
        """
        for path in files:
            for batch in stream_csv_path(c, path, ts, threads=1):
                rows_per_file[path] += batch.num_rows
                yield batch

    def _write_manifest(
        staging_dir: Path,
        *,
        source: str,
        manifest_id: str,
        input_sha256: str,
        ingest_ts: datetime,
        n_rows: int,
        shards: list[ShardInfo],
        parquet_compression: str,
        parquet_compression_level: int,
    ) -> None:
        # `rows_per_file` is final here — run_single_file_ingest drains
        # `_stream` (via write_shards) before invoking the manifest writer.
        source_files = [
            {
                "path": str(relative_to_repo(path)),
                "sha256": file_shas[path],
                "n_rows": rows_per_file[path],
            }
            for path in files
        ]
        write_stage_manifest(
            staging_dir,
            source=source,
            manifest_id=manifest_id,
            input_sha256=input_sha256,
            ingest_ts=ingest_ts,
            n_rows=n_rows,
            shards=shards,
            parquet_compression=parquet_compression,
            parquet_compression_level=parquet_compression_level,
            extra={"source_files": source_files},
        )

    return run_single_file_ingest(
        cfg,
        stream_batches=_stream,
        ingest_ts=ingest_ts,
        input_sha256=input_sha256,
        manifest_id=cfg.name,
        write_manifest=_write_manifest,
    )
