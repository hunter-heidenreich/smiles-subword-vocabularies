"""Shared raw_v1 helpers for Stage 0 ingest backends.

Defines the on-disk contract for raw_v1 Parquet shards:

    source_id: string, smiles: string, source: string, ingest_ts: timestamp[us]

Every backend writes through the same `write_shards` / `write_stage_manifest`
pair, so shard byte layout and manifest schema are byte-identical across sources.
The CSV/TSV backends share one config-driven reader (`build_csv_select_sql` +
`stream_csv_path`): every per-corpus difference is a config field, not a bespoke
SQL string.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, TypeVar

import duckdb
import pyarrow as pa
import yaml

from smiles_subword._hashing import sha256_file
from smiles_subword._io import atomic_output_dir
from smiles_subword._shards import ShardWriter
from smiles_subword._sql import quote_ident
from smiles_subword._time import utc_now_naive_seconds
from smiles_subword.ingest.types import IngestResult
from smiles_subword.manifest import ShardInfo, load_manifest_entry, shard_dicts
from smiles_subword.paths import REPO_ROOT

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from datetime import datetime
    from pathlib import Path

#: DuckDB ``memory_limit`` pragma applied to every ingest connection. It is an
#: upper bound on the in-memory working set, not a target — readers stream in
#: ``rows_per_batch`` chunks regardless.
DUCKDB_MEMORY_LIMIT = "8GB"

RAW_V1_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("smiles", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("ingest_ts", pa.timestamp("us"), nullable=False),
    ]
)


class _ShardingConfig(Protocol):
    shard_target_bytes: int
    parquet_compression: Literal["zstd", "snappy", "gzip"]
    parquet_compression_level: int


class _Verifiable(Protocol):
    @property
    def raw_path(self) -> Path: ...
    @property
    def manifest_id(self) -> str: ...


class _SingleFileConfig(_ShardingConfig, Protocol):
    @property
    def source(self) -> str: ...
    @property
    def output_dir(self) -> Path: ...


class _CsvReadConfig(Protocol):
    """The config surface the shared CSV reader needs (a `CsvFormatConfig`)."""

    @property
    def source(self) -> str: ...
    @property
    def smiles_column(self) -> str: ...
    @property
    def id_column(self) -> str: ...
    @property
    def id_column_type(self) -> Literal["VARCHAR", "BIGINT"]: ...
    @property
    def delim(self) -> str: ...
    @property
    def has_header(self) -> bool: ...
    @property
    def file_compression(self) -> Literal["gzip", "zstd", "none"]: ...
    @property
    def csv_read_mode(self) -> Literal["positional", "named"]: ...
    @property
    def positional_id_first(self) -> bool: ...
    @property
    def normalize_names(self) -> bool: ...
    @property
    def drop_null_smiles(self) -> bool: ...
    @property
    def coalesce_null_smiles(self) -> bool: ...
    @property
    def disable_quoting(self) -> bool: ...
    @property
    def rows_per_batch(self) -> int: ...


_CfgT = TypeVar("_CfgT", bound=_SingleFileConfig)


def ingest_timestamp() -> datetime:
    """Return the single per-call wall-clock stamp for the `ingest_ts` column."""
    return utc_now_naive_seconds()


def verify_input_sha(cfg: _Verifiable) -> str:
    """Hash `cfg.raw_path` and require it to match the manifest; return the SHA.

    Raises:
        ValueError: if `cfg.raw_path`'s SHA256 disagrees with the
            `data/MANIFEST.yaml` entry for `cfg.manifest_id`.
    """
    entry = load_manifest_entry(cfg.manifest_id)
    actual = sha256_file(cfg.raw_path)
    if actual != entry.sha256:
        raise ValueError(
            f"sha256 mismatch for {cfg.raw_path}: expected {entry.sha256}, got {actual}"
        )
    return actual


def relative_to_repo(path: Path) -> Path:
    """Return `path` relative to REPO_ROOT, or `path` itself if outside the tree."""
    try:
        return path.relative_to(REPO_ROOT)
    except ValueError:
        return path


def open_duckdb(
    *,
    memory_limit: str = DUCKDB_MEMORY_LIMIT,
    threads: int = 4,
) -> duckdb.DuckDBPyConnection:
    """Open an in-memory DuckDB connection with standard pragmas applied."""
    con = duckdb.connect(":memory:")
    con.execute(f"PRAGMA threads={threads}")
    con.execute(f"PRAGMA memory_limit='{memory_limit}'")
    return con


def stream_arrow_batches(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    params: Sequence[object],
    rows_per_batch: int,
) -> Iterator[pa.RecordBatch]:
    """Run `sql` and yield `RAW_V1_SCHEMA`-cast batches; close reader+con on exit."""
    reader = con.execute(sql, list(params)).to_arrow_reader(rows_per_batch)
    try:
        for batch in reader:
            yield batch.cast(RAW_V1_SCHEMA)
    finally:
        reader.close()
        con.close()


def build_csv_select_sql(cfg: _CsvReadConfig) -> str:
    """Build the raw_v1 projection SQL over a `read_csv(...)` for `cfg`.

    Two read modes:

    * ``positional`` — headerless / fixed 2-column files. Explicit
      ``columns={...}`` schema with ``auto_detect=false`` so types are pinned;
      optionally disables quote/escape so a SMILES with ``"`` / ``\\`` survives
      byte-for-byte.
    * ``named`` — header CSVs, extra columns dropped by name-projection.
      ``auto_detect=true``, two columns selected by (optionally normalized) name.

    Column identifiers are validated + quoted via `quote_ident`; every runtime
    value (path, delim, header, compression, source, ts) is bound as a ``?`` by
    `stream_csv_path`.
    """
    id_q = quote_ident(cfg.id_column)
    smiles_q = quote_ident(cfg.smiles_column)
    smiles_expr = f"COALESCE({smiles_q}, '')" if cfg.coalesce_null_smiles else smiles_q
    # Only CAST the id when its DuckDB type isn't already VARCHAR (named-mode
    # auto-detect, or a positional BIGINT id like PubChem's). A redundant CAST
    # over an already-VARCHAR column makes DuckDB's Arrow output encoding
    # nondeterministic across runs; the final RAW_V1_SCHEMA cast coerces type
    # regardless.
    needs_id_cast = cfg.csv_read_mode == "named" or cfg.id_column_type != "VARCHAR"
    id_expr = f"CAST({id_q} AS VARCHAR)" if needs_id_cast else id_q

    if cfg.csv_read_mode == "positional":
        # Dict order maps to file column order (DuckDB positional read), so a
        # smiles-first file (ZINC-22, REAL-Space) must declare smiles first.
        # Column names interpolate raw here (DuckDB's `columns` dict needs
        # single-quoted keys, not `quote_ident`'s double-quoted form) but are
        # already validated by the `quote_ident` calls above.
        id_decl = f"'{cfg.id_column}': '{cfg.id_column_type}'"
        smiles_decl = f"'{cfg.smiles_column}': 'VARCHAR'"
        ordered = (
            (id_decl, smiles_decl)
            if cfg.positional_id_first
            else (smiles_decl, id_decl)
        )
        columns = "{" + ", ".join(ordered) + "}"
        # The `?` clauses (path, delim, header, compression) fix the bind order
        # `stream_csv_path` follows.
        args = [
            "?",
            "delim=?",
            "header=?",
            "auto_detect=False",
            f"columns={columns}",
            "compression=?",
        ]
        if cfg.disable_quoting:
            args += ["quote=''", "escape=''"]
    else:
        args = ["?", "delim=?", "header=True", "auto_detect=True"]
        if cfg.normalize_names:
            args.append("normalize_names=True")
    reader = f"read_csv({', '.join(args)})"

    where = ""
    if cfg.drop_null_smiles:
        where = f"\n        WHERE {smiles_q} IS NOT NULL AND length({smiles_q}) > 0"

    return f"""
        SELECT
            {id_expr}                  AS source_id,
            {smiles_expr}              AS smiles,
            CAST(? AS VARCHAR)         AS source,
            CAST(? AS TIMESTAMP)       AS ingest_ts
        FROM {reader}{where}
    """


def stream_csv_path(
    cfg: _CsvReadConfig,
    path: Path,
    ingest_ts: datetime,
    *,
    threads: int = 4,
) -> Iterator[pa.RecordBatch]:
    """Stream one CSV/TSV `path` into RAW_V1 batches via the shared reader.

    Parameter order matches the ``?`` placeholders in `build_csv_select_sql`:
    the two SELECT scalars (source, ingest_ts), then the reader args (path,
    delim, [header,] compression). ``header``/``compression`` are bound only in
    positional mode.
    """
    sql = build_csv_select_sql(cfg)
    params: list[object] = [cfg.source, ingest_ts, str(path), cfg.delim]
    if cfg.csv_read_mode == "positional":
        params.extend([cfg.has_header, cfg.file_compression])
    yield from stream_arrow_batches(
        open_duckdb(threads=threads),
        sql,
        params,
        cfg.rows_per_batch,
    )


def write_shards(
    batches: Iterator[pa.RecordBatch],
    cfg: _ShardingConfig,
    staging_dir: Path,
) -> list[ShardInfo]:
    """Drain `batches` into byte-budgeted `raw_v1` Parquet shards via `ShardWriter`.

    Returns one `ShardInfo` per shard written; `cfg.shard_target_bytes` is a soft
    floor (see `ShardWriter` for the row-group buffering caveat).
    """
    writer = ShardWriter(
        staging_dir,
        schema=RAW_V1_SCHEMA,
        shard_prefix="raw_v1",
        target_bytes=cfg.shard_target_bytes,
        compression=cfg.parquet_compression,
        compression_level=cfg.parquet_compression_level,
    )
    for batch in batches:
        writer.write_batch(batch)
    writer.close_current()
    return writer.shards


def write_stage_manifest(
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
    extra: dict[str, object] | None = None,
) -> None:
    """Write the per-stage `MANIFEST.yaml` describing this raw_v1 directory.

    `extra` keys (e.g. REAL-Space's `source_files` provenance block) are spliced
    in just before the `shards` list, so a backend with richer provenance does
    not need to re-read and reorder the file after the fact.
    """
    payload: dict[str, object] = {
        "schema": "raw_v1",
        "source": source,
        "manifest_id": manifest_id,
        "input_sha256": input_sha256,
        "ingest_ts": ingest_ts.isoformat() + "Z",
        "n_rows": n_rows,
        "n_shards": len(shards),
        "parquet_compression": parquet_compression,
        "parquet_compression_level": parquet_compression_level,
    }
    if extra:
        payload.update(extra)
    payload["shards"] = shard_dicts(shards)
    with (staging_dir / "MANIFEST.yaml").open("w") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False)


def run_single_file_ingest(
    cfg: _CfgT,
    *,
    stream_batches: Callable[[_CfgT, datetime], Iterator[pa.RecordBatch]],
    ingest_ts: datetime,
    input_sha256: str,
    manifest_id: str,
    write_manifest: Callable[..., None] = write_stage_manifest,
) -> IngestResult:
    """Stage shards, write manifest, atomic-rename `cfg.output_dir` into place.

    `stream_batches` is the per-backend axis; it must yield batches conforming to
    `RAW_V1_SCHEMA`. `write_manifest` defaults to `write_stage_manifest`; a
    backend needing post-stream provenance (REAL-Space) passes a closure building
    the `extra` block at call time — `write_shards` has drained `stream_batches`
    by then, so per-file tallies are final.
    """
    with atomic_output_dir(cfg.output_dir, keep_on_error=False) as staging_dir:
        shards = write_shards(stream_batches(cfg, ingest_ts), cfg, staging_dir)
        n_rows = sum(s.n_rows for s in shards)
        write_manifest(
            staging_dir,
            source=cfg.source,
            manifest_id=manifest_id,
            input_sha256=input_sha256,
            ingest_ts=ingest_ts,
            n_rows=n_rows,
            shards=shards,
            parquet_compression=cfg.parquet_compression,
            parquet_compression_level=cfg.parquet_compression_level,
        )

    return IngestResult(
        n_rows=n_rows,
        output_dir=cfg.output_dir,
        ingest_ts=ingest_ts,
        shards=tuple(shards),
    )
