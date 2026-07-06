"""The `canon_dedup_v1` pipeline: raw_v1 -> canon_dedup_v1 Parquet shards.

The single uniform preprocessing policy for the study: RDKit isomeric
canonicalization (`isomericSmiles=True`, `kekuleSmiles=False`) then exact-string
dedup, and **nothing else** — no charge neutralization (would collapse the
charged bracketed glyphs `[O-]`, `[NH3+]`, zwitterions the study measures), no
salt-stripping, no heavy-atom cap (would clip the COCONUT macrocycles). It
composes only `canonicalize_minimal` (a pure RDKit round-trip) and the
window-dedup core; the config exposes no knob for those steps.

Rows RDKit cannot parse are dropped; the count and rate are recorded in the
per-stage `MANIFEST.yaml` alongside the resolved RDKit version (canonical output
is version-dependent).
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

import duckdb
from rdkit import rdBase

from smiles_subword._hashing import sha256_bytes
from smiles_subword.ingest._common import RAW_V1_SCHEMA
from smiles_subword.preprocess._dedup_core import (
    BUCKET_KEY_LABEL,
    apply_duckdb_pragmas,
    dedup_to_writer,
    partition_into_buckets,
)
from smiles_subword.preprocess._io import (
    ShardWriter,
    read_and_verify_source_manifest,
    shard_dicts,
    stage_run,
    write_manifest,
)
from smiles_subword.preprocess.canonicalize_minimal import canonicalize_minimal
from smiles_subword.preprocess.types import CanonDedupResult, ShardInfo

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

    from smiles_subword.config import CanonDedupConfig

__all__ = ["canon_dedup"]

_OUTPUT_COLUMNS: tuple[str, ...] = tuple(f.name for f in RAW_V1_SCHEMA)

_EXCLUDED_STEPS: tuple[str, ...] = (
    "salt_strip",
    "neutralize",
    "heavy_atom_cap",
    "descriptors",
    "decontaminate",
)


def canon_dedup(
    cfg: CanonDedupConfig, *, verify_input_sha: bool = True
) -> CanonDedupResult:
    """Run `canon_dedup_v1` over `cfg.input_dir` into `cfg.output_dir`.

    RDKit isomeric canonicalization + exact-string dedup, and nothing else.
    Output keeps the 4-field `raw_v1` schema.

    Args:
        cfg: validated `canon_dedup_v1` config.
        verify_input_sha: when True, hash every input shard and require it
            to match the SHA recorded in the source `MANIFEST.yaml`. Tests
            against synthetic fixtures may pass False.

    Raises:
        ValueError: if `verify_input_sha` is True and any input shard's
            SHA256 disagrees with the source manifest, or if
            `duckdb_memory_limit` is malformed.
        FileNotFoundError: if `cfg.input_dir/MANIFEST.yaml` is missing.
    """
    source_manifest_bytes, _ = read_and_verify_source_manifest(
        cfg.input_dir, verify=verify_input_sha
    )

    with stage_run(cfg.output_dir) as (staging_dir, started_ts):
        canonical_dir = staging_dir / "_canonical"
        canon = canonicalize_minimal(
            cfg.input_dir,
            canonical_dir,
            target_bytes=cfg.shard_target_bytes,
            n_workers=cfg.n_workers,
            rows_per_batch=cfg.rows_per_batch,
        )

        if cfg.mode == "bucket":
            shards, n_buckets = _dedup_canonical_bucket(cfg, canonical_dir, staging_dir)
        else:
            shards = _dedup_canonical(cfg, canonical_dir, staging_dir)
            n_buckets = None
        shutil.rmtree(canonical_dir)

        n_output_rows = sum(s.n_rows for s in shards)
        n_duplicates = canon.n_output_rows - n_output_rows
        rejection_rate = (
            canon.n_dropped_unparseable / canon.n_input_rows
            if canon.n_input_rows
            else 0.0
        )

        _write_stage_manifest(
            staging_dir,
            cfg=cfg,
            started_ts=started_ts,
            n_input_rows=canon.n_input_rows,
            n_rdkit_rejected=canon.n_dropped_unparseable,
            rejection_rate=rejection_rate,
            n_canonical_rows=canon.n_output_rows,
            n_duplicates=n_duplicates,
            n_output_rows=n_output_rows,
            source_manifest_bytes=source_manifest_bytes,
            shards=shards,
            n_buckets=n_buckets,
        )

    return CanonDedupResult(
        n_input_rows=canon.n_input_rows,
        n_rdkit_rejected=canon.n_dropped_unparseable,
        n_canonical_rows=canon.n_output_rows,
        n_duplicates=n_duplicates,
        n_output_rows=n_output_rows,
        rdkit_rejection_rate=rejection_rate,
        rdkit_version=rdBase.rdkitVersion,
        shards=tuple(shards),
        output_dir=cfg.output_dir,
        started_ts=started_ts,
    )


def _new_writer(cfg: CanonDedupConfig, staging_dir: Path) -> ShardWriter:
    return ShardWriter(
        staging_dir,
        schema=RAW_V1_SCHEMA,
        shard_prefix="canon_dedup_v1",
        target_bytes=cfg.shard_target_bytes,
        compression=cfg.parquet_compression,
        compression_level=cfg.parquet_compression_level,
    )


def _dedup_canonical(
    cfg: CanonDedupConfig, canonical_dir: Path, staging_dir: Path
) -> list[ShardInfo]:
    writer = _new_writer(cfg, staging_dir)
    if not sorted(canonical_dir.glob("canonical_v1-*.parquet")):
        return writer.shards

    con = duckdb.connect()
    try:
        apply_duckdb_pragmas(
            con, threads=cfg.duckdb_threads, memory_limit=cfg.duckdb_memory_limit
        )
        dedup_to_writer(
            con,
            parquet_source=str(canonical_dir / "canonical_v1-*.parquet"),
            output_columns=_OUTPUT_COLUMNS,
            writer=writer,
            out_schema=RAW_V1_SCHEMA,
            rows_per_batch=cfg.rows_per_batch,
        )
    finally:
        con.close()

    writer.close_current()
    return writer.shards


def _dedup_canonical_bucket(
    cfg: CanonDedupConfig, canonical_dir: Path, staging_dir: Path
) -> tuple[list[ShardInfo], int]:
    writer = _new_writer(cfg, staging_dir)
    input_shards = sorted(canonical_dir.glob("canonical_v1-*.parquet"))
    if not input_shards:
        return writer.shards, 0

    bucket_dir = staging_dir / "_buckets"
    bucket_dir.mkdir()
    bucket_paths = partition_into_buckets(
        input_shards,
        bucket_dir,
        rows_per_batch=cfg.rows_per_batch,
        out_schema=RAW_V1_SCHEMA,
        parquet_compression=cfg.parquet_compression,
        parquet_compression_level=cfg.parquet_compression_level,
    )
    if not bucket_paths:
        shutil.rmtree(bucket_dir)
        return writer.shards, 0

    con = duckdb.connect()
    try:
        apply_duckdb_pragmas(
            con, threads=cfg.duckdb_threads, memory_limit=cfg.duckdb_memory_limit
        )
        for key in sorted(bucket_paths):
            dedup_to_writer(
                con,
                parquet_source=str(bucket_paths[key]),
                output_columns=_OUTPUT_COLUMNS,
                writer=writer,
                out_schema=RAW_V1_SCHEMA,
                rows_per_batch=cfg.rows_per_batch,
            )
    finally:
        con.close()

    writer.close_current()
    shutil.rmtree(bucket_dir)
    return writer.shards, len(bucket_paths)


def _write_stage_manifest(
    staging_dir: Path,
    *,
    cfg: CanonDedupConfig,
    started_ts: datetime,
    n_input_rows: int,
    n_rdkit_rejected: int,
    rejection_rate: float,
    n_canonical_rows: int,
    n_duplicates: int,
    n_output_rows: int,
    source_manifest_bytes: bytes,
    shards: list[ShardInfo],
    n_buckets: int | None,
) -> None:
    payload: dict[str, object] = {
        "schema": "canon_dedup_v1",
        "name": cfg.name,
        "source_manifest_sha256": sha256_bytes(source_manifest_bytes),
        "tie_break": "smallest_source_id",
        "mode": cfg.mode,
        "rdkit_version": rdBase.rdkitVersion,
        "canonicalization": {"isomeric_smiles": True, "kekule_smiles": False},
        "excluded_steps": list(_EXCLUDED_STEPS),
        "started_ts": started_ts.isoformat() + "Z",
        "n_input_rows": n_input_rows,
        "n_rdkit_rejected": n_rdkit_rejected,
        "rdkit_rejection_rate": rejection_rate,
        "n_canonical_rows": n_canonical_rows,
        "n_duplicates": n_duplicates,
        "n_output_rows": n_output_rows,
        "n_shards": len(shards),
        "parquet_compression": cfg.parquet_compression,
        "parquet_compression_level": cfg.parquet_compression_level,
    }
    if cfg.mode == "bucket":
        payload["bucket_key"] = BUCKET_KEY_LABEL
        payload["n_buckets"] = n_buckets
    payload["shards"] = shard_dicts(shards)
    write_manifest(staging_dir, payload)
