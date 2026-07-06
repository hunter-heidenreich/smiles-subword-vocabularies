"""Shared fixtures for the corpus-prep (`canon_dedup_v1`) test suites."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from smiles_subword.config import HashSubsampleConfig, HoldoutSplitConfig
from smiles_subword.ingest._common import RAW_V1_SCHEMA


def _write_raw_v1_parquet(path: Path, rows: tuple[tuple[str, str], ...]) -> None:
    ts = datetime(2026, 4, 26, 12, 0, 0, tzinfo=UTC)
    table = pa.table(
        {
            "source_id": [r[0] for r in rows],
            "smiles": [r[1] for r in rows],
            "source": ["pubchem"] * len(rows),
            "ingest_ts": [ts] * len(rows),
        },
        schema=RAW_V1_SCHEMA,
    )
    pq.write_table(table, path, compression="zstd")


def _write_canon_dedup_v1_manifest(input_dir: Path, shards: list[Path]) -> None:
    n_rows = sum(pq.ParquetFile(s).metadata.num_rows for s in shards)
    payload = {
        "schema": "canon_dedup_v1",
        "name": "fx",
        "source_manifest_sha256": "0" * 64,
        "tie_break": "smallest_source_id",
        "mode": "single_pass",
        "rdkit_version": "2026.03.1",
        "started_ts": "2026-05-16T12:00:00Z",
        "n_input_rows": n_rows,
        "n_rdkit_rejected": 0,
        "rdkit_rejection_rate": 0.0,
        "n_canonical_rows": n_rows,
        "n_duplicates": 0,
        "n_output_rows": n_rows,
        "n_shards": len(shards),
        "parquet_compression": "zstd",
        "parquet_compression_level": 3,
        "shards": [
            {
                "file": s.name,
                "sha256": hashlib.sha256(s.read_bytes()).hexdigest(),
                "n_rows": pq.ParquetFile(s).metadata.num_rows,
                "n_bytes": s.stat().st_size,
            }
            for s in shards
        ],
    }
    with (input_dir / "MANIFEST.yaml").open("w") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False)


@pytest.fixture
def make_canon_dedup_v1_dir(tmp_path: Path) -> Callable[..., Path]:
    """Build a synthetic `canon_dedup_v1` directory (raw_v1 schema) + manifest.

    `rows` is a flat list of `(source_id, smiles)` tuples (one shard) or a
    list of such lists (one entry per shard). An empty list writes a
    directory with zero parquet shards (manifest only).
    """

    def _make(
        rows: list[tuple[str, str]] | list[list[tuple[str, str]]],
        *,
        name: str = "canon_dedup_v1",
    ) -> Path:
        input_dir = tmp_path / name
        input_dir.mkdir()
        if not rows:
            shard_groups: list[list[tuple[str, str]]] = []
        elif isinstance(rows[0], tuple):
            shard_groups = [rows]  # type: ignore[list-item]
        else:
            shard_groups = rows  # type: ignore[assignment]

        shards: list[Path] = []
        for i, recs in enumerate(shard_groups):
            shard = input_dir / f"canon_dedup_v1-{i:05d}.parquet"
            _write_raw_v1_parquet(shard, tuple(recs))
            shards.append(shard)
        _write_canon_dedup_v1_manifest(input_dir, shards)
        return input_dir

    return _make


@pytest.fixture
def make_hash_subsample_config(tmp_path: Path) -> Callable[..., HashSubsampleConfig]:
    """Build a `HashSubsampleConfig` for an existing canon_dedup_v1 directory."""

    def _make(input_dir: Path, **overrides: Any) -> HashSubsampleConfig:
        defaults: dict[str, Any] = {
            "name": "fx",
            "input_dir": input_dir,
            "output_dir": tmp_path / "canon_dedup_v1_sub",
            "target_n": 8,
            "rows_per_batch": 4,
            "shard_target_bytes": 2**20,
        }
        defaults.update(overrides)
        return HashSubsampleConfig(**defaults)

    return _make


@pytest.fixture
def make_holdout_split_config(tmp_path: Path) -> Callable[..., HoldoutSplitConfig]:
    """Build a `HoldoutSplitConfig` for an existing canon_dedup_v1 directory."""

    def _make(input_dir: Path, **overrides: Any) -> HoldoutSplitConfig:
        defaults: dict[str, Any] = {
            "name": "fx",
            "input_dir": input_dir,
            "output_dir": tmp_path / "canon_dedup_v1",
            "rows_per_batch": 4,
            "shard_target_bytes": 2**20,
        }
        defaults.update(overrides)
        return HoldoutSplitConfig(**defaults)

    return _make
