"""Tests for ``smiles_subword.preprocess.canonicalize_minimal``."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from smiles_subword.ingest._common import RAW_V1_SCHEMA
from smiles_subword.preprocess.canonicalize_minimal import canonicalize_minimal

if TYPE_CHECKING:
    from pathlib import Path


def _write_raw_v1_shard(path: Path, rows: list[tuple[str, str, str]]) -> None:
    ts = datetime(2026, 5, 3, tzinfo=UTC).replace(tzinfo=None)
    table = pa.table(
        {
            "source_id": [r[0] for r in rows],
            "smiles": [r[1] for r in rows],
            "source": [r[2] for r in rows],
            "ingest_ts": [ts] * len(rows),
        },
        schema=RAW_V1_SCHEMA,
    )
    pq.write_table(table, path)


def _write_raw_v1_dir(
    tmp_path: Path,
    rows: list[tuple[str, str, str]],
    *,
    source: str = "smoke",
) -> Path:
    raw_dir = tmp_path / "raw_v1"
    raw_dir.mkdir()
    _write_raw_v1_shard(raw_dir / "raw_v1-000.parquet", rows)
    manifest = {
        "schema": "raw_v1",
        "source": source,
        "shards": [
            {
                "file": "raw_v1-000.parquet",
                "n_rows": len(rows),
            }
        ],
    }
    (raw_dir / "MANIFEST.yaml").write_text(yaml.safe_dump(manifest))
    return raw_dir


class TestCanonicalizeMinimal:
    """Per-row canonicalization, drop-unparseable, manifest provenance."""

    def test_canonicalizes_valid_smiles_in_place(self, tmp_path: Path) -> None:
        raw_dir = _write_raw_v1_dir(
            tmp_path,
            [
                ("smoke-1", "OCC", "smoke"),
                ("smoke-2", "C(O)C", "smoke"),
            ],
        )
        out_dir = tmp_path / "canonical_v1"

        result = canonicalize_minimal(raw_dir, out_dir)

        assert result.n_input_rows == 2
        assert result.n_output_rows == 2
        assert result.n_dropped_unparseable == 0

    def test_drops_unparseable_smiles(self, tmp_path: Path) -> None:
        raw_dir = _write_raw_v1_dir(
            tmp_path,
            [
                ("smoke-1", "CCO", "smoke"),
                ("smoke-2", "not-a-smiles", "smoke"),
                ("smoke-3", "c1ccccc1", "smoke"),
            ],
        )
        out_dir = tmp_path / "canonical_v1"

        result = canonicalize_minimal(raw_dir, out_dir)

        assert result.n_input_rows == 3
        assert result.n_dropped_unparseable == 1
        assert result.n_output_rows == 2

    def test_preserves_raw_v1_schema(self, tmp_path: Path) -> None:
        raw_dir = _write_raw_v1_dir(
            tmp_path,
            [("smoke-1", "CCO", "smoke")],
        )
        out_dir = tmp_path / "canonical_v1"

        canonicalize_minimal(raw_dir, out_dir)
        shard = next(out_dir.glob("canonical_v1*.parquet"))

        assert pq.ParquetFile(shard).schema_arrow.equals(RAW_V1_SCHEMA)

    def test_writes_manifest_with_dropped_count(self, tmp_path: Path) -> None:
        raw_dir = _write_raw_v1_dir(
            tmp_path,
            [
                ("smoke-1", "CCO", "smoke"),
                ("smoke-2", "broken", "smoke"),
            ],
        )
        out_dir = tmp_path / "canonical_v1"

        canonicalize_minimal(raw_dir, out_dir)
        manifest = yaml.safe_load((out_dir / "MANIFEST.yaml").read_text())

        assert manifest["schema"] == "canonical_v1_minimal"
        assert manifest["n_dropped_unparseable"] == 1
        assert manifest["n_input_rows"] == 2
        assert manifest["n_output_rows"] == 1

    def test_canonical_smiles_are_isomeric(self, tmp_path: Path) -> None:
        raw_dir = _write_raw_v1_dir(
            tmp_path,
            [("smoke-1", "N[C@@H](C)C(=O)O", "smoke")],
        )
        out_dir = tmp_path / "canonical_v1"

        canonicalize_minimal(raw_dir, out_dir)
        shard = next(out_dir.glob("canonical_v1*.parquet"))
        smi = pq.read_table(shard).column("smiles").to_pylist()[0]

        assert "@" in smi

    def test_empty_input_dir_raises(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw_v1"
        raw_dir.mkdir()
        (raw_dir / "MANIFEST.yaml").write_text(yaml.safe_dump({"schema": "raw_v1"}))

        with pytest.raises(FileNotFoundError, match=r"no parquet shards"):
            canonicalize_minimal(raw_dir, tmp_path / "canonical_v1")

    def test_all_invalid_yields_empty_output(self, tmp_path: Path) -> None:
        raw_dir = _write_raw_v1_dir(
            tmp_path,
            [
                ("smoke-1", "broken", "smoke"),
                ("smoke-2", "alsobroken", "smoke"),
            ],
        )
        out_dir = tmp_path / "canonical_v1"

        result = canonicalize_minimal(raw_dir, out_dir)

        assert result.n_input_rows == 2
        assert result.n_output_rows == 0
        assert result.n_dropped_unparseable == 2
        assert result.n_shards == 0

    def test_parallel_run_is_byte_identical_to_serial(self, tmp_path: Path) -> None:
        # The docstring promises the pool path produces byte-identical shards to
        # the serial path (futures drain in submission order); since canonical_v1
        # shards are sha256'd into the manifest, pin the byte claim, not just
        # table content. The batch/worker counts force mid-stream backpressure.
        cycle = ["CCO", "c1ccccc1", "CC(=O)O", "not a smiles", "CCN"]
        rows = [(f"id-{i:04d}", cycle[i % len(cycle)], "smoke") for i in range(180)]
        raw_dir = _write_raw_v1_dir(tmp_path, rows)

        serial = canonicalize_minimal(
            raw_dir, tmp_path / "serial", n_workers=1, rows_per_batch=16
        )
        parallel = canonicalize_minimal(
            raw_dir, tmp_path / "parallel", n_workers=4, rows_per_batch=16
        )

        def _shard_bytes(name: str) -> list[bytes]:
            shards = sorted((tmp_path / name).glob("canonical_v1-*.parquet"))
            return [s.read_bytes() for s in shards]

        assert _shard_bytes("serial") == _shard_bytes("parallel")
        assert serial.n_dropped_unparseable == parallel.n_dropped_unparseable

    def test_drops_empty_smiles(self, tmp_path: Path) -> None:
        # ingest's coalesce_null_smiles turns a null SMILES into "", which the
        # canonicalizer drops via its empty-string guard (not a parse failure).
        raw_dir = _write_raw_v1_dir(
            tmp_path,
            [("smoke-1", "CCO", "smoke"), ("smoke-2", "", "smoke")],
        )
        out_dir = tmp_path / "canonical_v1"

        result = canonicalize_minimal(raw_dir, out_dir)

        assert result.n_input_rows == 2
        assert result.n_dropped_unparseable == 1
        assert result.n_output_rows == 1
