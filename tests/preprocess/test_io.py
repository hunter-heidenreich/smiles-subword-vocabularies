"""Direct tests for `preprocess._io` helpers the stage suites only hit indirectly."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pyarrow as pa
import pytest
import yaml

from smiles_subword.preprocess._io import (
    coerce_to_schema,
    read_and_verify_source_manifest,
    stage_run,
    verify_shard_sha256,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_stage_run_clears_stale_staging_dir(tmp_path: Path) -> None:
    # A crashed prior run leaves an `<output>.tmp` behind; stage_run must wipe it
    # on entry so the new run starts clean, then atomic-rename into place.
    out = tmp_path / "out"
    stale = tmp_path / "out.tmp"
    stale.mkdir()
    (stale / "leftover.txt").write_text("from a crashed run")

    with stage_run(out) as (staging_dir, _started):
        assert staging_dir == stale
        assert not (staging_dir / "leftover.txt").exists()
        (staging_dir / "fresh.txt").write_text("ok")

    assert not stale.exists()
    assert (out / "fresh.txt").read_text() == "ok"


def test_stage_run_leaves_staging_dir_on_exception(tmp_path: Path) -> None:
    # On failure the staging dir is left in place for inspection (no rename), so
    # the output_dir is never created from a partial run.
    out = tmp_path / "out"

    def _crash() -> None:
        with stage_run(out) as (staging_dir, _started):
            (staging_dir / "partial.txt").write_text("half-written")
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        _crash()

    assert (tmp_path / "out.tmp" / "partial.txt").exists()
    assert not out.exists()


# --- verify_shard_sha256 -----------------------------------------------------


def test_verify_shard_sha256_accepts_matching_digest(tmp_path: Path) -> None:
    shard = tmp_path / "shard.parquet"
    shard.write_bytes(b"some bytes")
    digest = hashlib.sha256(b"some bytes").hexdigest()
    verify_shard_sha256([(shard, digest)])  # must not raise


def test_verify_shard_sha256_raises_on_mismatch(tmp_path: Path) -> None:
    shard = tmp_path / "shard.parquet"
    shard.write_bytes(b"some bytes")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        verify_shard_sha256([(shard, "0" * 64)])


# --- coerce_to_schema --------------------------------------------------------


def test_coerce_to_schema_is_noop_when_equal() -> None:
    schema = pa.schema([("a", pa.int64())])
    batch = pa.record_batch([pa.array([1, 2, 3])], schema=schema)
    # Equal schema -> the same object back (no copy).
    assert coerce_to_schema(batch, schema) is batch


def test_coerce_to_schema_casts_differing_types() -> None:
    src = pa.record_batch([pa.array([1, 2, 3], pa.int64())], names=["a"])
    target = pa.schema([("a", pa.string())])

    out = coerce_to_schema(src, target)

    assert out.schema.equals(target)
    assert out.column("a").to_pylist() == ["1", "2", "3"]


def test_coerce_to_schema_projects_to_schema_fields_by_name() -> None:
    # Columns absent from the schema are dropped; selection is by name.
    src = pa.record_batch([pa.array([1]), pa.array(["x"])], names=["a", "extra"])
    target = pa.schema([("a", pa.int64())])

    out = coerce_to_schema(src, target)

    assert out.schema.names == ["a"]
    assert out.column("a").to_pylist() == [1]


# --- read_and_verify_source_manifest -----------------------------------------

_SHARD_BYTES = b"shard bytes"


def _make_source(input_dir: Path, *, recorded_sha: str) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "data-00000.parquet").write_bytes(_SHARD_BYTES)
    (input_dir / "MANIFEST.yaml").write_text(
        yaml.safe_dump(
            {"shards": [{"file": "data-00000.parquet", "sha256": recorded_sha}]}
        )
    )


def test_read_and_verify_skips_check_and_returns_when_not_verifying(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "in"
    _make_source(input_dir, recorded_sha="0" * 64)  # deliberately wrong

    raw, manifest = read_and_verify_source_manifest(input_dir, verify=False)

    assert isinstance(raw, bytes)  # bytes kept for provenance hashing
    assert manifest["shards"][0]["sha256"] == "0" * 64  # not checked when verify=False


def test_read_and_verify_raises_on_bad_shard_when_verifying(tmp_path: Path) -> None:
    input_dir = tmp_path / "in"
    _make_source(input_dir, recorded_sha="0" * 64)

    with pytest.raises(ValueError, match="sha256 mismatch"):
        read_and_verify_source_manifest(input_dir, verify=True)


def test_read_and_verify_raises_when_manifest_missing(tmp_path: Path) -> None:
    # The documented FileNotFoundError shared by the subsample / split /
    # canon_dedup stages; it fires before the verify gate (bytes are read first).
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        read_and_verify_source_manifest(input_dir, verify=False)
