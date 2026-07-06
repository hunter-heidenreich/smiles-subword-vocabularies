"""Behavioral tests for Stage 0 ZINC-22 ingest."""

from __future__ import annotations

import hashlib
import shutil
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from smiles_subword.config import Zinc22CorpusConfig
from smiles_subword.ingest._common import RAW_V1_SCHEMA
from smiles_subword.ingest.zinc22 import fetch_tranche, ingest

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def test_schema_matches_raw_v1_when_ingesting_synthetic_tranche(
    zinc22_smoke_config: Zinc22CorpusConfig,
    zinc22_manifest: Path,
    shard_paths: Callable[[Path], list[Path]],
) -> None:
    ingest(zinc22_smoke_config, fetch=False, manifest_path=zinc22_manifest)

    for shard in shard_paths(zinc22_smoke_config.output_dir):
        assert pq.ParquetFile(shard).schema_arrow.equals(RAW_V1_SCHEMA)


def test_source_column_is_zinc22_when_ingesting_synthetic_tranche(
    zinc22_smoke_config: Zinc22CorpusConfig,
    zinc22_manifest: Path,
    shard_columns: Callable[[Path], dict[str, list]],
) -> None:
    ingest(zinc22_smoke_config, fetch=False, manifest_path=zinc22_manifest)

    cols = shard_columns(zinc22_smoke_config.output_dir)
    assert set(cols["source"]) == {"zinc22"}


def test_source_id_is_zinc_id_when_ingesting_synthetic_tranche(
    zinc22_smoke_config: Zinc22CorpusConfig,
    zinc22_manifest: Path,
    shard_columns: Callable[[Path], dict[str, list]],
) -> None:
    ingest(zinc22_smoke_config, fetch=False, manifest_path=zinc22_manifest)

    cols = shard_columns(zinc22_smoke_config.output_dir)
    assert all(sid.startswith("ZINChq") for sid in cols["source_id"])
    assert cols["source_id"][0] == "ZINChq0000066mDY"
    assert cols["smiles"][0].startswith("CC(C)")


def test_row_count_preserved_when_ingesting_synthetic_tranche(
    zinc22_smoke_config: Zinc22CorpusConfig,
    zinc22_manifest: Path,
    shard_paths: Callable[[Path], list[Path]],
) -> None:
    result = ingest(zinc22_smoke_config, fetch=False, manifest_path=zinc22_manifest)

    assert result.n_rows == 5
    on_disk = sum(
        pq.ParquetFile(s).metadata.num_rows
        for s in shard_paths(zinc22_smoke_config.output_dir)
    )
    assert on_disk == 5


def test_deterministic_when_rerun(
    tmp_path: Path,
    zinc22_smi_gz: Path,
    zinc22_manifest: Path,
    shard_paths: Callable[[Path], list[Path]],
) -> None:
    cfg_a = Zinc22CorpusConfig(
        name="zinc22-smoke",
        source="zinc22",
        manifest_id="zinc22-test-fixture",
        tranche_id="test-fixture",
        tranche_url="https://example.invalid/test-fixture.smi.gz",
        transport="curl",
        raw_path=zinc22_smi_gz,
        output_dir=tmp_path / "out_a",
    )
    cfg_b = cfg_a.model_copy(update={"output_dir": tmp_path / "out_b"})

    ingest(cfg_a, fetch=False, manifest_path=zinc22_manifest)
    ingest(cfg_b, fetch=False, manifest_path=zinc22_manifest)

    table_a = pa.concat_tables(pq.read_table(s) for s in shard_paths(cfg_a.output_dir))
    table_b = pa.concat_tables(pq.read_table(s) for s in shard_paths(cfg_b.output_dir))
    assert (
        table_a.column("source_id").to_pylist()
        == table_b.column("source_id").to_pylist()
    )
    assert table_a.column("smiles").to_pylist() == table_b.column("smiles").to_pylist()


def test_per_stage_manifest_written_when_ingest_completes(
    zinc22_smoke_config: Zinc22CorpusConfig,
    zinc22_manifest: Path,
) -> None:
    result = ingest(zinc22_smoke_config, fetch=False, manifest_path=zinc22_manifest)

    payload = yaml.safe_load(
        (zinc22_smoke_config.output_dir / "MANIFEST.yaml").read_text()
    )
    assert payload["schema"] == "raw_v1"
    assert payload["source"] == "zinc22"
    assert payload["manifest_id"] == "zinc22-test-fixture"
    assert payload["n_rows"] == result.n_rows
    assert payload["n_shards"] == result.n_shards
    for shard_meta in payload["shards"]:
        on_disk = zinc22_smoke_config.output_dir / shard_meta["file"]
        assert hashlib.sha256(on_disk.read_bytes()).hexdigest() == shard_meta["sha256"]


def test_atomic_output_when_prior_dir_exists(
    zinc22_smoke_config: Zinc22CorpusConfig,
    zinc22_manifest: Path,
    shard_paths: Callable[[Path], list[Path]],
) -> None:
    zinc22_smoke_config.output_dir.mkdir(parents=True)
    (zinc22_smoke_config.output_dir / "stale.parquet").write_bytes(b"junk")

    ingest(zinc22_smoke_config, fetch=False, manifest_path=zinc22_manifest)

    assert not (zinc22_smoke_config.output_dir / "stale.parquet").exists()
    assert shard_paths(zinc22_smoke_config.output_dir)


def test_records_manifest_entry_on_first_observation(
    zinc22_smoke_config: Zinc22CorpusConfig,
    zinc22_manifest: Path,
) -> None:
    ingest(zinc22_smoke_config, fetch=False, manifest_path=zinc22_manifest)

    payload = yaml.safe_load(zinc22_manifest.read_text())
    assert len(payload["artifacts"]) == 1
    entry = payload["artifacts"][0]
    assert entry["id"] == zinc22_smoke_config.manifest_id
    assert len(entry["sha256"]) == 64
    assert entry["size_bytes"] == zinc22_smoke_config.raw_path.stat().st_size
    assert "first observation" in entry["notes"].lower()


def test_accepts_when_recorded_sha_matches_on_rerun(
    zinc22_smoke_config: Zinc22CorpusConfig,
    zinc22_manifest: Path,
    tmp_path: Path,
) -> None:
    # First run records the observed SHA (first observation). A second run with
    # verify on must load that entry, find it matches, and proceed without
    # raising — the documented "subsequent runs verify against the recorded
    # value" success path, which only the mismatch branch was pinning.
    ingest(zinc22_smoke_config, fetch=False, manifest_path=zinc22_manifest)
    cfg_b = zinc22_smoke_config.model_copy(update={"output_dir": tmp_path / "out_b"})

    result = ingest(cfg_b, fetch=False, manifest_path=zinc22_manifest)

    assert result.n_rows == 5
    # Idempotent: the matching rerun neither raised nor duplicated the entry.
    payload = yaml.safe_load(zinc22_manifest.read_text())
    assert len(payload["artifacts"]) == 1


def test_fetch_triggered_when_raw_path_missing(
    zinc22_smoke_config: Zinc22CorpusConfig,
    zinc22_manifest: Path,
    zinc22_smi_gz: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # fetch=True with a missing raw_path must call fetch_tranche to stage the
    # file before streaming — the download-trigger branch the fetch=False tests
    # never take.
    target = tmp_path / "fetched" / "tranche.smi.gz"
    cfg = zinc22_smoke_config.model_copy(update={"raw_path": target})
    calls: list[Path] = []

    def _fake_fetch(c: Zinc22CorpusConfig) -> Path:
        calls.append(c.raw_path)
        shutil.copy(zinc22_smi_gz, c.raw_path)
        return c.raw_path

    monkeypatch.setattr("smiles_subword.ingest.zinc22.fetch_tranche", _fake_fetch)

    result = ingest(cfg, fetch=True, manifest_path=zinc22_manifest)

    assert calls == [target]
    assert result.n_rows == 5


def test_rejects_when_recorded_sha_mismatches(
    zinc22_smoke_config: Zinc22CorpusConfig,
    tmp_path: Path,
) -> None:
    bogus_manifest = tmp_path / "MANIFEST.yaml"
    bogus_manifest.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "artifacts": [
                    {
                        "id": zinc22_smoke_config.manifest_id,
                        "path": str(zinc22_smoke_config.raw_path),
                        "source_url": zinc22_smoke_config.tranche_url,
                        "sha256": "0" * 64,
                        "size_bytes": zinc22_smoke_config.raw_path.stat().st_size,
                        "ingest_date": "2026-04-27",
                    }
                ],
            }
        )
    )

    with pytest.raises(ValueError, match=r"sha256"):
        ingest(zinc22_smoke_config, fetch=False, manifest_path=bogus_manifest)


def test_empty_input_when_zero_rows(
    tmp_path: Path,
    zinc22_empty_smi_gz: Path,
    zinc22_manifest: Path,
    shard_paths: Callable[[Path], list[Path]],
) -> None:
    cfg = Zinc22CorpusConfig(
        name="zinc22-smoke",
        source="zinc22",
        manifest_id="zinc22-empty-fixture",
        tranche_id="empty-fixture",
        tranche_url="https://example.invalid/empty.smi.gz",
        transport="curl",
        raw_path=zinc22_empty_smi_gz,
        output_dir=tmp_path / "out",
    )

    result = ingest(cfg, fetch=False, manifest_path=zinc22_manifest)

    assert result.n_rows == 0
    assert result.n_shards == 0
    assert shard_paths(cfg.output_dir) == []
    assert (cfg.output_dir / "MANIFEST.yaml").exists()


def test_fetch_tranche_rejects_size_mismatch(
    zinc22_smoke_config: Zinc22CorpusConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A truncated-but-HTTP-200 download must be caught here; on first
    # observation there is no recorded SHA to verify against, so size is the
    # only integrity signal before the truncated bytes become canonical.
    target = tmp_path / "frag.smi.gz"
    cfg = zinc22_smoke_config.model_copy(
        update={"expected_bytes": 100, "raw_path": target}
    )

    def _fake_run(cmd: list[str], **_kwargs: object) -> None:
        target.write_bytes(b"truncated")  # 9 bytes != 100

    monkeypatch.setattr("smiles_subword.ingest.zinc22.subprocess.run", _fake_run)

    with pytest.raises(ValueError, match="download size mismatch"):
        fetch_tranche(cfg)


def test_fetch_tranche_accepts_matching_size(
    zinc22_smoke_config: Zinc22CorpusConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "frag.smi.gz"
    payload = b"a-tranche-of-exactly-this-many-bytes"
    cfg = zinc22_smoke_config.model_copy(
        update={"expected_bytes": len(payload), "raw_path": target}
    )

    def _fake_run(cmd: list[str], **_kwargs: object) -> None:
        target.write_bytes(payload)

    monkeypatch.setattr("smiles_subword.ingest.zinc22.subprocess.run", _fake_run)

    assert fetch_tranche(cfg) == target
