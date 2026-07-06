"""Behavioral tests for Stage 0 PubChem ingest."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from smiles_subword.config import CorpusConfig
from smiles_subword.ingest._common import RAW_V1_SCHEMA
from smiles_subword.ingest.csv_corpus import ingest

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def test_schema_matches_raw_v1_when_ingesting_mini_fixture(
    mini_corpus_config: CorpusConfig,
    shard_paths: Callable[[Path], list[Path]],
) -> None:
    ingest(mini_corpus_config, verify_input_sha=False)

    for shard in shard_paths(mini_corpus_config.output_dir):
        assert pq.ParquetFile(shard).schema_arrow.equals(RAW_V1_SCHEMA)


def test_row_count_preserved_when_ingesting_mini_fixture(
    mini_corpus_config: CorpusConfig,
    shard_paths: Callable[[Path], list[Path]],
) -> None:
    result = ingest(mini_corpus_config, verify_input_sha=False)

    assert result.n_rows == 1000
    on_disk = sum(
        pq.ParquetFile(s).metadata.num_rows
        for s in shard_paths(mini_corpus_config.output_dir)
    )
    assert on_disk == 1000


def test_smiles_preserved_when_first_row_looks_numeric(
    mini_corpus_config: CorpusConfig,
    shard_columns: Callable[[Path], dict[str, list]],
) -> None:
    ingest(mini_corpus_config, verify_input_sha=False)

    cols = shard_columns(mini_corpus_config.output_dir)
    assert cols["smiles"][0] == "1"
    assert cols["source_id"][0] == "1"


def test_source_column_constant_when_ingesting(
    mini_corpus_config: CorpusConfig,
    shard_columns: Callable[[Path], dict[str, list]],
) -> None:
    ingest(mini_corpus_config, verify_input_sha=False)

    cols = shard_columns(mini_corpus_config.output_dir)
    assert set(cols["source"]) == {"pubchem"}


def test_ingest_ts_constant_within_run(
    mini_corpus_config: CorpusConfig,
    shard_columns: Callable[[Path], dict[str, list]],
) -> None:
    ingest(mini_corpus_config, verify_input_sha=False)

    cols = shard_columns(mini_corpus_config.output_dir)
    assert len(set(cols["ingest_ts"])) == 1


def test_multiple_shards_when_size_threshold_low(
    multi_shard_corpus_config: CorpusConfig,
    shard_paths: Callable[[Path], list[Path]],
) -> None:
    result = ingest(multi_shard_corpus_config, verify_input_sha=False)

    shards = shard_paths(multi_shard_corpus_config.output_dir)
    assert len(shards) >= 2
    assert result.n_shards == len(shards)
    assert [s.name for s in shards] == [
        f"raw_v1-{i:05d}.parquet" for i in range(len(shards))
    ]


def test_single_shard_when_size_threshold_large(
    mini_corpus_config: CorpusConfig,
) -> None:
    result = ingest(mini_corpus_config, verify_input_sha=False)

    assert result.n_shards == 1


def test_deterministic_when_rerun(
    tmp_path: Path,
    mini_input: Path,
    shard_paths: Callable[[Path], list[Path]],
) -> None:
    cfg_a = CorpusConfig(
        name="pubchem",
        source="pubchem",
        manifest_id="pubchem-cid-smiles",
        raw_path=mini_input,
        output_dir=tmp_path / "out_a",
        shard_target_bytes=2**12,
        rows_per_batch=128,
    )
    cfg_b = cfg_a.model_copy(update={"output_dir": tmp_path / "out_b"})

    res_a = ingest(cfg_a, verify_input_sha=False)
    res_b = ingest(cfg_b, verify_input_sha=False)

    assert res_a.n_shards == res_b.n_shards
    table_a = pa.concat_tables(pq.read_table(s) for s in shard_paths(cfg_a.output_dir))
    table_b = pa.concat_tables(pq.read_table(s) for s in shard_paths(cfg_b.output_dir))
    assert (
        table_a.column("source_id").to_pylist()
        == table_b.column("source_id").to_pylist()
    )
    assert table_a.column("smiles").to_pylist() == table_b.column("smiles").to_pylist()


def test_rejects_when_manifest_sha_mismatch(
    mini_corpus_config: CorpusConfig,
) -> None:
    with pytest.raises(ValueError, match=r"sha256"):
        ingest(mini_corpus_config, verify_input_sha=True)


def test_empty_input_when_zero_rows(
    tmp_path: Path,
    empty_input: Path,
    shard_paths: Callable[[Path], list[Path]],
) -> None:
    cfg = CorpusConfig(
        name="pubchem",
        source="pubchem",
        manifest_id="pubchem-cid-smiles",
        raw_path=empty_input,
        output_dir=tmp_path / "out",
        shard_target_bytes=2**20,
        rows_per_batch=1024,
    )

    result = ingest(cfg, verify_input_sha=False)

    assert result.n_rows == 0
    assert result.n_shards == 0
    assert shard_paths(cfg.output_dir) == []
    assert (cfg.output_dir / "MANIFEST.yaml").exists()


def test_per_stage_manifest_written_when_ingest_completes(
    mini_corpus_config: CorpusConfig,
) -> None:
    result = ingest(mini_corpus_config, verify_input_sha=False)

    manifest_path = mini_corpus_config.output_dir / "MANIFEST.yaml"
    payload = yaml.safe_load(manifest_path.read_text())

    assert payload["schema"] == "raw_v1"
    assert payload["source"] == "pubchem"
    assert payload["n_rows"] == result.n_rows
    assert payload["n_shards"] == result.n_shards
    assert len(payload["shards"]) == result.n_shards
    for shard_meta in payload["shards"]:
        on_disk = mini_corpus_config.output_dir / shard_meta["file"]
        assert on_disk.exists()
        assert hashlib.sha256(on_disk.read_bytes()).hexdigest() == shard_meta["sha256"]


def test_atomic_output_when_prior_dir_exists(
    mini_corpus_config: CorpusConfig,
    shard_paths: Callable[[Path], list[Path]],
) -> None:
    mini_corpus_config.output_dir.mkdir(parents=True)
    (mini_corpus_config.output_dir / "stale.parquet").write_bytes(b"junk")

    ingest(mini_corpus_config, verify_input_sha=False)

    assert not (mini_corpus_config.output_dir / "stale.parquet").exists()
    assert shard_paths(mini_corpus_config.output_dir)
