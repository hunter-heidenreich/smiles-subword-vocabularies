"""Behavioral tests for Stage 0 ZINC-22 multi-tranche ingest."""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

import pyarrow.parquet as pq
import pytest
import yaml

from smiles_subword.ingest._common import RAW_V1_SCHEMA, sha256_file
from smiles_subword.ingest.zinc22_multi_tranche import (
    TrancheSpec,
    _is_complete,
    ingest_multi_tranche,
    read_tranche_list,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from smiles_subword.config import Zinc22MultiTrancheConfig


def test_parses_tranches_tsv_when_well_formed(
    zinc22_multi_tranche_tranches_tsv: Path,
) -> None:
    specs = read_tranche_list(zinc22_multi_tranche_tranches_tsv)

    assert len(specs) == 3
    assert specs[0] == TrancheSpec(
        tranche_id="zinc-22x-H17P200",
        generation="x",
        heavy_atom_bin=17,
        logp_bin=200,
        url="https://example.invalid/zinc-22x-H17P200.smi.gz",
        expected_bytes=specs[0].expected_bytes,
    )


def test_rejects_tranche_list_when_header_does_not_match(tmp_path: Path) -> None:
    bad = tmp_path / "bad.tsv"
    bad.write_text("tranche_id\twrong_columns\n")

    with pytest.raises(ValueError, match="header"):
        read_tranche_list(bad)


def test_writes_per_tranche_dirs_when_ingesting(
    zinc22_multi_tranche_config: Zinc22MultiTrancheConfig,
    zinc22_manifest: Path,
    shard_paths: Callable[[Path], list[Path]],
) -> None:
    result = ingest_multi_tranche(
        zinc22_multi_tranche_config, fetch=False, manifest_path=zinc22_manifest
    )

    assert result.n_tranches == 3
    assert result.n_rows == 6
    for sub in ("zinc-22x-H17P200", "zinc-22x-H18P210", "zinc-22f-H19P220"):
        per_tranche = zinc22_multi_tranche_config.output_root / sub
        assert (per_tranche / "MANIFEST.yaml").exists()
        assert shard_paths(per_tranche)


def test_aggregate_manifest_pins_tranches_sha_when_ingesting(
    zinc22_multi_tranche_config: Zinc22MultiTrancheConfig,
    zinc22_manifest: Path,
) -> None:
    ingest_multi_tranche(
        zinc22_multi_tranche_config, fetch=False, manifest_path=zinc22_manifest
    )

    payload = yaml.safe_load(
        (zinc22_multi_tranche_config.output_root / "MANIFEST.yaml").read_text()
    )
    assert payload["schema"] == "raw_v1"
    assert payload["layout"] == "multi_tranche"
    assert payload["source"] == "zinc22"
    assert payload["n_tranches"] == 3
    assert payload["n_rows"] == 6
    assert payload["tranches_sha256"] == sha256_file(
        zinc22_multi_tranche_config.tranches_path
    )
    listed_ids = {t["tranche_id"] for t in payload["tranches"]}
    assert listed_ids == {
        "zinc-22x-H17P200",
        "zinc-22x-H18P210",
        "zinc-22f-H19P220",
    }


def test_records_per_tranche_global_manifest_entries_when_ingesting(
    zinc22_multi_tranche_config: Zinc22MultiTrancheConfig,
    zinc22_manifest: Path,
) -> None:
    ingest_multi_tranche(
        zinc22_multi_tranche_config, fetch=False, manifest_path=zinc22_manifest
    )

    entries = yaml.safe_load(zinc22_manifest.read_text())["artifacts"]
    ids = {e["id"] for e in entries}
    assert ids == {
        "zinc22-zinc-22x-H17P200",
        "zinc22-zinc-22x-H18P210",
        "zinc22-zinc-22f-H19P220",
    }


def test_concurrent_workers_do_not_clobber_global_manifest_when_ingesting(
    zinc22_multi_tranche_config: Zinc22MultiTrancheConfig,
    zinc22_manifest: Path,
) -> None:
    """Regression: each tranche's entry must survive concurrent recording.

    The original multi-tranche implementation called `record_manifest_entry`
    inside each worker thread, so two threads could both read an empty
    artifacts list, append their own entry, and race on the write — the
    slower writer clobbered the faster one and one tranche's entry was
    silently lost. Re-running the ingest with all three tranches under
    real concurrency (workers > 1) must always yield three distinct
    artifact entries on disk.
    """
    expected = {
        "zinc22-zinc-22x-H17P200",
        "zinc22-zinc-22x-H18P210",
        "zinc22-zinc-22f-H19P220",
    }
    for _ in range(8):
        zinc22_manifest.write_text(yaml.safe_dump({"version": 1, "artifacts": []}))
        if zinc22_multi_tranche_config.output_root.exists():
            shutil.rmtree(zinc22_multi_tranche_config.output_root)

        ingest_multi_tranche(
            zinc22_multi_tranche_config, fetch=False, manifest_path=zinc22_manifest
        )

        entries = yaml.safe_load(zinc22_manifest.read_text())["artifacts"]
        assert {e["id"] for e in entries} == expected


def test_skips_completed_tranches_when_rerun(
    zinc22_multi_tranche_config: Zinc22MultiTrancheConfig,
    zinc22_manifest: Path,
) -> None:
    first = ingest_multi_tranche(
        zinc22_multi_tranche_config, fetch=False, manifest_path=zinc22_manifest
    )
    first_mtimes = {
        t.tranche_id: (t.output_dir / "MANIFEST.yaml").stat().st_mtime_ns
        for t in first.tranches
    }

    second = ingest_multi_tranche(
        zinc22_multi_tranche_config, fetch=False, manifest_path=zinc22_manifest
    )

    assert all(t.skipped for t in second.tranches)
    for t in second.tranches:
        assert (t.output_dir / "MANIFEST.yaml").stat().st_mtime_ns == first_mtimes[
            t.tranche_id
        ]


def test_isolates_failure_to_one_tranche_when_input_missing(
    zinc22_multi_tranche_config: Zinc22MultiTrancheConfig,
    zinc22_manifest: Path,
) -> None:
    casualty = zinc22_multi_tranche_config.raw_root / "zinc-22x-H18P210.smi.gz"
    casualty.unlink()

    result = ingest_multi_tranche(
        zinc22_multi_tranche_config, fetch=False, manifest_path=zinc22_manifest
    )

    failed_ids = {f.tranche_id for f in result.failures}
    succeeded_ids = {t.tranche_id for t in result.tranches}
    assert "zinc-22x-H18P210" in failed_ids
    assert succeeded_ids == {"zinc-22x-H17P200", "zinc-22f-H19P220"}


def test_limit_caps_processed_tranches_when_set(
    zinc22_multi_tranche_config: Zinc22MultiTrancheConfig,
    zinc22_manifest: Path,
) -> None:
    result = ingest_multi_tranche(
        zinc22_multi_tranche_config,
        fetch=False,
        manifest_path=zinc22_manifest,
        limit=2,
    )

    assert result.n_tranches == 2


def test_per_tranche_shards_match_raw_v1_schema_when_ingesting(
    zinc22_multi_tranche_config: Zinc22MultiTrancheConfig,
    zinc22_manifest: Path,
    shard_paths: Callable[[Path], list[Path]],
) -> None:
    ingest_multi_tranche(
        zinc22_multi_tranche_config, fetch=False, manifest_path=zinc22_manifest
    )

    for sub in ("zinc-22x-H17P200", "zinc-22x-H18P210", "zinc-22f-H19P220"):
        for shard in shard_paths(zinc22_multi_tranche_config.output_root / sub):
            assert pq.ParquetFile(shard).schema_arrow.equals(RAW_V1_SCHEMA)


# --- read_tranche_list malformed-input guards -----------------------------

_TRANCHE_HEADER = (
    "tranche_id\tgeneration\theavy_atom_bin\tlogp_bin\turl\texpected_bytes\n"
)


def test_rejects_tranche_list_when_row_has_wrong_column_count(tmp_path: Path) -> None:
    bad = tmp_path / "bad.tsv"
    bad.write_text(_TRANCHE_HEADER + "zinc-22x-H10P000\tx\t10\n")  # 3 of 6 columns

    with pytest.raises(ValueError, match="malformed row 2"):
        read_tranche_list(bad)


def test_rejects_tranche_list_when_row_field_not_integer(tmp_path: Path) -> None:
    bad = tmp_path / "bad.tsv"
    row = "zinc-22x-H10P000\tx\tNOT_AN_INT\t100\thttps://x.invalid\t42\n"
    bad.write_text(_TRANCHE_HEADER + row)

    with pytest.raises(ValueError, match="malformed row 2"):
        read_tranche_list(bad)


def test_parses_tranche_list_skipping_blank_rows(tmp_path: Path) -> None:
    tsv = tmp_path / "with_blank.tsv"
    row = "zinc-22x-H10P000\tx\t10\t100\thttps://x.invalid\t42\n"
    tsv.write_text(_TRANCHE_HEADER + "\n" + row)  # blank line before the data row

    specs = read_tranche_list(tsv)

    assert len(specs) == 1
    assert specs[0].tranche_id == "zinc-22x-H10P000"


# --- _is_complete resume guards (partial/corrupt manifest -> re-ingest) ----


def _write_aggregate(output_dir: Path, payload: object) -> None:
    (output_dir / "MANIFEST.yaml").write_text(yaml.safe_dump(payload))


def test_is_complete_false_when_yaml_corrupt(tmp_path: Path) -> None:
    (tmp_path / "MANIFEST.yaml").write_text("schema: raw_v1\nshards: [1, 2")

    assert _is_complete(tmp_path) is False


def test_is_complete_false_when_schema_not_raw_v1(tmp_path: Path) -> None:
    _write_aggregate(
        tmp_path, {"schema": "clean_v1", "n_rows": 1, "n_shards": 1, "shards": []}
    )

    assert _is_complete(tmp_path) is False


def test_is_complete_false_when_required_fields_missing(tmp_path: Path) -> None:
    # schema is right but n_rows/n_shards absent — a half-written manifest from
    # an interrupted run must read as incomplete and be re-ingested.
    _write_aggregate(tmp_path, {"schema": "raw_v1", "shards": []})

    assert _is_complete(tmp_path) is False
