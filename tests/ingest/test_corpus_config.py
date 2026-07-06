"""Tests for the CorpusConfig schema."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yaml
from pydantic import ValidationError

from smiles_subword.config import CorpusConfig
from smiles_subword.paths import CONFIGS_DIR, REPO_ROOT

if TYPE_CHECKING:
    from pathlib import Path


def _base_payload(**overrides: object) -> dict:
    payload = {
        "name": "pubchem",
        "source": "pubchem",
        "manifest_id": "pubchem-cid-smiles",
        "raw_path": "data/raw/pubchem/CID-SMILES.gz",
        "output_dir": "data/processed/pubchem/raw_v1",
    }
    payload.update(overrides)
    return payload


def test_pubchem_yaml_round_trips() -> None:
    cfg = CorpusConfig.from_yaml(CONFIGS_DIR / "corpus" / "pubchem.yaml")

    assert cfg.name == "pubchem"
    assert cfg.source == "pubchem"
    assert cfg.manifest_id == "pubchem-cid-smiles"
    assert cfg.parquet_compression == "zstd"
    assert cfg.shard_target_bytes == 268435456


# The whole config-driven architecture lives in these YAMLs — a new corpus is a
# YAML, not code — yet only the hand-built configs in test_{coconut,tmqm,...}.py
# pin the read deltas. Editing tmqm's delim to ',' would otherwise pass the
# entire suite. Pin each shipped YAML's read-path deltas against the file.
_NAMED_CORPUS_DELTAS: dict[str, dict[str, object]] = {
    "coconut": {
        "csv_read_mode": "named",
        "delim": ",",
        "id_column": "identifier",
        "smiles_column": "canonical_smiles",
        "normalize_names": True,
        "drop_null_smiles": True,
    },
    "cycpeptmpdb": {
        "csv_read_mode": "named",
        "delim": ",",
        "id_column": "ID",
        "smiles_column": "SMILES",
        "normalize_names": False,
        "drop_null_smiles": True,
    },
    "tmqm": {
        "csv_read_mode": "named",
        "delim": ";",
        "id_column": "CSD_code",
        "smiles_column": "SMILES",
        "normalize_names": False,
        "drop_null_smiles": True,
    },
}


@pytest.mark.parametrize("corpus", list(_NAMED_CORPUS_DELTAS))
def test_named_corpus_yaml_pins_read_deltas(corpus: str) -> None:
    cfg = CorpusConfig.from_yaml(CONFIGS_DIR / "corpus" / f"{corpus}.yaml")

    assert cfg.name == corpus
    for field, expected in _NAMED_CORPUS_DELTAS[corpus].items():
        assert getattr(cfg, field) == expected, field


def test_relative_paths_resolve_against_repo_root() -> None:
    cfg = CorpusConfig.from_yaml(CONFIGS_DIR / "corpus" / "pubchem.yaml")

    assert cfg.raw_path == REPO_ROOT / "data" / "raw" / "pubchem" / "CID-SMILES.gz"
    assert cfg.output_dir == REPO_ROOT / "data" / "processed" / "pubchem" / "raw_v1"


def test_absolute_paths_pass_through(tmp_path: Path) -> None:
    cfg = CorpusConfig.model_validate(
        _base_payload(
            raw_path=str(tmp_path / "in.gz"), output_dir=str(tmp_path / "out")
        )
    )

    assert cfg.raw_path == tmp_path / "in.gz"
    assert cfg.output_dir == tmp_path / "out"


def test_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        CorpusConfig.model_validate(_base_payload(rogue_field=True))


def test_rejects_shard_target_below_minimum() -> None:
    with pytest.raises(ValidationError):
        CorpusConfig.model_validate(_base_payload(shard_target_bytes=512))


def test_rejects_invalid_compression() -> None:
    with pytest.raises(ValidationError):
        CorpusConfig.model_validate(_base_payload(parquet_compression="brotli"))


def test_rejects_compression_level_out_of_range() -> None:
    with pytest.raises(ValidationError):
        CorpusConfig.model_validate(_base_payload(parquet_compression_level=23))


def test_rejects_rows_per_batch_below_minimum() -> None:
    with pytest.raises(ValidationError):
        CorpusConfig.model_validate(_base_payload(rows_per_batch=10))


def test_accepts_lowered_floors_for_testing_workflows() -> None:
    cfg = CorpusConfig.model_validate(
        _base_payload(shard_target_bytes=4096, rows_per_batch=128)
    )

    assert cfg.shard_target_bytes == 4096
    assert cfg.rows_per_batch == 128


def test_rejects_extra_fields_via_yaml(tmp_path: Path) -> None:
    payload = _base_payload(rogue_field=True)
    path = tmp_path / "rogue.yaml"
    path.write_text(yaml.safe_dump(payload))

    with pytest.raises(ValidationError):
        CorpusConfig.from_yaml(path)


# In named read mode the reader hardcodes a header and cannot decompress, so a
# non-"none" file_compression or has_header=False would be silently ignored. The
# validator turns both into loud config-time errors; positional mode is exempt.
def test_named_mode_rejects_compression() -> None:
    with pytest.raises(ValidationError, match="file_compression"):
        CorpusConfig.model_validate(
            _base_payload(
                csv_read_mode="named", has_header=True, file_compression="gzip"
            )
        )


def test_named_mode_requires_header() -> None:
    with pytest.raises(ValidationError, match="has_header"):
        CorpusConfig.model_validate(
            _base_payload(
                csv_read_mode="named", file_compression="none", has_header=False
            )
        )


def test_named_mode_accepts_plain_header_csv() -> None:
    cfg = CorpusConfig.model_validate(
        _base_payload(csv_read_mode="named", has_header=True, file_compression="none")
    )
    assert cfg.csv_read_mode == "named"


def test_positional_mode_allows_compression_without_header() -> None:
    # The named-mode guard must not constrain positional configs — PubChem is
    # positional, gzipped, and headerless.
    cfg = CorpusConfig.model_validate(
        _base_payload(file_compression="gzip", has_header=False)
    )
    assert cfg.file_compression == "gzip"
