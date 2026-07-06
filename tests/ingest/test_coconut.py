"""Behavioral tests for Stage 0 COCONUT ingest."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow.parquet as pq
import pytest

from smiles_subword.config import CorpusConfig
from smiles_subword.ingest._common import RAW_V1_SCHEMA
from smiles_subword.ingest.csv_corpus import ingest

if TYPE_CHECKING:
    from collections.abc import Callable

_COCONUT_FIXTURE_ROWS: tuple[tuple[str, str], ...] = (
    ("CNP0000001", "OC[C@H]1OC(O)C(O)C(O)C1O"),
    (
        "CNP0000002",
        "CC(=O)O[C@H]1CC[C@@]2(C)C(=CCC3[C@@H]2CC[C@]2(C)[C@H](CC[C@@H]32)C(C)C)C1",
    ),
    ("CNP0000003", "COc1cc2ccc(=O)oc2cc1OC"),
    ("CNP0000004", "C[C@H]1CC[C@@H]2[C@H](C1)C(C)=CCC2"),
    ("CNP0000005", "OC1=CC(=O)C2=C(O1)C=CC=C2"),
)


def _write_coconut_csv(path: Path, rows: tuple[tuple[str, str], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        fh.write("identifier,canonical_smiles,molecular_formula,np_likeness\n")
        for cid, smi in rows:
            fh.write(f"{cid},{smi},C0H0,5.0\n")


def _coconut_config(csv: Path, out: Path, **overrides: object) -> CorpusConfig:
    """A COCONUT-shaped named-mode config over `csv`; override any field."""
    fields: dict[str, Any] = {
        "name": "coconut_test",
        "source": "coconut-test",
        "manifest_id": "coconut-test",
        "raw_path": csv,
        "output_dir": out,
        "csv_read_mode": "named",
        "delim": ",",
        "has_header": True,
        "file_compression": "none",
        "normalize_names": True,
        "drop_null_smiles": True,
        "id_column": "identifier",
        "smiles_column": "canonical_smiles",
        "shard_target_bytes": 2**20,
        "rows_per_batch": 64,
    }
    fields.update(overrides)
    return CorpusConfig(**fields)


@pytest.fixture
def coconut_csv(tmp_path: Path) -> Path:
    path = tmp_path / "raw" / "coconut.csv"
    _write_coconut_csv(path, _COCONUT_FIXTURE_ROWS)
    return path


@pytest.fixture
def coconut_corpus_config(tmp_path: Path, coconut_csv: Path) -> CorpusConfig:
    return _coconut_config(coconut_csv, tmp_path / "out")


class TestCoconutIngest:
    """Schema, row count, NP identifier provenance."""

    def test_schema_matches_raw_v1(
        self,
        coconut_corpus_config: CorpusConfig,
        shard_paths: Callable[[Path], list[Path]],
    ) -> None:
        ingest(coconut_corpus_config, verify_input_sha=False)

        for shard in shard_paths(coconut_corpus_config.output_dir):
            assert pq.ParquetFile(shard).schema_arrow.equals(RAW_V1_SCHEMA)

    def test_row_count_matches_input(self, coconut_corpus_config: CorpusConfig) -> None:
        result = ingest(coconut_corpus_config, verify_input_sha=False)

        assert result.n_rows == len(_COCONUT_FIXTURE_ROWS)

    def test_source_id_preserves_coconut_identifier(
        self,
        coconut_corpus_config: CorpusConfig,
        shard_columns: Callable[[Path], dict[str, list]],
    ) -> None:
        ingest(coconut_corpus_config, verify_input_sha=False)

        cols = shard_columns(coconut_corpus_config.output_dir)
        assert cols["source_id"] == [cid for cid, _ in _COCONUT_FIXTURE_ROWS]

    def test_extra_columns_dropped(
        self,
        coconut_corpus_config: CorpusConfig,
        shard_columns: Callable[[Path], dict[str, list]],
    ) -> None:
        ingest(coconut_corpus_config, verify_input_sha=False)

        cols = shard_columns(coconut_corpus_config.output_dir)
        assert set(cols) == {"source_id", "smiles", "source", "ingest_ts"}

    def test_null_smiles_rows_are_dropped(
        self, tmp_path: Path, shard_columns: Callable[[Path], dict[str, list]]
    ) -> None:
        # drop_null_smiles is set on every named corpus but never load-bearing
        # without a null/empty SMILES row to drop. The middle row must vanish.
        csv = tmp_path / "raw" / "coconut_nulls.csv"
        csv.parent.mkdir(parents=True, exist_ok=True)
        csv.write_text(
            "identifier,canonical_smiles,molecular_formula,np_likeness\n"
            "CNP0000001,CCO,C2H6O,5.0\n"
            "CNP0000002,,C0H0,5.0\n"  # empty SMILES -> dropped
            "CNP0000003,c1ccccc1,C6H6,5.0\n"
        )
        cfg = _coconut_config(csv, tmp_path / "out", drop_null_smiles=True)

        result = ingest(cfg, verify_input_sha=False)

        cols = shard_columns(cfg.output_dir)
        assert result.n_rows == 2
        assert cols["source_id"] == ["CNP0000001", "CNP0000003"]
        assert "" not in cols["smiles"]

    def test_normalize_names_resolves_denormalized_header(
        self, tmp_path: Path, shard_columns: Callable[[Path], dict[str, list]]
    ) -> None:
        # The header column "Canonical SMILES" (space + caps) only resolves to
        # the configured `canonical_smiles` after normalize_names rewrites it,
        # so a green test here proves the flag is load-bearing.
        csv = tmp_path / "raw" / "coconut_denorm.csv"
        csv.parent.mkdir(parents=True, exist_ok=True)
        csv.write_text(
            "Identifier,Canonical SMILES,Molecular Formula\n"
            "CNP0000001,CCO,C2H6O\n"
            "CNP0000002,c1ccccc1,C6H6\n"
        )
        cfg = _coconut_config(csv, tmp_path / "out", normalize_names=True)

        ingest(cfg, verify_input_sha=False)

        cols = shard_columns(cfg.output_dir)
        assert cols["source_id"] == ["CNP0000001", "CNP0000002"]
        assert cols["smiles"] == ["CCO", "c1ccccc1"]
