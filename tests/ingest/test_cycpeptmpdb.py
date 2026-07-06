"""Behavioral tests for Stage 0 CycPeptMPDB ingest."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow.parquet as pq
import pytest

from smiles_subword.config import CorpusConfig
from smiles_subword.ingest._common import RAW_V1_SCHEMA
from smiles_subword.ingest.csv_corpus import ingest

if TYPE_CHECKING:
    from collections.abc import Callable

_FIXTURE_ROWS: tuple[tuple[str, str], ...] = (
    ("1", "C1CCNC(=O)CCNC1=O"),
    ("2", "CC(C)C[C@@H]1NC(=O)[C@H](C)NC1=O"),
    ("3", "O=C1NCC(=O)N[C@@H](Cc2ccccc2)C(=O)N1"),
)


def _write_csv(path: Path, rows: tuple[tuple[str, str], ...]) -> None:
    """Write a CycPeptMPDB-style comma-delimited CSV with an extra HELM column."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        fh.write("ID,SMILES,HELM\n")
        for pid, smi in rows:
            fh.write(f"{pid},{smi},PEPTIDE1{{X}}$$$$\n")


@pytest.fixture
def cycpept_csv(tmp_path: Path) -> Path:
    path = tmp_path / "raw" / "CycPeptMPDB_Peptide_All.csv"
    _write_csv(path, _FIXTURE_ROWS)
    return path


@pytest.fixture
def cycpept_corpus_config(tmp_path: Path, cycpept_csv: Path) -> CorpusConfig:
    return CorpusConfig(
        name="cycpeptmpdb_test",
        source="cycpeptmpdb-test",
        manifest_id="cycpeptmpdb-test",
        raw_path=cycpept_csv,
        output_dir=tmp_path / "out",
        csv_read_mode="named",
        delim=",",
        has_header=True,
        file_compression="none",
        drop_null_smiles=True,
        id_column="ID",
        smiles_column="SMILES",
        shard_target_bytes=2**20,
        rows_per_batch=64,
    )


class TestCycpeptmpdbIngest:
    """Schema, row count, ID provenance, whole-molecule SMILES retained."""

    def test_schema_matches_raw_v1(
        self,
        cycpept_corpus_config: CorpusConfig,
        shard_paths: Callable[[Path], list[Path]],
    ) -> None:
        ingest(cycpept_corpus_config, verify_input_sha=False)

        for shard in shard_paths(cycpept_corpus_config.output_dir):
            assert pq.ParquetFile(shard).schema_arrow.equals(RAW_V1_SCHEMA)

    def test_row_count_matches_input(self, cycpept_corpus_config: CorpusConfig) -> None:
        result = ingest(cycpept_corpus_config, verify_input_sha=False)

        assert result.n_rows == len(_FIXTURE_ROWS)

    def test_source_id_preserves_peptide_id(
        self,
        cycpept_corpus_config: CorpusConfig,
        shard_columns: Callable[[Path], dict[str, list]],
    ) -> None:
        ingest(cycpept_corpus_config, verify_input_sha=False)

        cols = shard_columns(cycpept_corpus_config.output_dir)
        assert cols["source_id"] == [pid for pid, _ in _FIXTURE_ROWS]

    def test_helm_and_extra_columns_dropped(
        self,
        cycpept_corpus_config: CorpusConfig,
        shard_columns: Callable[[Path], dict[str, list]],
    ) -> None:
        ingest(cycpept_corpus_config, verify_input_sha=False)

        cols = shard_columns(cycpept_corpus_config.output_dir)
        assert set(cols) == {"source_id", "smiles", "source", "ingest_ts"}
