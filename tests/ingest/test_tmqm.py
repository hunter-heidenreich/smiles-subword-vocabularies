"""Behavioral tests for Stage 0 tmQM ingest."""

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

_TMQM_FIXTURE_ROWS: tuple[tuple[str, str], ...] = (
    ("ABCDEF", "[Cu+2].[Cl-].[Cl-]"),
    ("GHIJKL", "[Fe](Cl)(Cl)Cl"),
    ("MNOPQR", "[Pt](Cl)(Cl)([NH3])[NH3]"),
    ("STUVWX", "[Au]Cl"),
    ("YZABCD", "[Pd](Cl)(Cl)([PH3])[PH3]"),
)


def _write_tmqm_csv(path: Path, rows: tuple[tuple[str, str], ...]) -> None:
    """Write a tmQM-style semicolon-delimited CSV.

    The canonical uiocompcat/tmQM release ships ``tmQM_y.csv`` with
    ``;`` as the column separator. The fixture mirrors that layout
    so the ingest delim matches the upstream byte-for-byte.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        fh.write("CSD_code;SMILES;charge;spin_multiplicity\n")
        for code, smi in rows:
            fh.write(f"{code};{smi};0;1\n")


@pytest.fixture
def tmqm_csv(tmp_path: Path) -> Path:
    path = tmp_path / "raw" / "tmQM_smiles.csv"
    _write_tmqm_csv(path, _TMQM_FIXTURE_ROWS)
    return path


@pytest.fixture
def tmqm_corpus_config(tmp_path: Path, tmqm_csv: Path) -> CorpusConfig:
    return CorpusConfig(
        name="tmqm_test",
        source="tmqm-test",
        manifest_id="tmqm-test",
        raw_path=tmqm_csv,
        output_dir=tmp_path / "out",
        csv_read_mode="named",
        delim=";",
        has_header=True,
        file_compression="none",
        normalize_names=False,
        drop_null_smiles=True,
        id_column="CSD_code",
        smiles_column="SMILES",
        shard_target_bytes=2**20,
        rows_per_batch=64,
    )


class TestTmqmIngest:
    """Schema, row count, CSD-code provenance, metals retained."""

    def test_schema_matches_raw_v1(
        self,
        tmqm_corpus_config: CorpusConfig,
        shard_paths: Callable[[Path], list[Path]],
    ) -> None:
        ingest(tmqm_corpus_config, verify_input_sha=False)

        for shard in shard_paths(tmqm_corpus_config.output_dir):
            assert pq.ParquetFile(shard).schema_arrow.equals(RAW_V1_SCHEMA)

    def test_row_count_matches_input(self, tmqm_corpus_config: CorpusConfig) -> None:
        result = ingest(tmqm_corpus_config, verify_input_sha=False)

        assert result.n_rows == len(_TMQM_FIXTURE_ROWS)

    def test_source_id_preserves_csd_code(
        self,
        tmqm_corpus_config: CorpusConfig,
        shard_columns: Callable[[Path], dict[str, list]],
    ) -> None:
        ingest(tmqm_corpus_config, verify_input_sha=False)

        cols = shard_columns(tmqm_corpus_config.output_dir)
        assert cols["source_id"] == [code for code, _ in _TMQM_FIXTURE_ROWS]

    def test_metal_smiles_pass_through(
        self,
        tmqm_corpus_config: CorpusConfig,
        shard_columns: Callable[[Path], dict[str, list]],
    ) -> None:
        ingest(tmqm_corpus_config, verify_input_sha=False)

        cols = shard_columns(tmqm_corpus_config.output_dir)
        assert any("[Cu+2]" in smi for smi in cols["smiles"])
        assert any("[Pt]" in smi for smi in cols["smiles"])

    def test_extra_columns_dropped(
        self,
        tmqm_corpus_config: CorpusConfig,
        shard_columns: Callable[[Path], dict[str, list]],
    ) -> None:
        ingest(tmqm_corpus_config, verify_input_sha=False)

        cols = shard_columns(tmqm_corpus_config.output_dir)
        assert set(cols) == {"source_id", "smiles", "source", "ingest_ts"}
