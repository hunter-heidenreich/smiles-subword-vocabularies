"""Tests for ``smiles_subword.preprocess.conformance``."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from smiles_subword.preprocess.conformance import (
    BaseConformanceOracle,
    ConformanceResult,
    filter_conformant,
    offending_atoms,
)

_CONFORMANT = ("CCO", "c1ccccc1", "[Si](C)(C)C")
_NONCONFORMANT = ("C[si-]1cccc1", "c1nc2ccccc2[te]1")  # aromatic Si, aromatic Te


class TestOracle:
    def test_base_has_158_glyphs(self) -> None:
        assert len(BaseConformanceOracle().base_glyphs()) == 158

    def test_conformant_strings_pass(self) -> None:
        mask = BaseConformanceOracle().nonconformant_mask(list(_CONFORMANT))
        assert mask == [False, False, False]

    def test_nonconformant_aromatics_flagged(self) -> None:
        mask = BaseConformanceOracle().nonconformant_mask(list(_NONCONFORMANT))
        assert mask == [True, True]


class TestOffendingAtoms:
    def test_aromatic_silicon(self) -> None:
        assert offending_atoms("C[si-]1cccc1", {"c", "s"}) == ["aromatic-Si"]

    def test_aromatic_tellurium(self) -> None:
        glyphs = BaseConformanceOracle().base_glyphs()
        assert offending_atoms("c1nc2ccccc2[te]1", glyphs) == ["aromatic-Te"]

    def test_unparseable_returns_marker(self) -> None:
        assert offending_atoms("not a smiles [[[", set()) == ["rdkit-unparseable"]

    def test_no_aromatic_offender_returns_other(self) -> None:
        # Parses fine but carries no aromatic-non-base atom -> the catch-all.
        assert offending_atoms("CCO", set()) == ["other"]


def _write_corpus(path: Path, rows: list[tuple[str, str]]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "source_id": [r[0] for r in rows],
            "smiles": [r[1] for r in rows],
        }
    )
    pq.write_table(table, path / "canon_dedup_v1-00000.parquet")


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    rows = [
        ("a", "CCO"),
        ("b", "C[si-]1cccc1"),
        ("c", "c1ccccc1"),
        ("d", "c1nc2ccccc2[te]1"),
        ("e", "[Si](C)(C)C"),
    ]
    src = tmp_path / "in"
    _write_corpus(src, rows)
    return src


class TestFilterConformant:
    def test_drops_only_nonconformant(self, corpus: Path, tmp_path: Path) -> None:
        out = tmp_path / "out"
        result = filter_conformant(corpus, out, tmp_path / "dropped.jsonl")

        assert result.n_input_rows == 5
        assert result.n_dropped == 2
        assert result.n_kept == 3

    def test_output_preserves_schema_and_conformant_rows(
        self, corpus: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        filter_conformant(corpus, out, tmp_path / "dropped.jsonl")

        table = pa.concat_tables(
            pq.read_table(p) for p in out.glob("conformant_v1-*.parquet")
        )
        assert table.column_names == ["source_id", "smiles"]
        assert set(table.column("source_id").to_pylist()) == {"a", "c", "e"}

    def test_deposit_records_offenders(self, corpus: Path, tmp_path: Path) -> None:
        out = tmp_path / "out"
        deposit = tmp_path / "dropped.jsonl"
        filter_conformant(corpus, out, deposit)

        rows = [json.loads(line) for line in deposit.read_text().splitlines()]
        offenders = {r["smiles"]: r["offenders"] for r in rows}
        assert offenders["C[si-]1cccc1"] == ["aromatic-Si"]
        assert offenders["c1nc2ccccc2[te]1"] == ["aromatic-Te"]

    def test_manifest_written(self, corpus: Path, tmp_path: Path) -> None:
        out = tmp_path / "out"
        filter_conformant(corpus, out, tmp_path / "dropped.jsonl")

        assert (out / "MANIFEST.yaml").is_file()

    def test_idempotent_on_already_conformant(self, tmp_path: Path) -> None:
        src = tmp_path / "clean"
        _write_corpus(src, [("a", "CCO"), ("b", "c1ccccc1")])
        result = filter_conformant(src, tmp_path / "out", tmp_path / "d.jsonl")

        assert result.n_dropped == 0
        assert result.n_kept == 2

    def test_raises_when_no_parquet_shards(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match="no Parquet shards"):
            filter_conformant(empty, tmp_path / "out", tmp_path / "d.jsonl")

    def test_characterize_false_skips_offender_detail(
        self, corpus: Path, tmp_path: Path
    ) -> None:
        # The same rows still drop and still get one deposit line each, but the
        # RDKit characterization is skipped so each offenders list is empty.
        deposit = tmp_path / "dropped.jsonl"
        result = filter_conformant(
            corpus, tmp_path / "out", deposit, characterize=False
        )

        rows = [json.loads(line) for line in deposit.read_text().splitlines()]
        assert result.n_dropped == 2
        assert [r["offenders"] for r in rows] == [[], []]


def test_drop_rate_is_zero_for_empty_corpus() -> None:
    result = ConformanceResult(
        n_input_rows=0,
        n_kept=0,
        n_dropped=0,
        output_dir=Path("out"),
        deposit_path=Path("d"),
    )
    assert result.drop_rate == 0.0
