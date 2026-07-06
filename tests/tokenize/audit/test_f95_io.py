"""Tests for ``smiles_subword.tokenize.audit.f95_io`` (F_{p,n} deposition)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from smiles_subword.tokenize.audit import f95_io
from smiles_subword.tokenize.audit.f95 import F95Result
from smiles_subword.tokenize.grid import GridCell

A_CELL = GridCell(
    algo="bpe", vocab_size=256, corpus="pubchem", boundary="nmb", tier="headline"
)


def _result(*, unsafe: bool) -> F95Result:
    headline = 0.70 if unsafe else 1.0
    return F95Result(
        arm="bpe",
        v_observed=256,
        n_non_atomic=90,
        n_corpus_tokens=1000,
        n_corpus_molecules=100,
        fp_thresholds=[],
        clearance_by_n={50: 0.9, 100: headline, 200: 0.5},
        headline_clearance=headline,
        embedding_tail_unsafe=unsafe,
    )


@pytest.fixture(autouse=True)
def _redirect_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point every f95_io path at ``tmp_path`` so tests never touch the repo."""
    monkeypatch.setattr(f95_io, "F95_CELL_DIR", tmp_path / "f95")
    monkeypatch.setattr(f95_io, "F95_TABLE_JSON", tmp_path / "f95_table.json")
    monkeypatch.setattr(f95_io, "F95_TABLE_MD", tmp_path / "f95_table.md")


class TestJsonPath:
    def test_path_is_cell_id_under_the_cell_dir(self, tmp_path: Path) -> None:
        path = f95_io.f95_json_path("pubchem__smirk_gpe_v256_nmb")

        assert path == tmp_path / "f95" / "pubchem__smirk_gpe_v256_nmb.json"


class TestWriteReadRoundTrip:
    def test_write_then_read_returns_the_payload(self) -> None:
        f95_io.write_f95_json(A_CELL, _result(unsafe=True), training_corpus_sha="sha1")

        payload = f95_io.read_f95_json(A_CELL)

        assert payload is not None
        assert payload["headline_clearance"] == 0.70
        assert payload["embedding_tail_unsafe"] is True

    def test_payload_carries_grid_coordinates(self) -> None:
        f95_io.write_f95_json(A_CELL, _result(unsafe=False), training_corpus_sha="sha1")

        payload = f95_io.read_f95_json(A_CELL)

        assert payload is not None
        assert payload["cell_id"] == "pubchem__smirk_gpe_v256_nmb"
        assert payload["algo"] == "bpe"
        assert payload["vocab_size"] == 256
        assert payload["corpus"] == "pubchem"
        assert payload["boundary"] == "nmb"
        assert payload["tier"] == "headline"
        assert payload["training_corpus_sha"] == "sha1"
        assert payload["schema_version"] == f95_io.SCHEMA_VERSION

    def test_write_leaves_no_tmp_file(self) -> None:
        path = f95_io.write_f95_json(
            A_CELL, _result(unsafe=False), training_corpus_sha="sha1"
        )

        siblings = list(path.parent.iterdir())
        assert siblings == [path]

    def test_read_returns_none_when_absent(self) -> None:
        assert f95_io.read_f95_json(A_CELL) is None


class TestIsF95Done:
    def test_true_when_json_present_and_sha_matches(self) -> None:
        f95_io.write_f95_json(A_CELL, _result(unsafe=True), training_corpus_sha="sha1")

        assert f95_io.is_f95_done(A_CELL, training_corpus_sha="sha1") is True

    def test_false_when_sha_drifted(self) -> None:
        f95_io.write_f95_json(A_CELL, _result(unsafe=True), training_corpus_sha="stale")

        assert f95_io.is_f95_done(A_CELL, training_corpus_sha="fresh") is False

    def test_false_when_json_absent(self) -> None:
        assert f95_io.is_f95_done(A_CELL, training_corpus_sha="sha1") is False

    def test_false_when_json_corrupt(self) -> None:
        path = f95_io.f95_json_path(A_CELL.cell_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json")

        assert f95_io.is_f95_done(A_CELL, training_corpus_sha="sha1") is False


class TestBuildF95Table:
    def test_marks_cells_without_a_json_as_pending(self) -> None:
        f95_io.write_f95_json(A_CELL, _result(unsafe=False), training_corpus_sha="sha1")

        table_json, _md = f95_io.build_f95_table()
        table = json.loads(table_json.read_text())

        assert table["n_cells"] == 46
        assert table["n_present"] == 1
        assert len(table["pending"]) == 45
        assert A_CELL.cell_id not in table["pending"]

    def test_flags_embedding_tail_unsafe_cells(self) -> None:
        f95_io.write_f95_json(A_CELL, _result(unsafe=True), training_corpus_sha="sha1")

        table_json, _md = f95_io.build_f95_table()
        table = json.loads(table_json.read_text())

        assert table["flagged"] == [A_CELL.cell_id]

    def test_writes_a_markdown_table(self) -> None:
        f95_io.write_f95_json(A_CELL, _result(unsafe=True), training_corpus_sha="sha1")

        _json, table_md = f95_io.build_f95_table()
        text = table_md.read_text()

        assert A_CELL.cell_id in text
        assert "UNSAFE" in text
        assert "pending" in text

    def test_returns_both_table_paths(self) -> None:
        table_json, table_md = f95_io.build_f95_table()

        assert table_json.name == "f95_table.json"
        assert table_md.name == "f95_table.md"
        assert table_json.is_file()
        assert table_md.is_file()

    def test_omits_pending_section_when_every_cell_is_confirmed(self) -> None:
        from smiles_subword.tokenize.grid import load_grid_manifest

        for cell in load_grid_manifest():
            f95_io.write_f95_json(
                cell, _result(unsafe=False), training_corpus_sha="sha1"
            )

        table_json, table_md = f95_io.build_f95_table()
        table = json.loads(table_json.read_text())

        assert table["pending"] == []
        assert table["n_present"] == 46
        assert "pending" not in table_md.read_text()
