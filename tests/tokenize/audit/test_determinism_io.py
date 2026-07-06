"""Tests for ``smiles_subword.tokenize.audit.determinism_io`` (deposition)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from smiles_subword.tokenize.audit import determinism_io
from smiles_subword.tokenize.audit.determinism import ArtifactDigest, DeterminismResult
from smiles_subword.tokenize.grid import GridCell

BPE_CELL = GridCell(
    algo="bpe", vocab_size=256, corpus="pubchem", boundary="nmb", tier="headline"
)
EXPECTED_CELL = GridCell(
    algo="unigram", vocab_size=1024, corpus="pubchem", boundary="nmb", tier="headline"
)


def _digest(algo: str) -> ArtifactDigest:
    return ArtifactDigest(
        algo=algo,
        tokenizer_json_sha="t",
        merges_txt_sha="m" if algo == "bpe" else None,
        piece_set_sha=None if algo == "bpe" else "p",
        vocab_order_sha=None,
        log_probs_sha=None,
    )


def _result(
    *,
    algo: str = "bpe",
    deterministic: bool = True,
    mismatch_kind: str | None = None,
    spread: int = 0,
) -> DeterminismResult:
    digest = _digest(algo)
    return DeterminismResult(
        arm=algo,
        deterministic=deterministic,
        mismatch_kind=mismatch_kind,
        rerun_spread=spread,
        canonical=digest,
        rerun=digest,
    )


@pytest.fixture(autouse=True)
def _redirect_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point every determinism_io path at ``tmp_path`` so tests never touch the repo."""
    monkeypatch.setattr(
        determinism_io, "DETERMINISM_CELL_DIR", tmp_path / "determinism"
    )
    monkeypatch.setattr(
        determinism_io, "DETERMINISM_TABLE_JSON", tmp_path / "determinism_table.json"
    )
    monkeypatch.setattr(
        determinism_io, "DETERMINISM_TABLE_MD", tmp_path / "determinism_table.md"
    )


class TestJsonPath:
    def test_path_is_cell_id_under_the_cell_dir(self, tmp_path: Path) -> None:
        path = determinism_io.determinism_json_path("pubchem__smirk_gpe_v256_nmb")

        assert path == tmp_path / "determinism" / "pubchem__smirk_gpe_v256_nmb.json"


class TestWriteReadRoundTrip:
    def test_write_then_read_returns_the_payload(self) -> None:
        determinism_io.write_determinism_json(
            BPE_CELL, _result(), training_corpus_sha="sha1", expected_failure=False
        )

        payload = determinism_io.read_determinism_json(BPE_CELL)

        assert payload is not None
        assert payload["deterministic"] is True
        assert payload["rerun_spread"] == 0

    def test_payload_carries_grid_coordinates_and_flags(self) -> None:
        determinism_io.write_determinism_json(
            EXPECTED_CELL,
            _result(algo="unigram", deterministic=False, spread=2),
            training_corpus_sha="sha1",
            expected_failure=True,
        )

        payload = determinism_io.read_determinism_json(EXPECTED_CELL)

        assert payload is not None
        assert payload["cell_id"] == EXPECTED_CELL.cell_id
        assert payload["algo"] == "unigram"
        assert payload["vocab_size"] == 1024
        assert payload["boundary"] == "nmb"
        assert payload["tier"] == "headline"
        assert payload["training_corpus_sha"] == "sha1"
        assert payload["expected_failure"] is True
        assert payload["rerun_spread"] == 2
        assert payload["schema_version"] == determinism_io.SCHEMA_VERSION

    def test_write_leaves_no_tmp_file(self) -> None:
        path = determinism_io.write_determinism_json(
            BPE_CELL, _result(), training_corpus_sha="sha1", expected_failure=False
        )

        assert list(path.parent.iterdir()) == [path]

    def test_read_returns_none_when_absent(self) -> None:
        assert determinism_io.read_determinism_json(BPE_CELL) is None


class TestIsDeterminismDone:
    def test_true_when_json_present_and_sha_matches(self) -> None:
        determinism_io.write_determinism_json(
            BPE_CELL, _result(), training_corpus_sha="sha1", expected_failure=False
        )

        assert determinism_io.is_determinism_done(BPE_CELL, training_corpus_sha="sha1")

    def test_false_when_sha_drifted(self) -> None:
        determinism_io.write_determinism_json(
            BPE_CELL, _result(), training_corpus_sha="stale", expected_failure=False
        )

        assert not determinism_io.is_determinism_done(
            BPE_CELL, training_corpus_sha="fresh"
        )

    def test_false_when_json_absent(self) -> None:
        assert not determinism_io.is_determinism_done(BPE_CELL, training_corpus_sha="x")

    def test_false_when_json_corrupt(self) -> None:
        path = determinism_io.determinism_json_path(BPE_CELL.cell_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json")

        assert not determinism_io.is_determinism_done(BPE_CELL, training_corpus_sha="x")


class TestBuildDeterminismTable:
    def test_marks_cells_without_a_json_as_pending(self) -> None:
        determinism_io.write_determinism_json(
            BPE_CELL, _result(), training_corpus_sha="sha1", expected_failure=False
        )

        table_json, _md = determinism_io.build_determinism_table()
        table = json.loads(table_json.read_text())

        assert table["n_cells"] == 46
        assert table["n_present"] == 1
        assert len(table["pending"]) == 45
        assert BPE_CELL.cell_id not in table["pending"]

    def test_an_unexpected_failure_is_both_flagged_and_unexpected(self) -> None:
        determinism_io.write_determinism_json(
            BPE_CELL,
            _result(deterministic=False, mismatch_kind="bpe_byte"),
            training_corpus_sha="sha1",
            expected_failure=False,
        )

        table_json, _md = determinism_io.build_determinism_table()
        table = json.loads(table_json.read_text())

        assert table["flagged"] == [BPE_CELL.cell_id]
        assert table["unexpected"] == [BPE_CELL.cell_id]

    def test_an_expected_failure_is_flagged_but_not_unexpected(self) -> None:
        determinism_io.write_determinism_json(
            EXPECTED_CELL,
            _result(
                algo="unigram",
                deterministic=False,
                mismatch_kind="unigram_piece_set",
                spread=2,
            ),
            training_corpus_sha="sha1",
            expected_failure=True,
        )

        table_json, _md = determinism_io.build_determinism_table()
        table = json.loads(table_json.read_text())

        assert table["flagged"] == [EXPECTED_CELL.cell_id]
        assert table["unexpected"] == []

    def test_writes_a_markdown_table(self) -> None:
        determinism_io.write_determinism_json(
            BPE_CELL,
            _result(deterministic=False, mismatch_kind="bpe_byte"),
            training_corpus_sha="sha1",
            expected_failure=False,
        )

        _json, table_md = determinism_io.build_determinism_table()
        text = table_md.read_text()

        assert BPE_CELL.cell_id in text
        assert "**NO**" in text
        assert "pending" in text

    def test_returns_both_table_paths(self) -> None:
        table_json, table_md = determinism_io.build_determinism_table()

        assert table_json.name == "determinism_table.json"
        assert table_md.name == "determinism_table.md"
        assert table_json.is_file()
        assert table_md.is_file()

    def test_omits_pending_section_when_every_cell_is_verified(self) -> None:
        from smiles_subword.tokenize.grid import load_grid_manifest

        for cell in load_grid_manifest():
            determinism_io.write_determinism_json(
                cell, _result(), training_corpus_sha="sha1", expected_failure=False
            )

        table_json, table_md = determinism_io.build_determinism_table()
        table = json.loads(table_json.read_text())

        assert table["pending"] == []
        assert table["n_present"] == 46
        assert "pending" not in table_md.read_text()
