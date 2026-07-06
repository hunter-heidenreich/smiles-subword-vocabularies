"""Tests for ``measure._cellmeta`` (resolve a cell's meta + freshness).

``cell_meta`` / ``cell_training_sha_fresh`` / ``arm_info`` are exercised through
every ``*_io`` module's deposit + freshness tests; this pins
``resolve_cell_meta`` (whose error branches the held-out deposit path isn't
unit-tested through) directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from smiles_subword.tokenize.measure import _cellmeta
from smiles_subword.tokenize.measure._cellmeta import (
    CellMetaFields,
    resolve_cell_meta,
)


class TestResolveCellMeta:
    """resolve_cell_meta returns CellMetaFields, or an error reason per branch."""

    @pytest.fixture
    def artifacts(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
        root = tmp_path / "artifacts"
        monkeypatch.setattr(
            _cellmeta, "tokenizer_artifact_dir", lambda c, n: root / c / n
        )
        return root

    @staticmethod
    def _write_meta(artifacts: Path, cell_id: str, payload: dict[str, object]) -> None:
        corpus, _, name = cell_id.partition("__")
        cell = artifacts / corpus / name
        cell.mkdir(parents=True, exist_ok=True)
        (cell / "meta.yaml").write_text(yaml.safe_dump(payload))

    def test_malformed_cell_id_pends(self, artifacts: Path) -> None:
        result = resolve_cell_meta("nodoubleunderscore")
        assert isinstance(result, str)
        assert "malformed cell_id" in result

    def test_absent_meta_pends(self, artifacts: Path) -> None:
        result = resolve_cell_meta("pubchem__smirk_gpe_v256_nmb")
        assert isinstance(result, str)
        assert "no meta.yaml" in result

    def test_missing_merge_brackets_pends(self, artifacts: Path) -> None:
        self._write_meta(
            artifacts, "pubchem__smirk_gpe_v256_nmb", {"training_corpus_sha": "T"}
        )
        result = resolve_cell_meta("pubchem__smirk_gpe_v256_nmb")
        assert isinstance(result, str)
        assert "missing merge_brackets" in result

    def test_missing_training_corpus_sha_pends(self, artifacts: Path) -> None:
        self._write_meta(
            artifacts, "pubchem__smirk_gpe_v256_nmb", {"merge_brackets": False}
        )
        result = resolve_cell_meta("pubchem__smirk_gpe_v256_nmb")
        assert isinstance(result, str)
        assert "missing training_corpus_sha" in result

    def test_resolves_all_fields(self, artifacts: Path) -> None:
        self._write_meta(
            artifacts,
            "pubchem__smirk_gpe_v256_mb",
            {"merge_brackets": True, "training_corpus_sha": "T"},
        )
        result = resolve_cell_meta("pubchem__smirk_gpe_v256_mb")
        assert isinstance(result, CellMetaFields)
        assert result.corpus == "pubchem"
        assert result.name == "smirk_gpe_v256_mb"
        assert result.boundary == "mb"
        assert result.training_corpus_sha == "T"
        assert result.artifact_dir == artifacts / "pubchem" / "smirk_gpe_v256_mb"
