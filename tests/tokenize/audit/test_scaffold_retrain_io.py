"""Tests for ``smiles_subword.tokenize.audit.scaffold_retrain_io`` (rollup audit)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from smiles_subword.tokenize.audit import scaffold_retrain_io
from smiles_subword.tokenize.audit.scaffold_retrain import ScaffoldRetrainResult


@pytest.fixture(autouse=True)
def _redirect_audit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    audit_root = tmp_path / "audits"
    monkeypatch.setattr(scaffold_retrain_io, "AUDIT_DIR", audit_root)
    monkeypatch.setattr(
        scaffold_retrain_io,
        "AUDIT_JSON",
        audit_root / "scaffold_byte_identity_audit.json",
    )
    monkeypatch.setattr(
        scaffold_retrain_io, "AUDIT_MD", audit_root / "scaffold_byte_identity_audit.md"
    )
    return audit_root


def _result(
    cell_id: str, *, status: str, sha: str | None = None
) -> ScaffoldRetrainResult:
    return ScaffoldRetrainResult(
        cell_id=cell_id,
        status=status,  # type: ignore[arg-type]
        reason=None,
        scaffold_log_sha=sha,
    )


class TestRecordResults:
    def test_writes_fresh_json_and_md(self) -> None:
        json_path, md_path = scaffold_retrain_io.record_results(
            [_result("a__x", status="ok", sha="a" * 64)]
        )

        assert json_path.is_file()
        assert md_path.is_file()
        payload = json.loads(json_path.read_text())
        assert payload["schema_version"] == scaffold_retrain_io.SCHEMA_VERSION
        assert len(payload["cells"]) == 1
        assert payload["cells"][0]["cell_id"] == "a__x"
        md = md_path.read_text()
        assert "ok (retrained + byte-identical): **1**" in md
        assert "a__x" in md

    def test_upserts_by_cell_id_on_rerun(self) -> None:
        scaffold_retrain_io.record_results([_result("a__x", status="ok", sha="a" * 64)])

        scaffold_retrain_io.record_results([_result("a__x", status="ok", sha="b" * 64)])

        payload = json.loads(scaffold_retrain_io.AUDIT_JSON.read_text())
        cells = payload["cells"]
        assert len(cells) == 1
        assert cells[0]["scaffold_log_sha"] == "b" * 64

    def test_aggregates_multiple_cells(self) -> None:
        scaffold_retrain_io.record_results(
            [
                _result("a__x", status="ok", sha="a" * 64),
                _result("b__y", status="already_done", sha="b" * 64),
                _result("c__z", status="skipped"),
            ]
        )

        payload = json.loads(scaffold_retrain_io.AUDIT_JSON.read_text())

        assert len(payload["cells"]) == 3
        md = scaffold_retrain_io.AUDIT_MD.read_text()
        assert "ok (retrained + byte-identical): **1**" in md
        assert "already_done (idempotent skip): **1**" in md
        assert "skipped (non-BPE or dry-run): **1**" in md

    def test_handles_empty_input(self) -> None:
        json_path, _md_path = scaffold_retrain_io.record_results([])

        payload = json.loads(json_path.read_text())

        assert payload["cells"] == []

    def test_tolerates_corrupt_existing_json(self) -> None:
        scaffold_retrain_io.AUDIT_JSON.parent.mkdir(parents=True, exist_ok=True)
        scaffold_retrain_io.AUDIT_JSON.write_text("not json {")

        json_path, _md = scaffold_retrain_io.record_results(
            [_result("a__x", status="ok", sha="a" * 64)]
        )

        payload = json.loads(json_path.read_text())
        assert payload["cells"][0]["cell_id"] == "a__x"
