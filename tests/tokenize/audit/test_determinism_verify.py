"""Tests for ``smiles_subword.tokenize.audit.determinism_verify`` (grid + extras).

The shared orchestration is covered in ``test_runtime`` (``run_verify``); here we
test the driver-specific seam — the expected-jitter predicate — and that
the wrappers bind their seams onto ``run_verify``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from smiles_subword.tokenize.audit import determinism_verify
from smiles_subword.tokenize.extras import extras_cell_to_config
from smiles_subword.tokenize.grid import GridCell, grid_cell_to_config

BPE_CELL = GridCell(
    algo="bpe", vocab_size=256, corpus="pubchem", boundary="nmb", tier="headline"
)
UNI_CELL = GridCell(
    algo="unigram", vocab_size=256, corpus="pubchem", boundary="nmb", tier="headline"
)
EXPECTED_CELL = GridCell(
    algo="unigram", vocab_size=1024, corpus="pubchem", boundary="nmb", tier="headline"
)


class TestIsExpectedUnigramJitter:
    """The expected-jitter set is exactly Unigram, V=1024, NMB."""

    @pytest.mark.parametrize(
        ("cell", "expected"),
        [
            pytest.param(EXPECTED_CELL, True, id="unigram-v1024-nmb"),
            pytest.param(
                GridCell(
                    algo="unigram",
                    vocab_size=1024,
                    corpus="pubchem",
                    boundary="mb",
                    tier="headline",
                ),
                False,
                id="unigram-v1024-mb",
            ),
            pytest.param(UNI_CELL, False, id="unigram-v256-nmb"),
            pytest.param(
                GridCell(
                    algo="bpe",
                    vocab_size=1024,
                    corpus="pubchem",
                    boundary="nmb",
                    tier="headline",
                ),
                False,
                id="bpe-v1024-nmb",
            ),
        ],
    )
    def test_predicate(self, cell: GridCell, expected: bool) -> None:
        assert determinism_verify.is_expected_unigram_jitter(cell) is expected


class TestVerifyCellWrapper:
    """``verify_cell`` binds the grid seams onto the shared ``run_verify``."""

    def test_binds_grid_seams_onto_run_verify(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, object] = {}

        def spy(cell: object, **kwargs: object) -> None:
            captured["cell"] = cell
            captured.update(kwargs)
            return

        monkeypatch.setattr(determinism_verify._runtime, "run_verify", spy)

        assert determinism_verify.verify_cell(BPE_CELL, force=True) is None
        assert captured["cell"] == BPE_CELL
        assert captured["to_config"] is grid_cell_to_config
        assert (
            captured["is_expected_jitter"]
            is determinism_verify.is_expected_unigram_jitter
        )
        assert captured["prefix"] == f"det-{BPE_CELL.cell_id}-"
        assert captured["force"] is True
        assert captured["dry_run"] is False


class TestVerifyExtrasCellWrapper:
    """``verify_extras_cell`` binds the extras seams onto the shared core."""

    def test_binds_extras_seams_onto_run_verify(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, object] = {}

        def spy(cell: object, **kwargs: object) -> None:
            captured["cell"] = cell
            captured.update(kwargs)
            return

        monkeypatch.setattr(determinism_verify._runtime, "run_verify", spy)
        cell = SimpleNamespace(cell_id="extras-x")

        assert determinism_verify.verify_extras_cell(cell) is None  # type: ignore[arg-type]
        assert captured["cell"] is cell
        assert captured["to_config"] is extras_cell_to_config
        assert (
            captured["is_expected_jitter"] is determinism_verify._extras_expected_jitter
        )
        assert captured["prefix"] == "det-extras-extras-x-"

    def test_extras_predicate_is_always_false(self) -> None:
        cell = SimpleNamespace(algo="unigram", vocab_size=1024, boundary="nmb")
        assert determinism_verify._extras_expected_jitter(cell) is False  # type: ignore[arg-type]
