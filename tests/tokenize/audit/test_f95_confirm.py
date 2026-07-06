"""Tests for ``smiles_subword.tokenize.audit.f95_confirm`` (grid + extras driver).

The shared orchestration is covered in ``test_runtime`` (``run_confirm``); here we
test the driver-specific seams — the matched-BPE-cell lookup and the grid
glyph-alphabet resolver — and that the wrappers bind them onto ``run_confirm``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

from smiles_subword.tokenize.audit import f95_confirm
from smiles_subword.tokenize.extras import extras_cell_to_config
from smiles_subword.tokenize.grid import GridCell, grid_cell_to_config

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Never

    import pytest

BPE_CELL = GridCell(
    algo="bpe", vocab_size=256, corpus="pubchem", boundary="nmb", tier="headline"
)
UNI_CELL = GridCell(
    algo="unigram", vocab_size=256, corpus="pubchem", boundary="nmb", tier="headline"
)


class TestMatchedBpeCells:
    def test_only_same_corpus_same_boundary_bpe_cells(self) -> None:
        cells = f95_confirm._matched_bpe_cells(UNI_CELL)

        assert cells
        assert all(
            c.algo == "bpe" and c.corpus == "pubchem" and c.boundary == "nmb"
            for c in cells
        )

    def test_sorted_by_vocab_size(self) -> None:
        cells = f95_confirm._matched_bpe_cells(UNI_CELL)

        sizes = [c.vocab_size for c in cells]
        assert sizes == sorted(sizes)


class TestResolveAtomicTokens:
    def test_bpe_cell_uses_its_own_glyph_alphabet(self) -> None:
        from smiles_subword.tokenize.adapters.smirk import SmirkAdapter

        tok = SmirkAdapter.atomic()

        atomic = f95_confirm._grid_resolve_atomic_tokens(BPE_CELL, tok)

        assert atomic == frozenset(tok.hf_tokenizer.get_vocab())

    def test_unigram_cell_uses_a_matched_bpe_artifact(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_bpe = SimpleNamespace(
            hf_tokenizer=SimpleNamespace(get_vocab=lambda: {"C": 0, "O": 1, "N": 2})
        )
        monkeypatch.setattr(f95_confirm.SmirkAdapter, "load", lambda _d: fake_bpe)

        atomic = f95_confirm._grid_resolve_atomic_tokens(UNI_CELL, object())

        assert atomic == frozenset({"C", "O", "N"})

    def test_unigram_returns_none_when_no_matched_bpe_is_trained(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(_d: Path) -> Never:
            raise FileNotFoundError

        monkeypatch.setattr(f95_confirm.SmirkAdapter, "load", boom)

        assert f95_confirm._grid_resolve_atomic_tokens(UNI_CELL, object()) is None


class TestExtrasResolveAtomicTokens:
    """The extras resolver's two-tier fallback (grid BPE first, extras BPE).

    Mirrors ``TestResolveAtomicTokens`` for the grid resolver, but the Unigram
    path has an extra fallback the grid path lacks: a same-(corpus, boundary)
    extras BPE artifact when no grid BPE one is loadable.
    """

    def test_bpe_cell_uses_its_own_glyph_alphabet(self) -> None:
        from smiles_subword.tokenize.adapters.smirk import SmirkAdapter

        tok = SmirkAdapter.atomic()
        cell = SimpleNamespace(algo="bpe", corpus="pubchem", boundary="nmb")

        atomic = f95_confirm._extras_resolve_atomic_tokens(cell, tok)  # type: ignore[arg-type]

        assert atomic == frozenset(tok.hf_tokenizer.get_vocab())

    def test_unigram_prefers_the_grid_bpe_alphabet(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cell = SimpleNamespace(algo="unigram", corpus="pubchem", boundary="nmb")
        monkeypatch.setattr(
            f95_confirm, "_grid_bpe_alphabet", lambda _c, _b: frozenset({"C", "O"})
        )

        def must_not_fall_back(_c: str, _b: str) -> Never:
            raise AssertionError("grid alphabet present; extras fallback must not fire")

        monkeypatch.setattr(f95_confirm, "_extras_bpe_alphabet", must_not_fall_back)

        atomic = f95_confirm._extras_resolve_atomic_tokens(cell, object())  # type: ignore[arg-type]

        assert atomic == frozenset({"C", "O"})

    def test_unigram_falls_back_to_an_extras_bpe_alphabet(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cell = SimpleNamespace(algo="unigram", corpus="pubchem", boundary="nmb")
        monkeypatch.setattr(f95_confirm, "_grid_bpe_alphabet", lambda _c, _b: None)
        monkeypatch.setattr(
            f95_confirm, "_extras_bpe_alphabet", lambda _c, _b: frozenset({"N"})
        )

        atomic = f95_confirm._extras_resolve_atomic_tokens(cell, object())  # type: ignore[arg-type]

        assert atomic == frozenset({"N"})

    def test_unigram_returns_none_when_neither_is_trained(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cell = SimpleNamespace(algo="unigram", corpus="pubchem", boundary="nmb")
        monkeypatch.setattr(f95_confirm, "_grid_bpe_alphabet", lambda _c, _b: None)
        monkeypatch.setattr(f95_confirm, "_extras_bpe_alphabet", lambda _c, _b: None)

        assert (
            f95_confirm._extras_resolve_atomic_tokens(cell, object())  # type: ignore[arg-type]
            is None
        )


class TestConfirmCellWrapper:
    """``confirm_cell`` binds the grid seams onto the shared ``run_confirm``."""

    def test_binds_grid_seams_onto_run_confirm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, object] = {}

        def spy(cell: object, **kwargs: object) -> None:
            captured["cell"] = cell
            captured.update(kwargs)
            return

        monkeypatch.setattr(f95_confirm._runtime, "run_confirm", spy)

        assert f95_confirm.confirm_cell(BPE_CELL, force=True) is None
        assert captured["cell"] == BPE_CELL
        assert captured["to_config"] is grid_cell_to_config
        assert (
            captured["resolve_atomic_tokens"] is f95_confirm._grid_resolve_atomic_tokens
        )
        assert captured["force"] is True
        assert captured["dry_run"] is False


class TestConfirmExtrasCellWrapper:
    """``confirm_extras_cell`` binds the extras seams onto the shared core."""

    def test_binds_extras_seams_onto_run_confirm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, object] = {}

        def spy(cell: object, **kwargs: object) -> None:
            captured["cell"] = cell
            captured.update(kwargs)
            return

        monkeypatch.setattr(f95_confirm._runtime, "run_confirm", spy)
        cell = SimpleNamespace(cell_id="extras-x")

        assert f95_confirm.confirm_extras_cell(cell) is None  # type: ignore[arg-type]
        assert captured["cell"] is cell
        assert captured["to_config"] is extras_cell_to_config
        assert (
            captured["resolve_atomic_tokens"]
            is f95_confirm._extras_resolve_atomic_tokens
        )
