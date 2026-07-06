"""Tests for ``smiles_subword.tokenize.measure.supplementary.transfer.math``."""

from __future__ import annotations

import pytest

from smiles_subword.tokenize.measure.supplementary.transfer.math import (
    PerMoleculeTransfer,
    TransferRecord,
    compute_transfer_record,
)
from smiles_subword.tokenize.measure.supplementary.transfer.runner import (
    count_per_molecule,
    enumerate_transfer_cells,
)


class _FakeAdapter:
    """Minimal encode_batch surface for ``count_per_molecule``."""

    def __init__(self, enc: dict[str, list[int]]) -> None:
        self._enc = enc

    def encode_batch(
        self, smiles: list[str], *, add_special_tokens: bool = False
    ) -> list[list[int]]:
        return [self._enc[s] for s in smiles]


class TestCountPerMolecule:
    """The per-molecule tally: tokens, glyphs, and atom-level OOV ([UNK])."""

    def test_tallies_tokens_glyphs_and_unk(self) -> None:
        adapter = _FakeAdapter({"a": [0, 3, 4], "b": [3, 3], "c": [0, 0, 5]})
        # id 0 is the [UNK]/OOV id (absent from glyph_map -> counts as 1 glyph).
        glyph_map = {3: 2, 4: 1, 5: 3}

        pm = count_per_molecule(adapter, glyph_map, 0, ["a", "b", "c"])

        assert (pm[0].n_tokens, pm[0].n_glyphs, pm[0].n_unk) == (3, 4, 1)
        assert (pm[1].n_tokens, pm[1].n_glyphs, pm[1].n_unk) == (2, 4, 0)
        assert (pm[2].n_tokens, pm[2].n_glyphs, pm[2].n_unk) == (3, 5, 2)

    def test_no_unk_id_yields_zero_oov(self) -> None:
        adapter = _FakeAdapter({"a": [0, 3, 0]})

        pm = count_per_molecule(adapter, {3: 2}, None, ["a"])

        assert pm[0].n_unk == 0  # no atom-level OOV id for this arm


def _record(**kw: object) -> TransferRecord:
    base: dict[str, object] = {
        "train_corpus": "pubchem",
        "eval_corpus": "coconut",
        "arm": "bpe",
        "vocab_size": 1024,
        "boundary": "nmb",
        "train_corpus_sha": "T",
        "eval_split_sha": "E",
    }
    base.update(kw)
    return compute_transfer_record(**base)  # type: ignore[arg-type]


class TestComputeTransferRecord:
    def test_aggregates_fertility_glyphs_and_oov(self) -> None:
        per = [
            PerMoleculeTransfer(n_tokens=4, n_glyphs=6, n_unk=0),
            PerMoleculeTransfer(n_tokens=2, n_glyphs=5, n_unk=1),
            PerMoleculeTransfer(n_tokens=3, n_glyphs=3, n_unk=0),
        ]
        rec = _record(per_molecule=per)

        assert rec.n_molecules == 3
        assert rec.total_tokens == 9
        assert rec.fertility_mean == pytest.approx(3.0)
        assert rec.glyphs_per_token_mean == pytest.approx(14 / 9)
        assert rec.oov_token_rate == pytest.approx(1 / 9)
        assert rec.oov_molecule_rate == pytest.approx(1 / 3)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            _record(per_molecule=[])

    def test_diagonal_flag_and_cell_key(self) -> None:
        diag = _record(
            eval_corpus="pubchem", per_molecule=[PerMoleculeTransfer(1, 1, 0)]
        )
        off = _record(per_molecule=[PerMoleculeTransfer(1, 1, 0)])
        assert diag.is_diagonal is True
        assert off.is_diagonal is False
        assert off.cell_key == "pubchem__coconut__gpe_v1024_nmb"

    def test_unigram_cell_key_tag(self) -> None:
        rec = _record(arm="unigram", per_molecule=[PerMoleculeTransfer(1, 1, 0)])
        assert rec.cell_key == "pubchem__coconut__unigram_v1024_nmb"

    def test_ci_is_deterministic_for_fixed_key(self) -> None:
        per = [
            PerMoleculeTransfer(n_tokens=t, n_glyphs=t, n_unk=0) for t in range(1, 40)
        ]
        a = _record(per_molecule=per)
        b = _record(per_molecule=per)
        assert a.fertility_ci == b.fertility_ci
        lo, hi = a.fertility_ci
        assert lo <= a.fertility_mean <= hi


class TestEnumerateCells:
    def test_off_diagonal_only_at_headline_v(self) -> None:
        cells = enumerate_transfer_cells()
        # lean scope: 12 off-diagonal ordered pairs x 2 arms
        assert len(cells) == 24
        assert all(train != ev for train, ev, _arm, _v, _b in cells)
        assert all(v == 1024 and b == "nmb" for _t, _e, _arm, v, b in cells)
        assert {c[2] for c in cells} == {"bpe", "unigram"}
        assert {c[0] for c in cells} == {"pubchem", "zinc22", "coconut", "real_space"}
