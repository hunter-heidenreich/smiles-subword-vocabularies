"""Tests for ``smiles_subword.preprocess.dative``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rdkit import Chem

from smiles_subword.preprocess.dative import DATIVE_BOND_TYPES, dative_to_opensmiles

if TYPE_CHECKING:
    import pytest

_DATIVE = "CN(C)C=O->[Y+3](<-O)<-O"


def _has_dative(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return any(b.GetBondType() in DATIVE_BOND_TYPES for b in mol.GetBonds())


class TestDativeToOpenSmiles:
    def test_input_actually_uses_dative_bonds(self) -> None:
        assert _has_dative(_DATIVE)

    def test_conversion_removes_dative_arrow_glyphs(self) -> None:
        out = dative_to_opensmiles(_DATIVE)
        assert out is not None
        assert ">" not in out
        assert "<" not in out

    def test_metal_atom_is_preserved(self) -> None:
        out = dative_to_opensmiles(_DATIVE)
        assert out is not None
        assert "Y" in out

    def test_non_dative_input_is_canonicalized_passthrough(self) -> None:
        assert dative_to_opensmiles("OCC") == Chem.MolToSmiles(
            Chem.MolFromSmiles("OCC")
        )

    def test_output_is_canonical_idempotent(self) -> None:
        once = dative_to_opensmiles(_DATIVE)
        assert once is not None
        assert dative_to_opensmiles(once) == once

    def test_unparseable_input_returns_none(self) -> None:
        assert dative_to_opensmiles("not a smiles [[[") is None

    def test_conversion_failure_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A dative->single rewrite can leave a molecule MolToSmiles rejects
        # (bad valence/kekulization); that raise must be swallowed as a drop.
        def _raise(*_args: object, **_kwargs: object) -> str:
            raise RuntimeError("simulated write failure")

        monkeypatch.setattr("smiles_subword.preprocess.dative.Chem.MolToSmiles", _raise)
        assert dative_to_opensmiles(_DATIVE) is None
