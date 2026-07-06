"""Tests for ``fg_alignment_runner`` chemistry + mapping helpers.

Exercises the bracket-aware glyph segmenter, the atom-to-glyph-span mapping,
the multiply-bonded-heteroatom classification off a real RDKit graph, and the
per-molecule locality assembly with controlled token boundaries (no adapters).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from rdkit import Chem

from smiles_subword.tokenize.adapters.smirk import SmirkAdapter
from smiles_subword.tokenize.measure import _cells
from smiles_subword.tokenize.measure.fg_alignment import runner as fg_alignment_runner
from smiles_subword.tokenize.measure.fg_alignment.runner import (
    atom_spans,
    build_glyph_segmenter,
    mappable_functional_bonds,
    molecule_arms_locality,
    run_pair_fg_alignment,
    run_single_arm_fg_alignment,
)

# A glyph alphabet rich enough for the test molecules.
_GLYPHS = frozenset([*list("CHNOPSFIbcnops()[]=#+-123456"), "Cl", "Br"])
_SEG = build_glyph_segmenter(_GLYPHS)

# A len-1 glyph-tuple map (the runner reads the alphabet off length-1 entries).
_TUPLE_MAP = {i: (g,) for i, g in enumerate("CHNOPSFIbcnops()[]=#+-123456")}
_FIXTURE_SMILES = ["CCO", "CC(=O)O", "N#Cc1ccncc1"]


class TestSegmenter:
    def test_basic_round_trip(self) -> None:
        assert _SEG("CC(=O)N") == ("C", "C", "(", "=", "O", ")", "N")

    def test_bracket_disambiguates_two_letter_element(self) -> None:
        # Bare "Cn" is carbon + aromatic-n, not copernicium.
        assert _SEG("Cn") == ("C", "n")

    def test_bracket_atom_inside_keeps_multichar(self) -> None:
        assert _SEG("[nH]") == ("[", "n", "H", "]")

    def test_out_of_alphabet_returns_none(self) -> None:
        assert _SEG("C%C") is None  # '%' not in the test alphabet


class TestAtomSpans:
    def test_bare_and_bracket_atoms(self) -> None:
        glyphs = _SEG("CC(=O)N")
        assert glyphs is not None
        # four atoms at glyph indices 0, 1, 4, 6; structural glyphs are glue
        assert atom_spans(glyphs) == [(0, 1), (1, 2), (4, 5), (6, 7)]

    def test_bracket_atom_spans_whole_bracket(self) -> None:
        glyphs = _SEG("C[nH]")
        assert glyphs is not None
        assert atom_spans(glyphs) == [(0, 1), (1, 5)]

    def test_unbalanced_bracket_is_none(self) -> None:
        assert atom_spans(("C", "[", "n")) is None

    def test_atom_count_matches_rdkit(self) -> None:
        for smi in ("CC(=O)Nc1ccccc1", "O=S(=O)(N)c1ccccc1", "N#Cc1ccncc1"):
            glyphs = _SEG(smi)
            assert glyphs is not None
            spans = atom_spans(glyphs)
            assert spans is not None
            assert len(spans) == Chem.MolFromSmiles(smi).GetNumAtoms()


class TestMappableBonds:
    def _labels(self, smi: str) -> list[str]:
        mol = Chem.MolFromSmiles(smi)
        glyphs = _SEG(smi)
        assert glyphs is not None
        spans = atom_spans(glyphs)
        assert spans is not None
        return [label for label, _pos in mappable_functional_bonds(mol, spans, glyphs)]

    def test_carbonyl_and_nitrile_classified(self) -> None:
        assert self._labels("CC(=O)N") == ["C=O"]
        assert self._labels("CC#N") == ["C#N"]

    def test_heteroatom_central_classes(self) -> None:
        assert "S=O" in self._labels("CS(=O)(=O)C")
        assert "N=O" in self._labels("C[N+](=O)[O-]")

    def test_pure_carbon_unsaturation_excluded(self) -> None:
        # A double or triple bond between two carbons carries no heteroatom, so
        # it is not a functional bond and must not be counted.
        assert self._labels("C=CC") == []
        assert self._labels("CC#CC") == []


class TestMoleculeArmsLocality:
    def test_local_vs_straddle_arms(self) -> None:
        smi = "CC(=O)N"  # one carbonyl; glyph length 7
        # Arm A: a single token over the whole molecule -> no interior cut.
        # Arm B: a cut at position 4 (just before the carbonyl O) -> straddle.
        result = molecule_arms_locality(
            smi,
            arm_ids=[[100], [101, 102]],
            arm_glyph_len=[{100: 7}, {101: 4, 102: 3}],
            seg=_SEG,
        )
        assert result is not None
        arm_a, arm_b = result
        assert arm_a.n_bonds == 1
        assert arm_a.n_local == 1  # kept the =O local
        assert arm_a.class_local["C=O"] == 1
        assert arm_b.n_local == 0  # cut through it
        assert arm_b.class_bonds["C=O"] == 1

    def test_unparseable_molecule_dropped(self) -> None:
        assert molecule_arms_locality("not_a_smiles_(((", [[1]], [{1: 5}], _SEG) is None

    def test_glyph_length_mismatch_dropped(self) -> None:
        # The encoded glyph length (8) disagrees with the segmenter length (7).
        assert molecule_arms_locality("CC(=O)N", [[1]], [{1: 8}], _SEG) is None


@pytest.fixture
def _stub_runner_deps(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Patch the runner's I/O so it streams a fixed SMILES list (no artifacts).

    With the atomic glyph tokenizer as the arm(s) and a default 1-glyph count
    per token, the encoded length always matches the segmenter, so every fixture
    molecule is usable.
    """
    monkeypatch.setattr(fg_alignment_runner, "glyph_count_map", lambda *a, **k: {})
    monkeypatch.setattr(
        fg_alignment_runner, "glyph_tuple_map", lambda *a, **k: _TUPLE_MAP
    )
    monkeypatch.setattr(
        fg_alignment_runner, "tokenizer_artifact_dir", lambda *a, **k: tmp_path
    )
    monkeypatch.setattr(
        _cells,
        "iter_smiles_from_parquet",
        lambda *_a: iter(_FIXTURE_SMILES),
    )


@pytest.mark.usefixtures("_stub_runner_deps")
class TestRunOverHeldOut:
    def test_run_single_arm(self) -> None:
        arm = run_single_arm_fg_alignment(
            SmirkAdapter.atomic(),
            arm="bpe",
            corpus="pubchem",
            name="x",
            cell_id="pubchem__x",
            boundary="nmb",
            training_corpus_sha="t",
            eval_split_sha_value="e",
            batch_size=2,
        )
        # All three fixture molecules parse and carry a functional bond
        # (CC(=O)O -> C=O; N#Cc1ccncc1 -> C#N).
        assert arm.n_molecules == 3
        assert arm.n_bonds >= 2
        assert arm.arm == "bpe"

    def test_run_pair(self) -> None:
        rec = run_pair_fg_alignment(
            SmirkAdapter.atomic(),
            SmirkAdapter.atomic(),
            pair_key="pubchem__v256_nmb",
            tier="headline",
            corpus="pubchem",
            vocab_size=256,
            boundary="nmb",
            bpe_cell_id="pubchem__smirk_gpe_v256_nmb",
            unigram_cell_id="pubchem__smirk_unigram_v256_nmb",
            bpe_name="b",
            unigram_name="u",
            bpe_training_corpus_sha="tb",
            unigram_training_corpus_sha="tu",
            eval_split_sha_value="e",
            limit_molecules=2,
            batch_size=2,
        )
        assert rec.pair_status == "matched"
        assert rec.bpe.n_molecules == 2  # limit_molecules honored
        assert rec.unigram.n_molecules == 2
