"""Tests for ``noncanon_runner`` (variant generation + reduction + orbit pass)."""

from __future__ import annotations

import pytest
from rdkit import Chem

from smiles_subword.tokenize.adapters.smirk import SmirkAdapter
from smiles_subword.tokenize.measure import _cells
from smiles_subword.tokenize.measure.noncanon.runner import (
    explicit_h,
    kekulized,
    openbabel_available,
    openbabel_canon,
    per_molecule_readings,
    perturb_rings,
    randomized,
    run_pair_noncanon,
    run_single_arm_noncanon,
)

_FIXTURE_SMILES = ["CCO", "CC(=O)O", "c1ccccc1", "CC#N", "c1ccncc1"]


class TestVariantHelpers:
    def test_randomized_count_and_validity(self) -> None:
        mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
        rs = randomized(mol, 5, seed=7)
        assert len(rs) == 5
        canon = Chem.MolToSmiles(mol)
        for r in rs:
            assert Chem.MolToSmiles(Chem.MolFromSmiles(r)) == canon  # same molecule

    def test_kekulized_only_aromatic(self) -> None:
        assert kekulized(Chem.MolFromSmiles("CCO")) is None  # aliphatic
        kek = kekulized(Chem.MolFromSmiles("c1ccccc1"))
        assert kek is not None
        assert "c" not in kek  # aromatic lowercase gone after kekulization

    def test_explicit_h_brackets_atoms(self) -> None:
        eh = explicit_h(Chem.MolFromSmiles("CCO"))
        assert eh is not None
        assert "[" in eh  # atoms now bracketed with explicit H counts

    def test_perturb_rings_relabels_and_preserves(self) -> None:
        out = perturb_rings("c1ccccc1")
        assert out is not None
        assert out != "c1ccccc1"  # the digit was relabeled
        assert Chem.MolToSmiles(Chem.MolFromSmiles(out)) == "c1ccccc1"  # same molecule

    def test_perturb_rings_none_without_rings(self) -> None:
        assert perturb_rings("CCO") is None

    def test_perturb_rings_skips_two_digit_closures(self) -> None:
        assert perturb_rings("C%10CCCCCCCCCC%10") is None  # %nn -> skipped

    def test_openbabel_canon_roundtrips_and_gates(self) -> None:
        if not openbabel_available():
            pytest.skip("openbabel (crosstoolkit extra) not installed")
        # Caffeine: OpenBabel traverses the rings differently from RDKit, so the
        # string differs but must round-trip to the identical molecule.
        caf = Chem.MolToSmiles(Chem.MolFromSmiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C"))
        ob = openbabel_canon(caf)
        assert ob is not None
        assert ob != caf  # a genuine cross-toolkit rewrite
        assert Chem.MolToSmiles(Chem.MolFromSmiles(ob)) == caf  # identity preserved
        # Agreement is a genuine zero, not a skip: returns the canonical unchanged.
        eth = Chem.MolToSmiles(Chem.MolFromSmiles("CCO"))
        assert openbabel_canon(eth) == eth
        # Unparseable input is gated out.
        assert openbabel_canon("not_a_smiles") is None


class TestPerMoleculeReadings:
    def test_reduction(self) -> None:
        keys = [
            (0, "canonical"),
            (0, "random"),
            (0, "random"),
            (0, "kekule"),
            (0, "ringperm"),
            (0, "explicitH"),
        ]
        ids = [[1, 2, 3], [1, 2, 3, 4], [1, 2], [9, 9], [1, 2, 3], [5, 5, 5, 5, 5, 5]]
        (pm,) = per_molecule_readings(ids, keys)

        assert pm.canon_fert == 3
        assert pm.rand_fert_mean == pytest.approx(3.0)  # (4 + 2) / 2
        assert pm.axis_dfert["random"] == pytest.approx((1 / 3 + 1 / 3) / 2)
        assert pm.axis_dfert["ringperm"] == pytest.approx(0.0)  # count invariant
        assert pm.axis_bag["ringperm"] == pytest.approx(0.0)  # identical id multiset
        assert pm.axis_bag["kekule"] == pytest.approx(1.0)  # disjoint ids
        assert pm.axis_dfert["explicitH"] == pytest.approx(1.0)  # 6 vs 3 tokens

    def test_skips_molecule_without_canonical(self) -> None:
        assert per_molecule_readings([[1, 2]], [(0, "random")]) == []


@pytest.fixture
def _stub_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        _cells, "iter_smiles_from_parquet", lambda *_a: iter(_FIXTURE_SMILES)
    )


@pytest.mark.usefixtures("_stub_stream")
class TestOrbitPass:
    def test_run_single_arm(self) -> None:
        arm = run_single_arm_noncanon(
            SmirkAdapter.atomic(),
            arm="bpe",
            corpus="pubchem",
            cell_id="pubchem__x",
            boundary="nmb",
            training_corpus_sha="t",
            eval_split_sha_value="e",
            limit_molecules=10,
        )
        assert arm.n_molecules == len(_FIXTURE_SMILES)
        assert "random" in arm.axes  # always generated
        assert "kekule" in arm.axes  # the fixture has aromatic molecules
        if openbabel_available():
            assert "obcanon" in arm.axes  # cross-toolkit canonical swap

    def test_run_pair(self) -> None:
        rec = run_pair_noncanon(
            SmirkAdapter.atomic(),
            SmirkAdapter.atomic(),
            pair_key="pubchem__v1024_nmb",
            tier="headline",
            corpus="pubchem",
            vocab_size=1024,
            boundary="nmb",
            bpe_cell_id="pubchem__smirk_gpe_v1024_nmb",
            unigram_cell_id="pubchem__smirk_unigram_v1024_nmb",
            bpe_training_corpus_sha="tb",
            unigram_training_corpus_sha="tu",
            eval_split_sha_value="e",
            limit_molecules=10,
        )
        assert rec.pair_status == "matched"
        # identical atomic tokenizers -> the cross-arm gap ratio is 1.
        assert rec.gap_canon == pytest.approx(1.0)
