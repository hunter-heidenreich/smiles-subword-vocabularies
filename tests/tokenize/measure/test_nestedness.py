"""Tests for ``smiles_subword.tokenize.measure.nestedness`` (pure computation)."""

from __future__ import annotations

import math

import pytest

from smiles_subword.tokenize.measure.nestedness import (
    CLASSES,
    PerMoleculeNestedness,
    bootstrap_seed,
    classify_piece,
    compare_molecule,
    compute_pair_nestedness,
    make_unpaired_nestedness,
)


class TestClassifyPiece:
    def test_bracket_piece_takes_priority(self) -> None:
        # contains an aromatic glyph but the '[' wins.
        assert classify_piece(("[", "c", "H", "]")) == "bracket"

    def test_aromatic_atom(self) -> None:
        assert classify_piece(("c", "c", "c")) == "aromatic"

    def test_aliphatic_heteroatom(self) -> None:
        assert classify_piece(("C", "O", "C")) == "heteroatom"

    def test_multichar_halogen_is_heteroatom(self) -> None:
        assert classify_piece(("C", "Cl")) == "heteroatom"

    def test_unsaturated_carbon(self) -> None:
        assert classify_piece(("C", "=", "C")) == "unsat-C"

    def test_saturated_carbon(self) -> None:
        assert classify_piece(("C", "C", "C")) == "sat-C"

    def test_every_class_is_in_CLASSES(self) -> None:
        for piece in [
            ("[", "O", "-"),
            ("c", "c"),
            ("C", "N"),
            ("C", "#", "C"),
            ("C", "C"),
        ]:
            assert classify_piece(piece) in CLASSES


class TestCompareMolecule:
    def test_perfect_nesting_bpe_coarser(self) -> None:
        # glyph stream length 5; BPE cuts {2}, UL atomic cuts {1,2,3,4}.
        result = compare_molecule([2, 3], [("C",), ("C",), ("C",), ("C",), ("C",)])

        assert result.n_positions == 4
        assert result.n_agree_cut == 1  # position 2
        assert result.n_nest == 3  # 1,3,4 — UL cuts, BPE merges
        assert result.n_conflict == 0
        assert result.n_agree_merge == 0
        assert result.is_nested is True

    def test_conflict_when_bpe_cuts_inside_a_ul_piece(self) -> None:
        # length 5; BPE cuts {1,2}; UL pieces ('C','C')|('C','C','C') cut {2}.
        result = compare_molecule([1, 1, 3], [("C", "C"), ("C", "C", "C")])

        assert result.n_agree_cut == 1  # position 2
        assert result.n_nest == 0
        assert result.n_conflict == 1  # position 1, inside UL's ('C','C')
        assert result.n_agree_merge == 2  # positions 3,4
        assert result.is_nested is False

    def test_conflict_localizes_to_the_spanning_ul_piece(self) -> None:
        result = compare_molecule([1, 1, 3], [("C", "C"), ("C", "C", "C")])
        sat_idx = CLASSES.index("sat-C")

        # both UL pieces are saturated-C; only ('C','C') is cut through.
        assert result.emitted_by_class[sat_idx] == 2
        assert result.cut_through_by_class[sat_idx] == 1

    def test_identical_segmentations_all_agree(self) -> None:
        result = compare_molecule([1, 1, 1], [("C",), ("C",), ("C",)])

        assert result.n_agree_cut == 2
        assert result.n_nest == 0
        assert result.n_conflict == 0
        assert result.is_nested is True

    def test_single_glyph_molecule_has_no_positions(self) -> None:
        result = compare_molecule([1], [("C",)])

        assert result.n_positions == 0
        assert result.is_nested is True

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="glyph-length mismatch"):
            compare_molecule([2], [("C",)])


def _pm(
    *,
    positions: int,
    agree_cut: int,
    nest: int,
    conflict: int,
    emitted: tuple[int, ...] = (0, 0, 0, 0, 0),
    cut_through: tuple[int, ...] = (0, 0, 0, 0, 0),
) -> PerMoleculeNestedness:
    agree_merge = positions - agree_cut - nest - conflict
    return PerMoleculeNestedness(
        n_positions=positions,
        n_agree_cut=agree_cut,
        n_nest=nest,
        n_conflict=conflict,
        n_agree_merge=agree_merge,
        emitted_by_class=emitted,
        cut_through_by_class=cut_through,
    )


def _pair(per_molecule: list[PerMoleculeNestedness]) -> object:
    return compute_pair_nestedness(
        per_molecule,
        pair_key="pubchem__v1024_nmb",
        tier="headline",
        corpus="pubchem",
        vocab_size=1024,
        boundary="nmb",
        bpe_cell_id="pubchem__smirk_gpe_v1024_nmb",
        unigram_cell_id="pubchem__smirk_unigram_v1024_nmb",
        bpe_training_corpus_sha="sha-bpe",
        unigram_training_corpus_sha="sha-ul",
        eval_split_sha="eval-A",
    )


class TestComputePairNestedness:
    def test_aggregates_the_2x2_and_headline_scalars(self) -> None:
        per_mol = [
            _pm(positions=10, agree_cut=6, nest=3, conflict=0),
            _pm(positions=10, agree_cut=5, nest=2, conflict=1),
        ]

        pair = _pair(per_mol)

        assert pair.n_agree_cut == 11
        assert pair.n_nest == 5
        assert pair.n_conflict == 1
        assert pair.n_positions == 20
        # boundary Jaccard = agree_cut / (agree_cut + nest + conflict)
        assert pair.boundary_jaccard == pytest.approx(11 / 17)
        assert pair.conflict_rate == pytest.approx(1 / 20)
        assert pair.nest_rate == pytest.approx(5 / 20)
        assert pair.conflict_share_of_disagreement == pytest.approx(1 / 6)

    def test_nested_molecule_fraction_counts_zero_conflict_molecules(self) -> None:
        per_mol = [
            _pm(positions=5, agree_cut=3, nest=1, conflict=0),  # nested
            _pm(positions=5, agree_cut=3, nest=1, conflict=1),  # not nested
            _pm(positions=5, agree_cut=4, nest=1, conflict=0),  # nested
        ]

        pair = _pair(per_mol)

        assert pair.n_nested_molecules == 2
        assert pair.nested_molecule_fraction == pytest.approx(2 / 3)

    def test_per_class_localization_and_cut_rate(self) -> None:
        het = CLASSES.index("heteroatom")
        sat = CLASSES.index("sat-C")
        emit = [0, 0, 0, 0, 0]
        cut = [0, 0, 0, 0, 0]
        emit[het], cut[het] = 10, 4
        emit[sat], cut[sat] = 20, 1
        per_mol = [
            _pm(
                positions=8,
                agree_cut=5,
                nest=2,
                conflict=1,
                emitted=tuple(emit),
                cut_through=tuple(cut),
            )
        ]

        pair = _pair(per_mol)
        payload = pair.as_dict()
        cut_rate = payload["cut_rate_by_class"]

        assert payload["emitted_by_class"]["heteroatom"] == 10
        assert cut_rate["heteroatom"] == pytest.approx(0.4)
        assert cut_rate["sat-C"] == pytest.approx(0.05)

    def test_cut_rate_is_none_for_unemitted_class(self) -> None:
        pair = _pair([_pm(positions=5, agree_cut=3, nest=2, conflict=0)])

        assert pair.as_dict()["cut_rate_by_class"]["aromatic"] is None

    def test_bootstrap_ci_is_deterministic_for_same_pair(self) -> None:
        per_mol = [
            _pm(positions=10, agree_cut=6, nest=3, conflict=1) for _ in range(30)
        ]

        a = _pair(per_mol)
        b = _pair(per_mol)

        assert a.boundary_jaccard_ci == b.boundary_jaccard_ci
        assert a.conflict_rate_ci == b.conflict_rate_ci
        assert a.bootstrap_seed == b.bootstrap_seed

    def test_distinct_pairs_have_distinct_seeds(self) -> None:
        assert bootstrap_seed("pubchem__v256_nmb") != bootstrap_seed("pubchem__v256_mb")

    def test_empty_input_yields_nan_scalars(self) -> None:
        pair = _pair([])

        assert math.isnan(pair.boundary_jaccard)
        assert math.isnan(pair.nested_molecule_fraction)
        assert pair.n_molecules == 0


class TestUnpaired:
    def test_make_unpaired_records_present_arm(self) -> None:
        rec = make_unpaired_nestedness(
            pair_key="zinc22__v2048_nmb",
            tier="conditional",
            corpus="zinc22",
            vocab_size=2048,
            boundary="nmb",
            extras_kind=None,
            extras_label=None,
            present_arm="bpe",
            present_cell_id="zinc22__smirk_gpe_v2048_nmb",
            present_training_corpus_sha="sha-cond",
            eval_split_sha="eval-A",
            missing_arm="unigram",
            unpaired_reason="conditional_negative_branch",
        )

        assert rec.pair_status == "single_arm"
        assert rec.missing_arm == "unigram"
        assert rec.as_dict()["present_cell_id"] == "zinc22__smirk_gpe_v2048_nmb"

    def test_missing_arm_equal_to_present_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot equal present_arm"):
            make_unpaired_nestedness(
                pair_key="pk",
                tier="conditional",
                corpus="zinc22",
                vocab_size=2048,
                boundary="nmb",
                extras_kind=None,
                extras_label=None,
                present_arm="bpe",
                present_cell_id="x",
                present_training_corpus_sha="sha",
                eval_split_sha="eval-A",
                missing_arm="bpe",
                unpaired_reason="conditional_negative_branch",
            )
