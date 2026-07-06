"""Tests for ``closure`` (pure compositional-closure math).

Covers the three metrics, the BPE merge-closure invariant
(``c_bin == 1`` for any vocabulary built by iterated in-vocab concatenation),
a hand-built non-closed Unigram example, and the matched / unpaired joins.
"""

from __future__ import annotations

import pytest

from smiles_subword.tokenize.measure.closure import (
    ArmClosure,
    binary_split_closed,
    compute_arm_closure,
    compute_matched_pair_closure,
    compute_unpaired_closure,
    full_substring_closed,
    is_orphan,
    proper_substrings,
)

C = ("C",)
CC = ("C", "C")
CCC = ("C", "C", "C")
CCCC = ("C", "C", "C", "C")
CO = ("C", "O")


def _arm(
    tuples: list[tuple[str, ...]], *, arm: str = "unigram", sha: str = "sha"
) -> ArmClosure:
    return compute_arm_closure(
        tuples,
        cell_id=f"pubchem__smirk_{'gpe' if arm == 'bpe' else 'unigram'}_v256_nmb",
        arm=arm,  # type: ignore[arg-type]
        boundary="nmb",
        vocab_size=256,
        training_corpus_sha=sha,
    )


class TestProperSubstrings:
    def test_excludes_whole_and_singletons(self) -> None:
        assert proper_substrings(CCC) == [CC, CC]  # only length-2 windows

    def test_length_four_windows(self) -> None:
        subs = proper_substrings(CCCC)
        assert subs.count(CC) == 3  # three length-2 windows
        assert subs.count(CCC) == 2  # two length-3 windows
        assert CCCC not in subs  # whole tuple excluded

    def test_too_short_is_empty(self) -> None:
        assert proper_substrings(CC) == []


class TestBinarySplitClosed:
    def test_closed_when_a_split_is_in_vocab(self) -> None:
        vocab = frozenset({C, CC, CCC})
        assert binary_split_closed(CCC, vocab) is True  # (C, CC) or (CC, C)

    def test_not_closed_without_the_len2_piece(self) -> None:
        # Only single glyphs in vocab: a length-3 split always needs a >=2 part.
        assert binary_split_closed(CCC, frozenset({C, ("O",)})) is False


class TestIsOrphan:
    def test_orphan_when_no_multi_subpiece(self) -> None:
        assert is_orphan(CCCC, frozenset({CCCC, CO})) is True

    def test_not_orphan_with_a_multi_subpiece(self) -> None:
        assert is_orphan(CCCC, frozenset({CCCC, CC})) is False


class TestFullSubstringClosed:
    def test_all_substrings_present(self) -> None:
        assert full_substring_closed(CCC, frozenset({C, CC, CCC})) is True

    def test_one_missing_substring_breaks_it(self) -> None:
        assert full_substring_closed(CCCC, frozenset({C, CC, CCCC})) is False  # no CCC


class TestComputeArmClosureInvariant:
    def test_merge_closed_vocab_is_fully_binary_closed(self) -> None:
        # Simulate BPE: every multi piece is the concatenation of two in-vocab
        # pieces, so c_bin must be exactly 1 and the orphan rate exactly 0.
        base = [("C",), ("O",), ("N",)]
        merges = [CC, CCC, CCCC, ("C", "O"), ("C", "O", "N")]
        arm = _arm(base + merges, arm="bpe")

        assert arm.c_bin == 1.0
        assert arm.c_orph == 0.0
        assert arm.n_multi == len(merges)

    def test_unigram_like_vocab_has_orphans(self) -> None:
        # CCCC floats free (no CC/CCC in vocab); CO is binary-closed via C+O.
        arm = _arm([("C",), ("O",), CCCC, CO])

        assert arm.n_multi == 2
        assert arm.c_bin == pytest.approx(0.5)  # CO closed, CCCC not
        assert arm.n_ge3 == 1  # only CCCC is length >= 3
        assert arm.c_orph == pytest.approx(1.0)  # CCCC is an orphan
        assert arm.c_full == pytest.approx(0.0)

    def test_empty_multi_yields_nan(self) -> None:
        arm = _arm([("C",), ("O",)])
        assert arm.n_multi == 0
        assert arm.c_bin != arm.c_bin  # nan


class TestBracketNonClosed:
    """``n_bracket_nonclosed`` counts bracket-glyph multi-pieces that are not
    binary-split-closed (a reported stratification count)."""

    def test_counts_an_unclosed_bracket_piece(self) -> None:
        # ("[O-]", "C") is a bracket multi-piece; ("[O-]",) is absent from the
        # vocab, so it is not binary-split-closed and is counted.
        arm = _arm([("C",), ("[O-]", "C")])
        assert arm.n_bracket_nonclosed == 1

    def test_closed_bracket_piece_is_not_counted(self) -> None:
        # With ("[O-]",) in the vocab, ("[O-]", "C") is binary-split-closed, so
        # the bracket-nonclosed count drops to zero.
        arm = _arm([("C",), ("[O-]",), ("[O-]", "C")])
        assert arm.n_bracket_nonclosed == 0


class TestComputeMatchedPairClosure:
    def _pair(self):  # noqa: ANN202 - test helper
        bpe = _arm([("C",), CC, CCC], arm="bpe")
        ul = _arm([("C",), ("O",), CCCC, CO])
        return compute_matched_pair_closure(
            bpe,
            ul,
            pair_key="pubchem__v256_nmb",
            tier="headline",
            corpus="pubchem",
            vocab_size=256,
            boundary="nmb",
        )

    def test_gaps_are_bpe_minus_ul(self) -> None:
        pair = self._pair()
        assert pair.delta_c_bin == pytest.approx(1.0 - 0.5)
        assert pair.pair_status == "matched"

    def test_rejects_swapped_arms(self) -> None:
        bpe = _arm([("C",), CC], arm="bpe")
        ul = _arm([("C",), CO])
        with pytest.raises(ValueError, match="first arg must be the BPE arm"):
            compute_matched_pair_closure(
                ul,  # type: ignore[arg-type]
                bpe,  # type: ignore[arg-type]
                pair_key="p",
                tier="headline",
                corpus="pubchem",
                vocab_size=256,
                boundary="nmb",
            )

    def test_rejects_boundary_mismatch(self) -> None:
        bpe = _arm([("C",), CC], arm="bpe")
        ul = _arm([("C",), CO])
        with pytest.raises(ValueError, match="boundaries must match"):
            compute_matched_pair_closure(
                bpe,
                ul,
                pair_key="p",
                tier="headline",
                corpus="pubchem",
                vocab_size=256,
                boundary="mb",  # arms are nmb
            )

    def test_rejects_second_arm_not_unigram(self) -> None:
        bpe = _arm([("C",), CC], arm="bpe")
        other = _arm([("C",), CC], arm="bpe")  # both BPE
        with pytest.raises(ValueError, match="second arg must be the Unigram arm"):
            compute_matched_pair_closure(
                bpe,
                other,  # type: ignore[arg-type]
                pair_key="p",
                tier="headline",
                corpus="pubchem",
                vocab_size=256,
                boundary="nmb",
            )

    def test_gap_is_nan_when_an_arm_is_undefined(self) -> None:
        bpe = _arm([("C",), CC, CCC], arm="bpe")
        ul = _arm([("C",), ("O",)])  # no multi pieces -> c_bin is nan
        pair = compute_matched_pair_closure(
            bpe,
            ul,
            pair_key="p",
            tier="headline",
            corpus="pubchem",
            vocab_size=256,
            boundary="nmb",
        )
        assert pair.delta_c_bin != pair.delta_c_bin  # nan propagates


class TestComputeUnpairedClosure:
    def test_carries_present_arm_reading(self) -> None:
        present = _arm([("C",), CC, CCC], arm="bpe")
        rec = compute_unpaired_closure(
            present,
            pair_key="zinc22__v2048_nmb",
            tier="conditional",
            corpus="zinc22",
            vocab_size=2048,
            boundary="nmb",
            extras_kind=None,
            extras_label=None,
            present_arm="bpe",
            missing_arm="unigram",
            unpaired_reason="conditional_negative_branch",
        )
        assert rec.pair_status == "single_arm"
        assert rec.present.c_bin == 1.0
        assert rec.as_dict()["bpe"]["c_bin"] == 1.0

    def test_rejects_missing_equals_present(self) -> None:
        present = _arm([("C",), CC], arm="bpe")
        with pytest.raises(ValueError, match="cannot equal present_arm"):
            compute_unpaired_closure(
                present,
                pair_key="p",
                tier="conditional",
                corpus="zinc22",
                vocab_size=2048,
                boundary="nmb",
                extras_kind=None,
                extras_label=None,
                present_arm="bpe",
                missing_arm="bpe",
                unpaired_reason="conditional_negative_branch",
            )
