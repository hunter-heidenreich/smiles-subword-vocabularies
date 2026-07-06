"""Tests for ``smiles_subword.tokenize.measure.jaccard`` (pure Jaccard math)."""

from __future__ import annotations

import numpy as np
import pytest

from smiles_subword.tokenize.measure.jaccard import (
    ArmJaccardInputs,
    GlyphTuple,
    JwMoleculeData,
    MatchedPairJaccard,
    compute_matched_pair_jaccard,
    compute_unpaired_jaccard,
    jaccard,
    normalized_weights,
    weighted_jaccard,
)


def _jw(
    n_molecules: int,
    entries: list[tuple[int, int, float]],
    local_tuples: list[GlyphTuple],
) -> JwMoleculeData:
    return JwMoleculeData(
        n_molecules=n_molecules,
        mol_idx=np.asarray([e[0] for e in entries], dtype=np.int64),
        sub_local=np.asarray([e[1] for e in entries], dtype=np.int64),
        count=np.asarray([e[2] for e in entries], dtype=np.float64),
        local_tuples=tuple(local_tuples),
    )


def _inputs(
    arm: str,
    *,
    multi: set[GlyphTuple],
    structural: set[GlyphTuple],
    bracket_internal: set[GlyphTuple],
    unseen: set[GlyphTuple],
    jw: JwMoleculeData,
    boundary: str = "nmb",
) -> ArmJaccardInputs:
    return ArmJaccardInputs(
        cell_id=f"corpus__{arm}",
        arm=arm,  # type: ignore[arg-type]
        boundary=boundary,  # type: ignore[arg-type]
        training_corpus_sha="train-sha",
        eval_split_sha="eval-sha",
        multi_subwords=frozenset(multi),
        structural_subwords=frozenset(structural),
        bracket_internal_subwords=frozenset(bracket_internal),
        unseen_subwords=frozenset(unseen),
        n_distinct_bracket_chunks=1,
        n_distinct_nonbracket_chunks=1,
        nonbracket_cap_bound=False,
        jw=jw,
        bootstrap_seed=7,
    )


class TestJaccard:
    def test_both_empty_is_nan(self) -> None:
        assert jaccard(frozenset(), frozenset()) != jaccard(frozenset(), frozenset())

    def test_disjoint_sets_are_zero(self) -> None:
        assert jaccard(frozenset({("a",)}), frozenset({("b",)})) == 0.0

    def test_identical_sets_are_one(self) -> None:
        s = frozenset({("a", "b"), ("c",)})
        assert jaccard(s, s) == 1.0

    def test_partial_overlap(self) -> None:
        a = frozenset({("x",), ("y",)})
        b = frozenset({("y",), ("z",)})
        assert jaccard(a, b) == pytest.approx(1 / 3)


class TestNormalizedWeights:
    def test_weights_sum_to_one(self) -> None:
        out = normalized_weights({("a",): 3.0, ("b",): 1.0})
        assert out[("a",)] == pytest.approx(0.75)
        assert sum(out.values()) == pytest.approx(1.0)

    def test_empty_counts_yield_empty(self) -> None:
        assert normalized_weights({}) == {}


class TestWeightedJaccard:
    def test_known_example(self) -> None:
        w_a = {("C", "C"): 0.75, ("c", "c"): 0.25}
        w_b = {("C", "C"): 2 / 3, ("N", "N"): 1 / 3}
        assert weighted_jaccard(w_a, w_b) == pytest.approx(0.5)

    def test_identical_distributions_are_one(self) -> None:
        w = {("C", "C"): 0.6, ("c", "c"): 0.4}
        assert weighted_jaccard(w, w) == pytest.approx(1.0)

    def test_disjoint_distributions_are_zero(self) -> None:
        assert weighted_jaccard({("a",): 1.0}, {("b",): 1.0}) == 0.0

    def test_both_empty_is_nan(self) -> None:
        out = weighted_jaccard({}, {})
        assert out != out


class TestComputeMatchedPair:
    def _pair(self) -> MatchedPairJaccard:
        a_multi = {("C", "C"), ("c", "c")}
        b_multi = {("C", "C"), ("N", "N")}
        bpe = _inputs(
            "bpe",
            multi=a_multi,
            structural={("C", "C"), ("c", "c")},
            bracket_internal=set(),
            unseen=set(),
            jw=_jw(
                3, [(0, 0, 2.0), (1, 0, 1.0), (2, 1, 1.0)], [("C", "C"), ("c", "c")]
            ),
        )
        unigram = _inputs(
            "unigram",
            multi=b_multi,
            structural={("C", "C")},
            bracket_internal={("N", "N")},
            unseen=set(),
            jw=_jw(
                3, [(0, 0, 1.0), (1, 1, 1.0), (2, 0, 1.0)], [("C", "C"), ("N", "N")]
            ),
        )
        return compute_matched_pair_jaccard(
            bpe,
            unigram,
            pair_key="corpus__v256_nmb",
            tier="headline",
            corpus="corpus",
            vocab_size=256,
            boundary="nmb",
            n_resamples=200,
        )

    def test_unweighted_jaccard_over_multi_sets(self) -> None:
        assert self._pair().jaccard == pytest.approx(1 / 3)

    def test_structural_jaccard_over_structural_sets(self) -> None:
        assert self._pair().jaccard_struct == pytest.approx(0.5)

    def test_gap_is_j_minus_jstruct(self) -> None:
        pair = self._pair()
        assert pair.jaccard_minus_struct == pytest.approx(pair.jaccard - 0.5)

    def test_weighted_jaccard_point_estimate(self) -> None:
        assert self._pair().weighted_jaccard == pytest.approx(0.5)

    def test_weighted_jaccard_ci_brackets_point(self) -> None:
        pair = self._pair()
        lo, hi = pair.weighted_jaccard_ci
        assert lo <= pair.weighted_jaccard <= hi

    def test_weighted_jaccard_struct_renormalizes_over_structural_mass(self) -> None:
        assert self._pair().weighted_jaccard_struct == pytest.approx(0.6)

    def test_weighted_jaccard_struct_ci_brackets_point(self) -> None:
        pair = self._pair()
        lo, hi = pair.weighted_jaccard_struct_ci
        assert lo <= pair.weighted_jaccard_struct <= hi

    def test_wrong_bpe_arm_tag_raises(self) -> None:
        jw = _jw(1, [(0, 0, 1.0)], [("C", "C")])
        a = _inputs(
            "unigram",
            multi=set(),
            structural=set(),
            bracket_internal=set(),
            unseen=set(),
            jw=jw,
        )
        b = _inputs(
            "unigram",
            multi=set(),
            structural=set(),
            bracket_internal=set(),
            unseen=set(),
            jw=jw,
        )
        with pytest.raises(ValueError, match="must be the BPE arm"):
            compute_matched_pair_jaccard(
                a, b, pair_key="p", tier="t", corpus="c", vocab_size=1, boundary="nmb"
            )

    def test_boundary_mismatch_raises(self) -> None:
        jw = _jw(1, [(0, 0, 1.0)], [("C", "C")])
        a = _inputs(
            "bpe",
            multi=set(),
            structural=set(),
            bracket_internal=set(),
            unseen=set(),
            jw=jw,
            boundary="mb",
        )
        b = _inputs(
            "unigram",
            multi=set(),
            structural=set(),
            bracket_internal=set(),
            unseen=set(),
            jw=jw,
            boundary="nmb",
        )
        with pytest.raises(ValueError, match="boundaries must match"):
            compute_matched_pair_jaccard(
                a, b, pair_key="p", tier="t", corpus="c", vocab_size=1, boundary="nmb"
            )


class TestWeightedJaccardStruct:
    def _pair(
        self,
        *,
        bpe_structural: set[GlyphTuple],
        ul_structural: set[GlyphTuple],
    ) -> MatchedPairJaccard:
        bpe = _inputs(
            "bpe",
            multi={("C", "C"), ("c", "c")},
            structural=bpe_structural,
            bracket_internal={("C", "C"), ("c", "c")} - bpe_structural,
            unseen=set(),
            jw=_jw(
                3, [(0, 0, 2.0), (1, 0, 1.0), (2, 1, 1.0)], [("C", "C"), ("c", "c")]
            ),
        )
        unigram = _inputs(
            "unigram",
            multi={("C", "C"), ("N", "N")},
            structural=ul_structural,
            bracket_internal={("C", "C"), ("N", "N")} - ul_structural,
            unseen=set(),
            jw=_jw(
                3, [(0, 0, 1.0), (1, 1, 1.0), (2, 0, 1.0)], [("C", "C"), ("N", "N")]
            ),
        )
        return compute_matched_pair_jaccard(
            bpe,
            unigram,
            pair_key="corpus__v256_nmb",
            tier="headline",
            corpus="corpus",
            vocab_size=256,
            boundary="nmb",
            n_resamples=200,
        )

    def test_equals_weighted_jaccard_when_nothing_masked(self) -> None:
        pair = self._pair(
            bpe_structural={("C", "C"), ("c", "c")},
            ul_structural={("C", "C"), ("N", "N")},
        )
        assert pair.weighted_jaccard_struct == pytest.approx(pair.weighted_jaccard)

    def test_masking_unigram_bracket_internal_shifts_estimate(self) -> None:
        pair = self._pair(
            bpe_structural={("C", "C"), ("c", "c")},
            ul_structural={("C", "C")},
        )
        assert pair.weighted_jaccard == pytest.approx(0.5)
        assert pair.weighted_jaccard_struct == pytest.approx(0.6)

    def test_nan_when_both_structural_sets_empty(self) -> None:
        pair = self._pair(bpe_structural=set(), ul_structural=set())
        assert pair.weighted_jaccard_struct != pair.weighted_jaccard_struct

    def test_struct_ci_is_deterministic_across_reruns(self) -> None:
        kw = {"bpe_structural": {("C", "C")}, "ul_structural": {("C", "C")}}
        assert (
            self._pair(**kw).weighted_jaccard_struct_ci  # type: ignore[arg-type]
            == self._pair(**kw).weighted_jaccard_struct_ci  # type: ignore[arg-type]
        )


class TestBootstrapWeightedJaccard:
    """The J_w bootstrap CI computes J_w over *molecule* resamples, shared across
    arms — not row resamples, and recomputing the same J_w the analytic path does.
    """

    def test_identical_molecules_collapse_ci_to_the_analytic_point(self) -> None:
        # Every held-out molecule emits the same multiset, so any molecule
        # resample reproduces the same per-arm proportions: the CI must collapse
        # to exactly the analytic J_w. A resample over subword rows (rather than
        # molecules) would jitter instead, so this pins the resample *unit* and
        # cross-validates the numpy bootstrap against the dict-based J_w.
        bpe = _inputs(
            "bpe",
            multi={("C", "C"), ("c", "c")},
            structural={("C", "C"), ("c", "c")},
            bracket_internal=set(),
            unseen=set(),
            jw=_jw(
                3,
                [
                    (0, 0, 1.0),
                    (0, 1, 1.0),
                    (1, 0, 1.0),
                    (1, 1, 1.0),
                    (2, 0, 1.0),
                    (2, 1, 1.0),
                ],
                [("C", "C"), ("c", "c")],
            ),
        )
        unigram = _inputs(
            "unigram",
            multi={("C", "C"), ("N", "N")},
            structural={("C", "C"), ("N", "N")},
            bracket_internal=set(),
            unseen=set(),
            jw=_jw(
                3,
                [
                    (0, 0, 1.0),
                    (0, 1, 1.0),
                    (1, 0, 1.0),
                    (1, 1, 1.0),
                    (2, 0, 1.0),
                    (2, 1, 1.0),
                ],
                [("C", "C"), ("N", "N")],
            ),
        )

        pair = compute_matched_pair_jaccard(
            bpe,
            unigram,
            pair_key="corpus__v256_nmb",
            tier="headline",
            corpus="corpus",
            vocab_size=256,
            boundary="nmb",
            n_resamples=300,
        )

        lo, hi = pair.weighted_jaccard_ci
        assert pair.weighted_jaccard == pytest.approx(1 / 3)
        assert lo == pytest.approx(pair.weighted_jaccard)
        assert hi == pytest.approx(pair.weighted_jaccard)

    def test_arms_disagreeing_on_molecule_count_raises(self) -> None:
        # The shared-resample contract: both arms must encode the same held-out
        # molecule set, so a molecule-count mismatch is rejected, not averaged.
        bpe = _inputs(
            "bpe",
            multi={("C", "C")},
            structural={("C", "C")},
            bracket_internal=set(),
            unseen=set(),
            jw=_jw(3, [(0, 0, 1.0)], [("C", "C")]),
        )
        unigram = _inputs(
            "unigram",
            multi={("C", "C")},
            structural={("C", "C")},
            bracket_internal=set(),
            unseen=set(),
            jw=_jw(4, [(0, 0, 1.0)], [("C", "C")]),
        )

        with pytest.raises(ValueError, match="disagree on held-out molecule count"):
            compute_matched_pair_jaccard(
                bpe,
                unigram,
                pair_key="corpus__v256_nmb",
                tier="headline",
                corpus="corpus",
                vocab_size=256,
                boundary="nmb",
                n_resamples=50,
            )


class TestBootstrapDeterminism:
    def test_same_pair_key_reproduces_ci(self) -> None:
        jw_a = _jw(4, [(0, 0, 1.0), (1, 0, 1.0), (2, 1, 1.0)], [("C", "C"), ("c", "c")])
        jw_b = _jw(4, [(0, 0, 1.0), (2, 1, 1.0), (3, 0, 1.0)], [("C", "C"), ("N", "N")])
        kw = {
            "pair_key": "corpus__v512_nmb",
            "tier": "headline",
            "corpus": "corpus",
            "vocab_size": 512,
            "boundary": "nmb",
            "n_resamples": 300,
        }
        a1 = _inputs(
            "bpe",
            multi={("C", "C")},
            structural={("C", "C")},
            bracket_internal=set(),
            unseen=set(),
            jw=jw_a,
        )
        b1 = _inputs(
            "unigram",
            multi={("C", "C")},
            structural={("C", "C")},
            bracket_internal=set(),
            unseen=set(),
            jw=jw_b,
        )
        p1 = compute_matched_pair_jaccard(a1, b1, **kw)  # type: ignore[arg-type]
        p2 = compute_matched_pair_jaccard(a1, b1, **kw)  # type: ignore[arg-type]
        assert p1.weighted_jaccard_ci == p2.weighted_jaccard_ci


class TestComputeUnpaired:
    def test_present_arm_record_and_null_jaccards(self) -> None:
        jw = _jw(2, [(0, 0, 1.0)], [("C", "C")])
        arm = _inputs(
            "bpe",
            multi={("C", "C")},
            structural={("C", "C")},
            bracket_internal=set(),
            unseen=set(),
            jw=jw,
        )
        rec = compute_unpaired_jaccard(
            arm,
            pair_key="corpus__v2048_nmb",
            tier="headline",
            corpus="corpus",
            vocab_size=2048,
            boundary="nmb",
            extras_kind=None,
            extras_label=None,
            missing_arm="unigram",
            unpaired_reason="conditional_negative_branch",
        )
        d = rec.as_dict()
        assert d["pair_status"] == "single_arm"
        assert d["bpe"] is not None
        assert d["unigram"] is None
        assert d["jaccard"] is None
        assert d["weighted_jaccard_ci"] is None
        assert d["missing_arm"] == "unigram"

    def test_missing_arm_equal_present_raises(self) -> None:
        jw = _jw(1, [(0, 0, 1.0)], [("C", "C")])
        arm = _inputs(
            "bpe",
            multi=set(),
            structural=set(),
            bracket_internal=set(),
            unseen=set(),
            jw=jw,
        )
        with pytest.raises(ValueError, match="cannot equal the present arm"):
            compute_unpaired_jaccard(
                arm,
                pair_key="p",
                tier="t",
                corpus="c",
                vocab_size=1,
                boundary="nmb",
                extras_kind=None,
                extras_label=None,
                missing_arm="bpe",
                unpaired_reason="conditional_negative_branch",
            )


class TestArmRecordDict:
    def test_matched_as_dict_carries_all_metrics(self) -> None:
        jw = _jw(2, [(0, 0, 1.0)], [("C", "C")])
        bpe = _inputs(
            "bpe",
            multi={("C", "C")},
            structural={("C", "C")},
            bracket_internal=set(),
            unseen=set(),
            jw=jw,
        )
        ul = _inputs(
            "unigram",
            multi={("C", "C")},
            structural={("C", "C")},
            bracket_internal=set(),
            unseen=set(),
            jw=jw,
        )
        d = compute_matched_pair_jaccard(
            bpe,
            ul,
            pair_key="p",
            tier="t",
            corpus="c",
            vocab_size=1,
            boundary="nmb",
            n_resamples=20,
        ).as_dict()
        for key in (
            "jaccard",
            "jaccard_struct",
            "weighted_jaccard",
            "weighted_jaccard_ci",
            "weighted_jaccard_struct",
            "weighted_jaccard_struct_ci",
        ):
            assert key in d
        assert d["bpe"]["n_multi_subwords"] == 1  # type: ignore[index]
