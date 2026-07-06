"""Tests for ``smiles_subword.tokenize.measure.segmentation`` (pure computation).

The load-bearing test is :class:`TestChunkSegmentationEntropy` — the forward DP
is checked against an independent brute-force enumeration of every segmentation.
"""

from __future__ import annotations

import math
import random

import pytest

from smiles_subword.tokenize.measure.segmentation import (
    GlyphTuple,
    PerMoleculeSegmentation,
    chunk_segmentation_entropy,
    compute_arm_segmentation,
    compute_bpe_arm_segmentation,
    compute_matched_pair_segmentation,
    compute_unpaired_segmentation,
)


def _brute_force_entropy(
    glyphs: GlyphTuple, piece_scores: dict[GlyphTuple, float]
) -> float:
    """Reference: enumerate every segmentation, normalize, ``−Σ p log p``."""
    log_scores: list[float] = []

    def recurse(pos: int, acc: float) -> None:
        if pos == len(glyphs):
            log_scores.append(acc)
            return
        for end in range(pos + 1, len(glyphs) + 1):
            score = piece_scores.get(glyphs[pos:end])
            if score is not None:
                recurse(end, acc + score)

    recurse(0, 0.0)
    if not log_scores:
        return 0.0
    weights = [math.exp(s) for s in log_scores]
    total = math.fsum(weights)
    probs = [w / total for w in weights]
    return -math.fsum(p * math.log(p) for p in probs if p > 0)


class TestChunkSegmentationEntropy:
    def test_single_glyph_single_path_is_zero(self) -> None:
        scores = {("C",): -1.0}

        assert chunk_segmentation_entropy(("C",), scores) == 0.0

    def test_empty_chunk_is_zero(self) -> None:
        assert chunk_segmentation_entropy((), {("C",): -1.0}) == 0.0

    def test_only_singletons_path_is_zero(self) -> None:
        scores = {("C",): -1.0, ("O",): -2.0}

        assert chunk_segmentation_entropy(("C", "O", "C"), scores) == pytest.approx(0.0)

    def test_two_glyph_binary_entropy_matches_brute_force(self) -> None:
        scores = {("C",): -1.0, ("C", "C"): -1.5}

        dp = chunk_segmentation_entropy(("C", "C"), scores)

        assert dp == pytest.approx(_brute_force_entropy(("C", "C"), scores))
        assert dp > 0.0

    def test_equal_weights_give_ln2(self) -> None:
        scores = {("C",): math.log(0.5), ("C", "C"): math.log(0.25)}

        assert chunk_segmentation_entropy(("C", "C"), scores) == pytest.approx(
            math.log(2)
        )

    @pytest.mark.parametrize(
        "glyphs",
        [
            ("a", "b", "c"),
            ("a", "a", "a", "a"),
            ("a", "b", "a", "b"),
            ("a", "b", "c", "d", "e"),
        ],
    )
    def test_dp_matches_brute_force_on_dense_vocab(self, glyphs: GlyphTuple) -> None:
        alphabet = sorted(set(glyphs))
        scores: dict[GlyphTuple, float] = {
            (g,): -1.0 - i for i, g in enumerate(alphabet)
        }
        for i in range(len(glyphs)):
            for j in range(i + 2, len(glyphs) + 1):
                scores[glyphs[i:j]] = -0.5 * (j - i) - 0.1 * i

        assert chunk_segmentation_entropy(glyphs, scores) == pytest.approx(
            _brute_force_entropy(glyphs, scores)
        )

    def test_dp_matches_brute_force_on_random_vocabularies(self) -> None:
        rng = random.Random(20260522)
        alphabet = ["a", "b", "c"]
        for _ in range(200):
            glyphs = tuple(rng.choice(alphabet) for _ in range(rng.randint(1, 7)))
            scores: dict[GlyphTuple, float] = {
                (g,): rng.uniform(-4.0, -0.5) for g in alphabet
            }
            for i in range(len(glyphs)):
                for j in range(i + 2, len(glyphs) + 1):
                    if rng.random() < 0.6:
                        scores[glyphs[i:j]] = rng.uniform(-5.0, -0.2)

            assert chunk_segmentation_entropy(glyphs, scores) == pytest.approx(
                _brute_force_entropy(glyphs, scores)
            )

    def test_disconnected_lattice_yields_zero(self) -> None:
        # A glyph with no covering piece leaves the final node unreachable; the
        # DP degrades to 0.0 (matching brute-force, which finds no segmentation)
        # rather than crashing — the defensive path the connected lattice never
        # exercises in production.
        glyphs = ("a", "b")
        scores = {("a",): -1.0}  # no ("b",) and no ("a","b")

        assert chunk_segmentation_entropy(glyphs, scores) == 0.0
        assert _brute_force_entropy(glyphs, scores) == 0.0

    def test_max_piece_len_bound_matches_unbounded(self) -> None:
        scores = {
            ("a",): -1.0,
            ("a", "a"): -1.3,
            ("a", "a", "a"): -2.0,
        }
        glyphs = ("a", "a", "a", "a")

        bounded = chunk_segmentation_entropy(glyphs, scores, max_piece_len=3)
        unbounded = chunk_segmentation_entropy(glyphs, scores)

        assert bounded == pytest.approx(unbounded)


class TestComputeArmSegmentation:
    def test_aggregates_means_and_totals(self) -> None:
        per_mol = [PerMoleculeSegmentation(2.0, 10), PerMoleculeSegmentation(4.0, 30)]

        arm = compute_arm_segmentation(
            per_mol,
            cell_id="pubchem__smirk_unigram_v256_nmb",
            arm="unigram",
            boundary="nmb",
            training_corpus_sha="sha-A",
            eval_split_sha="eval-A",
        )

        assert arm.n_molecules == 2
        assert arm.total_glyphs == 40
        assert arm.total_entropy_nats == pytest.approx(6.0)
        assert arm.entropy_per_molecule_mean == pytest.approx(3.0)
        assert arm.entropy_per_glyph == pytest.approx(6.0 / 40)
        assert arm.verified_by_construction is False

    def test_identical_molecules_collapse_both_cis_to_their_points(self) -> None:
        # Every held-out molecule has the same entropy and glyph count, so any
        # molecule resample reproduces the same statistics: both CIs collapse to
        # their points. Pins the resample unit (molecules) and that per-molecule
        # is Σentropy/n while per-glyph is Σentropy/Σglyphs.
        per_mol = [PerMoleculeSegmentation(1.5, 8)] * 6

        arm = compute_arm_segmentation(
            per_mol,
            cell_id="x",
            arm="unigram",
            boundary="nmb",
            training_corpus_sha="s",
            eval_split_sha="e",
            n_resamples=300,
        )

        assert arm.entropy_per_molecule_mean == pytest.approx(1.5)
        assert arm.entropy_per_molecule_ci == pytest.approx((1.5, 1.5))
        assert arm.entropy_per_glyph == pytest.approx(1.5 / 8)
        assert arm.entropy_per_glyph_ci == pytest.approx((1.5 / 8, 1.5 / 8))

    def test_zero_molecules_yields_nan_cis(self) -> None:
        arm = compute_arm_segmentation(
            [],
            cell_id="x",
            arm="unigram",
            boundary="nmb",
            training_corpus_sha="s",
            eval_split_sha="e",
        )

        assert arm.n_molecules == 0
        assert arm.entropy_per_molecule_ci[0] != arm.entropy_per_molecule_ci[0]
        assert arm.entropy_per_glyph_ci[0] != arm.entropy_per_glyph_ci[0]

    def test_bootstrap_ci_is_deterministic_for_same_cell(self) -> None:
        per_mol = [
            PerMoleculeSegmentation(0.1 * (i % 7), 5 + (i % 4)) for i in range(60)
        ]

        a = compute_arm_segmentation(
            per_mol,
            cell_id="zinc22__smirk_unigram_v512_nmb",
            arm="unigram",
            boundary="nmb",
            training_corpus_sha="s",
            eval_split_sha="e",
        )
        b = compute_arm_segmentation(
            per_mol,
            cell_id="zinc22__smirk_unigram_v512_nmb",
            arm="unigram",
            boundary="nmb",
            training_corpus_sha="s",
            eval_split_sha="e",
        )

        assert a.entropy_per_molecule_ci == b.entropy_per_molecule_ci
        assert a.entropy_per_glyph_ci == b.entropy_per_glyph_ci


class TestComputeBpeArmSegmentation:
    def test_zero_by_construction(self) -> None:
        arm = compute_bpe_arm_segmentation(
            cell_id="pubchem__smirk_gpe_v256_nmb",
            boundary="nmb",
            training_corpus_sha="sha-A",
        )

        assert arm.arm == "bpe"
        assert arm.verified_by_construction is True
        assert arm.total_entropy_nats == 0.0
        assert arm.entropy_per_molecule_mean == 0.0
        assert arm.entropy_per_glyph == 0.0
        assert arm.eval_split_sha is None


class TestMatchedPairSegmentation:
    def _unigram_arm(self) -> object:
        return compute_arm_segmentation(
            [PerMoleculeSegmentation(3.0, 10), PerMoleculeSegmentation(1.0, 6)],
            cell_id="pubchem__smirk_unigram_v256_nmb",
            arm="unigram",
            boundary="nmb",
            training_corpus_sha="s",
            eval_split_sha="e",
        )

    def test_delta_equals_unigram_reading(self) -> None:
        bpe = compute_bpe_arm_segmentation(
            cell_id="pubchem__smirk_gpe_v256_nmb",
            boundary="nmb",
            training_corpus_sha="s",
        )
        unigram = self._unigram_arm()

        pair = compute_matched_pair_segmentation(
            bpe,
            unigram,  # type: ignore[arg-type]
            pair_key="pubchem__v256_nmb",
            tier="headline",
            corpus="pubchem",
            vocab_size=256,
            boundary="nmb",
        )

        assert pair.delta_entropy_per_molecule == pytest.approx(
            unigram.entropy_per_molecule_mean  # type: ignore[attr-defined]
        )
        assert pair.delta_entropy_per_glyph == pytest.approx(
            unigram.entropy_per_glyph  # type: ignore[attr-defined]
        )

    def test_rejects_swapped_arms(self) -> None:
        bpe = compute_bpe_arm_segmentation(
            cell_id="c", boundary="nmb", training_corpus_sha="s"
        )

        with pytest.raises(ValueError, match="must be the BPE arm"):
            compute_matched_pair_segmentation(
                self._unigram_arm(),  # type: ignore[arg-type]
                bpe,  # type: ignore[arg-type]
                pair_key="k",
                tier="headline",
                corpus="pubchem",
                vocab_size=256,
                boundary="nmb",
            )

    def test_rejects_non_unigram_second_arg(self) -> None:
        # The swap test trips the bpe-side guard first; this pins its sibling.
        bpe = compute_bpe_arm_segmentation(
            cell_id="c", boundary="nmb", training_corpus_sha="s"
        )

        with pytest.raises(ValueError, match="must be the Unigram arm"):
            compute_matched_pair_segmentation(
                bpe,
                compute_bpe_arm_segmentation(
                    cell_id="c2", boundary="nmb", training_corpus_sha="s"
                ),
                pair_key="k",
                tier="headline",
                corpus="pubchem",
                vocab_size=256,
                boundary="nmb",
            )

    def test_rejects_boundary_mismatch(self) -> None:
        bpe = compute_bpe_arm_segmentation(
            cell_id="c", boundary="mb", training_corpus_sha="s"
        )

        with pytest.raises(ValueError, match="boundaries must match"):
            compute_matched_pair_segmentation(
                bpe,
                self._unigram_arm(),  # type: ignore[arg-type]
                pair_key="k",
                tier="headline",
                corpus="pubchem",
                vocab_size=256,
                boundary="nmb",
            )


class TestUnpairedSegmentation:
    def test_wraps_present_arm(self) -> None:
        arm = compute_arm_segmentation(
            [PerMoleculeSegmentation(2.0, 8)],
            cell_id="pubchem__smirk_unigram_v1024_mb__seed_uncapped",
            arm="unigram",
            boundary="mb",
            training_corpus_sha="s",
            eval_split_sha="e",
        )

        rec = compute_unpaired_segmentation(
            arm,
            pair_key="pubchem__v1024_mb__seed_uncapped",
            tier="extras_seed_cap",
            corpus="pubchem",
            vocab_size=1024,
            boundary="mb",
            extras_kind="seed_cap",
            extras_label="uncapped",
            missing_arm="bpe",
            unpaired_reason="extras_single_arm_knob",
        )

        assert rec.pair_status == "single_arm"
        assert rec.missing_arm == "bpe"
        payload = rec.as_dict()
        assert payload["bpe"] is None
        assert payload["unigram"] is not None

    def test_rejects_boundary_mismatch(self) -> None:
        arm = compute_arm_segmentation(
            [PerMoleculeSegmentation(2.0, 8)],
            cell_id="c",
            arm="unigram",
            boundary="mb",
            training_corpus_sha="s",
            eval_split_sha="e",
        )

        with pytest.raises(ValueError, match="disagrees with pair"):
            compute_unpaired_segmentation(
                arm,
                pair_key="k",
                tier="t",
                corpus="zinc22",
                vocab_size=2048,
                boundary="nmb",
                extras_kind=None,
                extras_label=None,
                missing_arm="bpe",
                unpaired_reason="conditional_negative_branch",
            )

    def test_rejects_missing_arm_equal_to_present(self) -> None:
        arm = compute_bpe_arm_segmentation(
            cell_id="c", boundary="nmb", training_corpus_sha="s"
        )

        with pytest.raises(ValueError, match="cannot equal the present arm"):
            compute_unpaired_segmentation(
                arm,
                pair_key="k",
                tier="t",
                corpus="zinc22",
                vocab_size=2048,
                boundary="nmb",
                extras_kind=None,
                extras_label=None,
                missing_arm="bpe",
                unpaired_reason="conditional_negative_branch",
            )
