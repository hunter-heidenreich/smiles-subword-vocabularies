"""Tests for ``smiles_subword.tokenize.measure.absorption`` (pure computation)."""

from __future__ import annotations

import pytest

from smiles_subword.tokenize.measure.absorption import (
    ArmAbsorption,
    PerMoleculeAbsorption,
    bootstrap_seed,
    classify_chunks,
    compute_arm_absorption,
    compute_matched_pair_absorption,
    compute_unpaired_absorption,
)


def _per_molecule(
    n_chunks: int, n_absorbed: int, n_cross_chunk: int | None
) -> PerMoleculeAbsorption:
    return PerMoleculeAbsorption(
        n_chunks=n_chunks, n_absorbed=n_absorbed, n_cross_chunk=n_cross_chunk
    )


class TestClassifyChunks:
    def test_chunk_with_matching_token_offsets_is_absorbed(self) -> None:
        chunks = [("CC", (0, 2)), ("(", (2, 3)), ("=O", (3, 5)), (")", (5, 6))]
        tokens = [(0, 2), (2, 3), (3, 5), (5, 6)]

        result = classify_chunks(chunks, tokens, boundary="nmb")

        assert result == _per_molecule(4, 4, None)

    def test_chunk_split_across_tokens_under_nmb_is_neither_absorbed_nor_xchunk(
        self,
    ) -> None:
        chunks = [("CC", (0, 2))]
        tokens = [(0, 1), (1, 2)]

        result = classify_chunks(chunks, tokens, boundary="nmb")

        assert result == _per_molecule(1, 0, None)

    def test_chunk_strictly_inside_larger_token_under_mb_is_cross_chunk(self) -> None:
        chunks = [("C", (0, 1)), ("(", (1, 2)), ("C", (2, 3)), (")", (3, 4))]
        tokens = [(0, 4)]

        result = classify_chunks(chunks, tokens, boundary="mb")

        assert result == _per_molecule(4, 0, 4)

    def test_mb_counts_absorbed_separately_from_cross_chunk(self) -> None:
        chunks = [("CC", (0, 2)), ("(", (2, 3)), ("=O", (3, 5))]
        tokens = [(0, 2), (2, 5)]

        result = classify_chunks(chunks, tokens, boundary="mb")

        assert result == _per_molecule(3, 1, 2)

    def test_mb_chunk_straddled_by_tokens_is_neither(self) -> None:
        # A chunk only *partially* covered (token boundaries fall inside it) is
        # not strictly contained by any token, so it is neither absorbed nor
        # cross-chunk: cross-chunk means strict containment, not any overlap.
        chunks = [("CC", (1, 3))]
        tokens = [(0, 2), (2, 4)]  # neither contains (1, 3)

        result = classify_chunks(chunks, tokens, boundary="mb")

        assert result == _per_molecule(1, 0, 0)

    def test_nmb_never_emits_cross_chunk(self) -> None:
        chunks = [("CC", (0, 2)), ("(", (2, 3))]
        tokens = [(0, 3)]

        result = classify_chunks(chunks, tokens, boundary="nmb")

        assert result.n_cross_chunk is None

    def test_empty_chunks_yields_zero_counts(self) -> None:
        result = classify_chunks([], [(0, 1)], boundary="mb")

        assert result == _per_molecule(0, 0, 0)


class TestComputeArmAbsorption:
    def test_nmb_aggregates_chunks_and_omits_cross_chunk(self) -> None:
        per_mol = [_per_molecule(10, 7, None), _per_molecule(8, 6, None)]

        arm = compute_arm_absorption(
            per_mol,
            cell_id="pubchem__smirk_gpe_v256_nmb",
            arm="bpe",
            boundary="nmb",
            training_corpus_sha="sha-A",
            eval_split_sha="eval-A",
        )

        assert arm.n_chunks == 18
        assert arm.n_absorbed == 13
        assert arm.absorbed_fraction == pytest.approx(13 / 18)
        assert arm.n_cross_chunk_total is None
        assert arm.cross_chunk_fraction is None
        assert arm.cross_chunk_ci is None

    def test_mb_reports_both_absorbed_and_cross_chunk(self) -> None:
        per_mol = [_per_molecule(10, 4, 3), _per_molecule(6, 2, 2)]

        arm = compute_arm_absorption(
            per_mol,
            cell_id="pubchem__smirk_gpe_v256_mb",
            arm="bpe",
            boundary="mb",
            training_corpus_sha="sha-A",
            eval_split_sha="eval-A",
        )

        assert arm.absorbed_fraction == pytest.approx(6 / 16)
        assert arm.cross_chunk_fraction == pytest.approx(5 / 16)
        assert arm.cross_chunk_ci is not None

    def test_identical_molecules_collapse_both_cis_to_their_points(self) -> None:
        # Identical molecules: any molecule resample reproduces the same chunk
        # fractions, so each CI collapses to its point. Pins the resample unit
        # (molecules) and that both CIs use n_chunks as the denominator.
        per_mol = [_per_molecule(4, 1, 2)] * 5

        arm = compute_arm_absorption(
            per_mol,
            cell_id="x",
            arm="bpe",
            boundary="mb",
            training_corpus_sha="sha-A",
            eval_split_sha="eval-A",
            n_resamples=300,
        )

        assert arm.absorbed_fraction == pytest.approx(0.25)
        assert arm.absorbed_ci == pytest.approx((0.25, 0.25))
        assert arm.cross_chunk_fraction == pytest.approx(0.5)
        assert arm.cross_chunk_ci == pytest.approx((0.5, 0.5))

    def test_mb_zero_chunks_yields_nan_cross_fraction(self) -> None:
        arm = compute_arm_absorption(
            [_per_molecule(0, 0, 0)],
            cell_id="x",
            arm="bpe",
            boundary="mb",
            training_corpus_sha="sha-A",
            eval_split_sha="eval-A",
        )

        assert arm.absorbed_fraction != arm.absorbed_fraction
        assert arm.cross_chunk_fraction != arm.cross_chunk_fraction

    def test_bootstrap_seed_is_deterministic_for_same_cell(self) -> None:
        per_mol = [_per_molecule(10, 5, None) for _ in range(20)]

        a = compute_arm_absorption(
            per_mol,
            cell_id="pubchem__smirk_gpe_v256_nmb",
            arm="bpe",
            boundary="nmb",
            training_corpus_sha="sha-A",
            eval_split_sha="eval-A",
        )
        b = compute_arm_absorption(
            per_mol,
            cell_id="pubchem__smirk_gpe_v256_nmb",
            arm="bpe",
            boundary="nmb",
            training_corpus_sha="sha-A",
            eval_split_sha="eval-A",
        )

        assert a.absorbed_ci == b.absorbed_ci
        assert a.bootstrap_seed == b.bootstrap_seed

    def test_distinct_cells_have_distinct_seeds(self) -> None:
        assert bootstrap_seed("pubchem__smirk_gpe_v256_nmb") != bootstrap_seed(
            "pubchem__smirk_unigram_v256_nmb"
        )

    def test_nmb_with_cross_chunk_int_raises(self) -> None:
        with pytest.raises(ValueError, match="NMB cells"):
            compute_arm_absorption(
                [_per_molecule(1, 1, 0)],
                cell_id="x",
                arm="bpe",
                boundary="nmb",
                training_corpus_sha="sha-A",
                eval_split_sha="eval-A",
            )

    def test_mb_with_missing_cross_chunk_raises(self) -> None:
        with pytest.raises(ValueError, match="MB cells require"):
            compute_arm_absorption(
                [_per_molecule(1, 1, None)],
                cell_id="x",
                arm="bpe",
                boundary="mb",
                training_corpus_sha="sha-A",
                eval_split_sha="eval-A",
            )

    def test_zero_chunks_yields_nan_fraction(self) -> None:
        arm = compute_arm_absorption(
            [_per_molecule(0, 0, None)],
            cell_id="x",
            arm="bpe",
            boundary="nmb",
            training_corpus_sha="sha-A",
            eval_split_sha="eval-A",
        )

        assert arm.absorbed_fraction != arm.absorbed_fraction


def _arm(
    *,
    arm: str,
    cell_id: str,
    boundary: str,
    absorbed: float,
    cross: float | None = None,
) -> ArmAbsorption:
    return ArmAbsorption(
        cell_id=cell_id,
        arm=arm,  # type: ignore[arg-type]
        boundary=boundary,  # type: ignore[arg-type]
        n_molecules=100,
        n_chunks=1000,
        n_absorbed=int(absorbed * 1000),
        n_cross_chunk_total=int((cross or 0.0) * 1000) if cross is not None else None,
        absorbed_fraction=absorbed,
        absorbed_ci=(absorbed - 0.01, absorbed + 0.01),
        cross_chunk_fraction=cross,
        cross_chunk_ci=(cross - 0.01, cross + 0.01) if cross is not None else None,
        training_corpus_sha="sha-A",
        eval_split_sha="eval-A",
        bootstrap_seed=42,
        n_resamples=1000,
    )


class TestMatchedPair:
    def test_delta_absorbed_is_bpe_minus_unigram(self) -> None:
        bpe = _arm(
            arm="bpe",
            cell_id="pubchem__smirk_gpe_v256_nmb",
            boundary="nmb",
            absorbed=0.80,
        )
        unigram = _arm(
            arm="unigram",
            cell_id="pubchem__smirk_unigram_v256_nmb",
            boundary="nmb",
            absorbed=0.65,
        )

        pair = compute_matched_pair_absorption(
            bpe,
            unigram,
            pair_key="pubchem__v256_nmb",
            tier="headline",
            corpus="pubchem",
            vocab_size=256,
            boundary="nmb",
        )

        assert pair.delta_absorbed == pytest.approx(0.15)
        assert pair.delta_cross_chunk is None

    def test_mb_pair_reports_delta_cross_chunk(self) -> None:
        bpe = _arm(
            arm="bpe",
            cell_id="pubchem__smirk_gpe_v256_mb",
            boundary="mb",
            absorbed=0.50,
            cross=0.20,
        )
        unigram = _arm(
            arm="unigram",
            cell_id="pubchem__smirk_unigram_v256_mb",
            boundary="mb",
            absorbed=0.40,
            cross=0.10,
        )

        pair = compute_matched_pair_absorption(
            bpe,
            unigram,
            pair_key="pubchem__v256_mb",
            tier="headline",
            corpus="pubchem",
            vocab_size=256,
            boundary="mb",
        )

        assert pair.delta_cross_chunk == pytest.approx(0.10)

    def test_arm_order_swap_raises(self) -> None:
        bpe = _arm(arm="bpe", cell_id="a", boundary="nmb", absorbed=0.80)
        unigram = _arm(arm="unigram", cell_id="b", boundary="nmb", absorbed=0.60)

        with pytest.raises(ValueError, match="first argument must be the BPE arm"):
            compute_matched_pair_absorption(
                unigram,  # type: ignore[arg-type]
                bpe,  # type: ignore[arg-type]
                pair_key="pk",
                tier="headline",
                corpus="pubchem",
                vocab_size=256,
                boundary="nmb",
            )

    def test_non_unigram_second_arg_raises(self) -> None:
        # The swap test trips the bpe-side guard first; this pins its sibling.
        bpe = _arm(arm="bpe", cell_id="a", boundary="nmb", absorbed=0.80)

        with pytest.raises(ValueError, match="must be the Unigram arm"):
            compute_matched_pair_absorption(
                bpe,
                _arm(arm="bpe", cell_id="b", boundary="nmb", absorbed=0.60),
                pair_key="pk",
                tier="headline",
                corpus="pubchem",
                vocab_size=256,
                boundary="nmb",
            )

    def test_boundary_mismatch_raises(self) -> None:
        bpe = _arm(arm="bpe", cell_id="a", boundary="nmb", absorbed=0.80)
        unigram = _arm(
            arm="unigram", cell_id="b", boundary="mb", absorbed=0.60, cross=0.1
        )

        with pytest.raises(ValueError, match="boundaries must match"):
            compute_matched_pair_absorption(
                bpe,
                unigram,
                pair_key="pk",
                tier="headline",
                corpus="pubchem",
                vocab_size=256,
                boundary="nmb",
            )


class TestUnpaired:
    def test_unpaired_records_present_arm(self) -> None:
        arm = _arm(
            arm="bpe",
            cell_id="zinc22__smirk_gpe_v2048_nmb",
            boundary="nmb",
            absorbed=0.70,
        )

        rec = compute_unpaired_absorption(
            arm,
            pair_key="zinc22__v2048_nmb",
            tier="conditional",
            corpus="zinc22",
            vocab_size=2048,
            boundary="nmb",
            extras_kind=None,
            extras_label=None,
            missing_arm="unigram",
            unpaired_reason="conditional_negative_branch",
        )

        assert rec.present_arm is arm
        assert rec.missing_arm == "unigram"
        assert rec.unpaired_reason == "conditional_negative_branch"

    def test_boundary_mismatch_raises(self) -> None:
        arm = _arm(arm="bpe", cell_id="x", boundary="mb", absorbed=0.5, cross=0.1)

        with pytest.raises(ValueError, match="disagrees with pair"):
            compute_unpaired_absorption(
                arm,
                pair_key="pk",
                tier="conditional",
                corpus="zinc22",
                vocab_size=2048,
                boundary="nmb",
                extras_kind=None,
                extras_label=None,
                missing_arm="unigram",
                unpaired_reason="conditional_negative_branch",
            )

    def test_missing_arm_equal_to_present_raises(self) -> None:
        arm = _arm(arm="bpe", cell_id="x", boundary="nmb", absorbed=0.5)

        with pytest.raises(ValueError, match="cannot equal the present arm"):
            compute_unpaired_absorption(
                arm,
                pair_key="pk",
                tier="conditional",
                corpus="zinc22",
                vocab_size=2048,
                boundary="nmb",
                extras_kind=None,
                extras_label=None,
                missing_arm="bpe",
                unpaired_reason="conditional_negative_branch",
            )
