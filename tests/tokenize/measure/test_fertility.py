"""Tests for ``smiles_subword.tokenize.measure.fertility`` (pure computation)."""

from __future__ import annotations

import pytest

from smiles_subword.tokenize.measure.fertility import (
    ArmFertility,
    PerMoleculeFertility,
    bootstrap_seed,
    compute_arm_fertility,
    compute_matched_pair_fertility,
    compute_unpaired_fertility,
)


def _pm(n_tokens: int, n_glyphs: int) -> PerMoleculeFertility:
    return PerMoleculeFertility(n_tokens=n_tokens, n_glyphs=n_glyphs)


class TestComputeArmFertility:
    def test_aggregates_fertility_and_glyphs_per_token(self) -> None:
        per_mol = [_pm(4, 10), _pm(6, 14)]

        arm = compute_arm_fertility(
            per_mol,
            cell_id="pubchem__smirk_gpe_v256_nmb",
            arm="bpe",
            boundary="nmb",
            training_corpus_sha="sha-A",
            eval_split_sha="eval-A",
        )

        assert arm.n_molecules == 2
        assert arm.total_tokens == 10
        assert arm.total_glyphs == 24
        assert arm.fertility_mean == pytest.approx(5.0)
        assert arm.glyphs_per_token_mean == pytest.approx(24 / 10)

    def test_across_molecule_variance_is_population_variance(self) -> None:
        per_mol = [_pm(2, 2), _pm(4, 4), _pm(6, 6)]

        arm = compute_arm_fertility(
            per_mol,
            cell_id="x",
            arm="bpe",
            boundary="nmb",
            training_corpus_sha="sha-A",
            eval_split_sha="eval-A",
        )

        assert arm.tokens_per_molecule_variance == pytest.approx(8 / 3)

    def test_bootstrap_ci_is_deterministic_for_same_cell(self) -> None:
        per_mol = [_pm(3 + (i % 4), 8 + (i % 5)) for i in range(40)]

        a = compute_arm_fertility(
            per_mol,
            cell_id="pubchem__smirk_gpe_v256_nmb",
            arm="bpe",
            boundary="nmb",
            training_corpus_sha="sha-A",
            eval_split_sha="eval-A",
        )
        b = compute_arm_fertility(
            per_mol,
            cell_id="pubchem__smirk_gpe_v256_nmb",
            arm="bpe",
            boundary="nmb",
            training_corpus_sha="sha-A",
            eval_split_sha="eval-A",
        )

        assert a.fertility_ci == b.fertility_ci
        assert a.glyphs_per_token_ci == b.glyphs_per_token_ci
        assert a.bootstrap_seed == b.bootstrap_seed

    def test_identical_molecules_collapse_each_ci_to_its_own_point(self) -> None:
        # With every held-out molecule identical, any molecule resample reproduces
        # the same statistic, so each CI must collapse to *its* point estimate.
        # fertility is a mean (unit denominators), glyphs/token a ratio (token
        # denominators); a swapped denominator would bracket the wrong quantity.
        per_mol = [_pm(5, 12)] * 6

        arm = compute_arm_fertility(
            per_mol,
            cell_id="x",
            arm="bpe",
            boundary="nmb",
            training_corpus_sha="sha-A",
            eval_split_sha="eval-A",
            n_resamples=300,
        )

        assert arm.fertility_mean == pytest.approx(5.0)
        assert arm.fertility_ci == pytest.approx((5.0, 5.0))
        assert arm.glyphs_per_token_mean == pytest.approx(2.4)
        assert arm.glyphs_per_token_ci == pytest.approx((2.4, 2.4))

    def test_fertility_and_glyph_ci_use_distinct_streams(self) -> None:
        per_mol = [_pm(3 + (i % 4), 8 + (i % 5)) for i in range(40)]

        arm = compute_arm_fertility(
            per_mol,
            cell_id="x",
            arm="bpe",
            boundary="nmb",
            training_corpus_sha="sha-A",
            eval_split_sha="eval-A",
        )

        assert arm.fertility_ci != arm.glyphs_per_token_ci

    def test_distinct_cells_have_distinct_seeds(self) -> None:
        assert bootstrap_seed("pubchem__smirk_gpe_v256_nmb") != bootstrap_seed(
            "pubchem__smirk_unigram_v256_nmb"
        )

    def test_zero_molecules_yields_nan_fertility(self) -> None:
        arm = compute_arm_fertility(
            [],
            cell_id="x",
            arm="bpe",
            boundary="nmb",
            training_corpus_sha="sha-A",
            eval_split_sha="eval-A",
        )

        assert arm.fertility_mean != arm.fertility_mean
        assert arm.glyphs_per_token_mean != arm.glyphs_per_token_mean

    def test_zero_tokens_yields_nan_glyphs_per_token(self) -> None:
        arm = compute_arm_fertility(
            [_pm(0, 0)],
            cell_id="x",
            arm="bpe",
            boundary="nmb",
            training_corpus_sha="sha-A",
            eval_split_sha="eval-A",
        )

        assert arm.glyphs_per_token_mean != arm.glyphs_per_token_mean


def _arm(
    *,
    arm: str,
    cell_id: str,
    boundary: str = "nmb",
    fertility: float = 5.0,
    glyphs_per_token: float = 2.0,
    total_glyphs: int = 1000,
) -> ArmFertility:
    return ArmFertility(
        cell_id=cell_id,
        arm=arm,  # type: ignore[arg-type]
        boundary=boundary,  # type: ignore[arg-type]
        n_molecules=100,
        total_tokens=int(fertility * 100),
        total_glyphs=total_glyphs,
        fertility_mean=fertility,
        fertility_ci=(fertility - 0.1, fertility + 0.1),
        glyphs_per_token_mean=glyphs_per_token,
        glyphs_per_token_ci=(glyphs_per_token - 0.05, glyphs_per_token + 0.05),
        tokens_per_molecule_variance=1.5,
        training_corpus_sha="sha-A",
        eval_split_sha="eval-A",
        bootstrap_seed=42,
        n_resamples=1000,
    )


class TestMatchedPair:
    def test_delta_fertility_is_bpe_minus_unigram(self) -> None:
        bpe = _arm(arm="bpe", cell_id="pubchem__smirk_gpe_v256_nmb", fertility=4.0)
        unigram = _arm(
            arm="unigram", cell_id="pubchem__smirk_unigram_v256_nmb", fertility=5.0
        )

        pair = compute_matched_pair_fertility(
            bpe,
            unigram,
            pair_key="pubchem__v256_nmb",
            tier="headline",
            corpus="pubchem",
            vocab_size=256,
            boundary="nmb",
        )

        assert pair.delta_fertility == pytest.approx(-1.0)
        assert pair.delta_fertility_relative == pytest.approx(1.0 / 4.5)

    def test_delta_glyphs_per_token_recorded(self) -> None:
        bpe = _arm(arm="bpe", cell_id="a", glyphs_per_token=2.4)
        unigram = _arm(arm="unigram", cell_id="b", glyphs_per_token=1.9)

        pair = compute_matched_pair_fertility(
            bpe,
            unigram,
            pair_key="pk",
            tier="headline",
            corpus="pubchem",
            vocab_size=256,
            boundary="nmb",
        )

        assert pair.delta_glyphs_per_token == pytest.approx(0.5)

    def test_equal_total_glyphs_is_consistent(self) -> None:
        bpe = _arm(arm="bpe", cell_id="a", total_glyphs=1000)
        unigram = _arm(arm="unigram", cell_id="b", total_glyphs=1000)

        pair = compute_matched_pair_fertility(
            bpe,
            unigram,
            pair_key="pk",
            tier="headline",
            corpus="pubchem",
            vocab_size=256,
            boundary="nmb",
        )

        assert pair.total_glyphs_consistent is True
        assert pair.total_glyphs_delta == 0

    def test_unequal_total_glyphs_is_flagged_not_raised(self) -> None:
        bpe = _arm(arm="bpe", cell_id="a", total_glyphs=1001)
        unigram = _arm(arm="unigram", cell_id="b", total_glyphs=1000)

        pair = compute_matched_pair_fertility(
            bpe,
            unigram,
            pair_key="pk",
            tier="headline",
            corpus="pubchem",
            vocab_size=256,
            boundary="nmb",
        )

        assert pair.total_glyphs_consistent is False
        assert pair.total_glyphs_delta == 1

    def test_arm_order_swap_raises(self) -> None:
        bpe = _arm(arm="bpe", cell_id="a")
        unigram = _arm(arm="unigram", cell_id="b")

        with pytest.raises(ValueError, match="first argument must be the BPE arm"):
            compute_matched_pair_fertility(
                unigram,  # type: ignore[arg-type]
                bpe,  # type: ignore[arg-type]
                pair_key="pk",
                tier="headline",
                corpus="pubchem",
                vocab_size=256,
                boundary="nmb",
            )

    def test_non_unigram_second_arg_raises(self) -> None:
        # The swap test above trips the bpe-side guard first; this pins its
        # sibling — a second argument that is not the Unigram arm.
        bpe = _arm(arm="bpe", cell_id="a")

        with pytest.raises(ValueError, match="must be the Unigram arm"):
            compute_matched_pair_fertility(
                bpe,
                _arm(arm="bpe", cell_id="b"),
                pair_key="pk",
                tier="headline",
                corpus="pubchem",
                vocab_size=256,
                boundary="nmb",
            )

    def test_boundary_mismatch_raises(self) -> None:
        bpe = _arm(arm="bpe", cell_id="a", boundary="nmb")
        unigram = _arm(arm="unigram", cell_id="b", boundary="mb")

        with pytest.raises(ValueError, match="boundaries must match"):
            compute_matched_pair_fertility(
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
        arm = _arm(arm="bpe", cell_id="zinc22__smirk_gpe_v2048_nmb")

        rec = compute_unpaired_fertility(
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
        arm = _arm(arm="bpe", cell_id="x", boundary="mb")

        with pytest.raises(ValueError, match="disagrees with pair"):
            compute_unpaired_fertility(
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
        arm = _arm(arm="bpe", cell_id="x")

        with pytest.raises(ValueError, match="cannot equal the present arm"):
            compute_unpaired_fertility(
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
