"""Tests for ``smiles_subword.tokenize.measure.distribution`` (pure computation)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from smiles_subword.tokenize.measure.distribution import (
    ArmDistribution,
    DistributionMoleculeData,
    bootstrap_seed,
    compute_arm_distribution,
    compute_matched_pair_distribution,
    compute_unpaired_distribution,
)


def _data(
    per_mol: list[dict[int, int]], *, v_effective: int
) -> DistributionMoleculeData:
    """Build sparse per-molecule data from per-molecule ``{token_id: count}``."""
    local_ids: dict[int, int] = {}
    local_token_ids: list[int] = []
    mol_idx: list[int] = []
    sub_local: list[int] = []
    count: list[int] = []
    for m, counts in enumerate(per_mol):
        for tid, c in counts.items():
            lid = local_ids.get(tid)
            if lid is None:
                lid = len(local_token_ids)
                local_ids[tid] = lid
                local_token_ids.append(tid)
            mol_idx.append(m)
            sub_local.append(lid)
            count.append(c)
    return DistributionMoleculeData(
        n_molecules=len(per_mol),
        mol_idx=np.asarray(mol_idx, dtype=np.int64),
        sub_local=np.asarray(sub_local, dtype=np.int64),
        count=np.asarray(count, dtype=np.float64),
        local_token_ids=tuple(local_token_ids),
        v_effective=v_effective,
    )


class TestComputeArmDistribution:
    def test_uniform_distribution_has_known_intrinsics(self) -> None:
        data = _data(
            [{10: 1}, {11: 1}, {12: 1}, {13: 1}],
            v_effective=4,
        )

        arm = compute_arm_distribution(
            data,
            cell_id="pubchem__smirk_gpe_v256_nmb",
            arm="bpe",
            boundary="nmb",
            vocab_size=259,
            training_corpus_sha="sha-A",
            eval_split_sha="eval-A",
        )

        assert arm.d == pytest.approx(0.0)
        assert arm.eta == pytest.approx(1.0)
        assert arm.renyi == pytest.approx(1.0)
        assert arm.live_token_count == 4
        assert arm.total_tokens == 4
        assert arm.v_effective == 4
        assert arm.vocab_size == 259

    def test_live_token_count_below_v_effective_with_dead_glyphs(self) -> None:
        data = _data([{10: 5}, {10: 3}, {11: 2}], v_effective=50)

        arm = compute_arm_distribution(
            data,
            cell_id="zinc22__smirk_gpe_v256_nmb",
            arm="bpe",
            boundary="nmb",
            vocab_size=53,
            training_corpus_sha="sha-A",
            eval_split_sha="eval-A",
        )

        assert arm.live_token_count == 2
        assert arm.live_token_count < arm.v_effective

    def test_d_matches_token_imbalance_closed_form(self) -> None:
        data = _data([{10: 70}, {11: 30}], v_effective=10)

        arm = compute_arm_distribution(
            data,
            cell_id="x",
            arm="bpe",
            boundary="nmb",
            vocab_size=14,
            training_corpus_sha="sha-A",
            eval_split_sha="eval-A",
        )

        inv_v = 1.0 / 10
        live = abs(0.7 - inv_v) + abs(0.3 - inv_v)
        expected_d = 0.5 * (live + (10 - 2) * inv_v)
        assert arm.d == pytest.approx(expected_d)

    def test_bootstrap_ci_is_deterministic_for_same_cell(self) -> None:
        per_mol = [{10: 1 + (i % 3), 11: 1 + (i % 2), 12: i % 4} for i in range(40)]

        a = compute_arm_distribution(
            _data(per_mol, v_effective=20),
            cell_id="pubchem__smirk_gpe_v256_nmb",
            arm="bpe",
            boundary="nmb",
            vocab_size=23,
            training_corpus_sha="sha-A",
            eval_split_sha="eval-A",
        )
        b = compute_arm_distribution(
            _data(per_mol, v_effective=20),
            cell_id="pubchem__smirk_gpe_v256_nmb",
            arm="bpe",
            boundary="nmb",
            vocab_size=23,
            training_corpus_sha="sha-A",
            eval_split_sha="eval-A",
        )

        assert a.d_ci == b.d_ci
        assert a.eta_ci == b.eta_ci
        assert a.renyi_ci == b.renyi_ci
        assert a.bootstrap_seed == b.bootstrap_seed

    def test_identical_molecules_collapse_all_three_cis_to_their_points(self) -> None:
        # Every held-out molecule emits the same multiset, so a molecule resample
        # reproduces the same per-token proportions and each CI collapses to its
        # own point. This pins the resample unit (molecules, not token rows) and
        # that the bootstrap recomputes D / eta / Renyi with v_effective held
        # fixed — a token-row resample would jitter instead.
        per_mol = [{10: 2, 11: 1}] * 5

        arm = compute_arm_distribution(
            _data(per_mol, v_effective=10),
            cell_id="x",
            arm="bpe",
            boundary="nmb",
            vocab_size=14,
            training_corpus_sha="sha-A",
            eval_split_sha="eval-A",
            n_resamples=300,
        )

        assert arm.d_ci == pytest.approx((arm.d, arm.d))
        assert arm.eta_ci == pytest.approx((arm.eta, arm.eta))
        assert arm.renyi_ci == pytest.approx((arm.renyi, arm.renyi))

    def test_bootstrap_ci_brackets_point_estimate(self) -> None:
        per_mol = [
            {10: 1 + (i % 3), 11: 1 + (i % 5), 12: 1 + (i % 2)} for i in range(60)
        ]

        arm = compute_arm_distribution(
            _data(per_mol, v_effective=20),
            cell_id="x",
            arm="bpe",
            boundary="nmb",
            vocab_size=23,
            training_corpus_sha="sha-A",
            eval_split_sha="eval-A",
        )

        lo, hi = arm.d_ci
        assert lo <= arm.d <= hi

    def test_distinct_cells_have_distinct_seeds(self) -> None:
        assert bootstrap_seed("pubchem__smirk_gpe_v256_nmb") != bootstrap_seed(
            "pubchem__smirk_unigram_v256_nmb"
        )

    def test_zero_molecules_yields_nan_ci_and_zero_point(self) -> None:
        arm = compute_arm_distribution(
            _data([], v_effective=10),
            cell_id="x",
            arm="bpe",
            boundary="nmb",
            vocab_size=14,
            training_corpus_sha="sha-A",
            eval_split_sha="eval-A",
        )

        assert arm.live_token_count == 0
        assert arm.total_tokens == 0
        assert arm.d_ci[0] != arm.d_ci[0]


def _arm6(
    *,
    arm: str,
    cell_id: str,
    boundary: str = "nmb",
    d: float = 0.30,
    eta: float = 0.80,
    renyi: float = 0.70,
    v_effective: int = 256,
    live_token_count: int = 200,
) -> ArmDistribution:
    return ArmDistribution(
        cell_id=cell_id,
        arm=arm,  # type: ignore[arg-type]
        boundary=boundary,  # type: ignore[arg-type]
        n_molecules=100,
        total_tokens=5000,
        vocab_size=v_effective + 3,
        v_effective=v_effective,
        live_token_count=live_token_count,
        d=d,
        d_ci=(d - 0.01, d + 0.01),
        eta=eta,
        eta_ci=(eta - 0.01, eta + 0.01),
        renyi=renyi,
        renyi_ci=(renyi - 0.01, renyi + 0.01),
        training_corpus_sha="sha-A",
        eval_split_sha="eval-A",
        bootstrap_seed=42,
        n_resamples=1000,
    )


class TestMatchedPair:
    def test_delta_d_is_bpe_minus_unigram(self) -> None:
        bpe = _arm6(arm="bpe", cell_id="pubchem__smirk_gpe_v256_nmb", d=0.40)
        unigram = _arm6(
            arm="unigram", cell_id="pubchem__smirk_unigram_v256_nmb", d=0.25
        )

        pair = compute_matched_pair_distribution(
            bpe,
            unigram,
            pair_key="pubchem__v256_nmb",
            tier="headline",
            corpus="pubchem",
            vocab_size=256,
            boundary="nmb",
        )

        assert pair.delta_d == pytest.approx(0.15)
        assert pair.abs_delta_d == pytest.approx(0.15)
        assert pair.delta_d_exceeds_threshold is True

    def test_small_delta_d_does_not_clear_noise_floor(self) -> None:
        bpe = _arm6(arm="bpe", cell_id="a", d=0.281)
        unigram = _arm6(arm="unigram", cell_id="b", d=0.280)

        pair = compute_matched_pair_distribution(
            bpe,
            unigram,
            pair_key="pk",
            tier="headline",
            corpus="pubchem",
            vocab_size=256,
            boundary="nmb",
        )

        assert pair.abs_delta_d == pytest.approx(0.001)
        assert pair.delta_d_exceeds_threshold is False

    def test_delta_eta_and_renyi_recorded(self) -> None:
        bpe = _arm6(arm="bpe", cell_id="a", eta=0.85, renyi=0.75)
        unigram = _arm6(arm="unigram", cell_id="b", eta=0.80, renyi=0.70)

        pair = compute_matched_pair_distribution(
            bpe,
            unigram,
            pair_key="pk",
            tier="headline",
            corpus="pubchem",
            vocab_size=256,
            boundary="nmb",
        )

        assert pair.delta_eta == pytest.approx(0.05)
        assert pair.delta_renyi == pytest.approx(0.05)

    def test_equal_v_effective_is_consistent(self) -> None:
        bpe = _arm6(arm="bpe", cell_id="a", v_effective=256)
        unigram = _arm6(arm="unigram", cell_id="b", v_effective=256)

        pair = compute_matched_pair_distribution(
            bpe,
            unigram,
            pair_key="pk",
            tier="headline",
            corpus="pubchem",
            vocab_size=256,
            boundary="nmb",
        )

        assert pair.v_effective_consistent is True
        assert pair.v_effective_delta == 0

    def test_unequal_v_effective_flagged_not_raised(self) -> None:
        bpe = _arm6(arm="bpe", cell_id="a", v_effective=256)
        unigram = _arm6(arm="unigram", cell_id="b", v_effective=255)

        pair = compute_matched_pair_distribution(
            bpe,
            unigram,
            pair_key="pk",
            tier="headline",
            corpus="pubchem",
            vocab_size=256,
            boundary="nmb",
        )

        assert pair.v_effective_consistent is False
        assert pair.v_effective_delta == 1

    def test_arm_order_swap_raises(self) -> None:
        bpe = _arm6(arm="bpe", cell_id="a")
        unigram = _arm6(arm="unigram", cell_id="b")

        with pytest.raises(ValueError, match="first argument must be the BPE arm"):
            compute_matched_pair_distribution(
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
        bpe = _arm6(arm="bpe", cell_id="a")

        with pytest.raises(ValueError, match="must be the Unigram arm"):
            compute_matched_pair_distribution(
                bpe,
                _arm6(arm="bpe", cell_id="b"),
                pair_key="pk",
                tier="headline",
                corpus="pubchem",
                vocab_size=256,
                boundary="nmb",
            )

    def test_as_dict_is_json_serializable_from_numpy_backed_arms(self) -> None:
        bpe = compute_arm_distribution(
            _data([{10: 70}, {11: 30}], v_effective=10),
            cell_id="pubchem__smirk_gpe_v256_nmb",
            arm="bpe",
            boundary="nmb",
            vocab_size=14,
            training_corpus_sha="sha-A",
            eval_split_sha="eval-A",
        )
        unigram = compute_arm_distribution(
            _data([{10: 40}, {11: 60}], v_effective=10),
            cell_id="pubchem__smirk_unigram_v256_nmb",
            arm="unigram",
            boundary="nmb",
            vocab_size=14,
            training_corpus_sha="sha-A",
            eval_split_sha="eval-A",
        )

        pair = compute_matched_pair_distribution(
            bpe,
            unigram,
            pair_key="pubchem__v256_nmb",
            tier="headline",
            corpus="pubchem",
            vocab_size=256,
            boundary="nmb",
        )

        payload = json.loads(json.dumps(pair.as_dict()))

        assert isinstance(pair.delta_d_exceeds_threshold, bool)
        assert payload["pair_key"] == "pubchem__v256_nmb"

    def test_boundary_mismatch_raises(self) -> None:
        bpe = _arm6(arm="bpe", cell_id="a", boundary="nmb")
        unigram = _arm6(arm="unigram", cell_id="b", boundary="mb")

        with pytest.raises(ValueError, match="boundaries must match"):
            compute_matched_pair_distribution(
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
        arm = _arm6(arm="bpe", cell_id="zinc22__smirk_gpe_v2048_nmb")

        rec = compute_unpaired_distribution(
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
        assert rec.as_dict()["delta_d"] is None

    def test_boundary_mismatch_raises(self) -> None:
        arm = _arm6(arm="bpe", cell_id="x", boundary="mb")

        with pytest.raises(ValueError, match="disagrees with pair"):
            compute_unpaired_distribution(
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
        arm = _arm6(arm="bpe", cell_id="x")

        with pytest.raises(ValueError, match="cannot equal the present arm"):
            compute_unpaired_distribution(
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
