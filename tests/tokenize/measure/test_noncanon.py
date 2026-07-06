"""Tests for ``noncanon`` pure math (SMILES-rewrite-orbit robustness).

Covers per-arm per-axis aggregation (means, deterministic CIs, absent-axis
handling), the matched-pair cross-arm deltas and fertility-gap ratios with
arm/boundary validation, and the within-arm single-arm wrap.
"""

from __future__ import annotations

import math

import pytest

from smiles_subword.tokenize.measure.noncanon import (
    ArmNoncanon,
    PerMoleculeNoncanon,
    compute_arm_noncanon,
    compute_matched_pair_noncanon,
    compute_unpaired_noncanon,
)


def _pm(
    canon: int,
    rand_mean: float,
    dfert: dict[str, float],
    bag: dict[str, float],
) -> PerMoleculeNoncanon:
    return PerMoleculeNoncanon(
        canon_fert=canon, rand_fert_mean=rand_mean, axis_dfert=dfert, axis_bag=bag
    )


def _arm(
    per_molecule: list[PerMoleculeNoncanon],
    *,
    arm: str = "bpe",
    cell_id: str = "pubchem__smirk_gpe_v1024_nmb",
) -> ArmNoncanon:
    return compute_arm_noncanon(
        per_molecule,
        cell_id=cell_id,
        arm=arm,  # type: ignore[arg-type]
        boundary="nmb",
        training_corpus_sha="sha-train",
        eval_split_sha="sha-eval",
        n_resamples=64,
    )


class TestComputeArm:
    def test_per_axis_means_and_fert(self) -> None:
        arm = _arm(
            [
                _pm(
                    10,
                    12.0,
                    {"random": 0.2, "kekule": 0.1},
                    {"random": 0.3, "kekule": 0.5},
                ),
                _pm(
                    20,
                    24.0,
                    {"random": 0.4, "kekule": 0.3},
                    {"random": 0.1, "kekule": 0.7},
                ),
            ]
        )
        assert arm.axes["random"].bag_instab == pytest.approx(0.2)
        assert arm.axes["random"].rel_dfert == pytest.approx(0.3)
        assert arm.axes["kekule"].bag_instab == pytest.approx(0.6)
        assert arm.axes["random"].n == 2
        assert arm.mean_canon_fert == pytest.approx(15.0)
        assert arm.mean_rand_fert == pytest.approx(18.0)

    def test_absent_axis_omitted(self) -> None:
        # No molecule carries explicitH / ringperm -> those axes are absent.
        arm = _arm([_pm(10, 11.0, {"random": 0.2}, {"random": 0.3})])
        assert "random" in arm.axes
        assert "explicitH" not in arm.axes
        assert "ringperm" not in arm.axes

    def test_ci_is_deterministic(self) -> None:
        pm = [
            _pm(10 + i, 12.0, {"random": (i % 3) / 10}, {"random": (i % 5) / 10})
            for i in range(40)
        ]
        assert (
            _arm(pm).axes["random"].bag_instab_ci
            == _arm(pm).axes["random"].bag_instab_ci
        )

    def test_identical_molecules_collapse_axis_cis_to_their_points(self) -> None:
        # Every molecule reports the same per-axis movement, so any molecule
        # resample reproduces the same mean: each axis CI collapses to its point.
        # Pins the resample unit (molecules) for both bag_instab and rel_dfert.
        pm = [_pm(10, 12.0, {"random": 0.3}, {"random": 0.5})] * 5
        arm = compute_arm_noncanon(
            pm,
            cell_id="c",
            arm="bpe",
            boundary="nmb",
            training_corpus_sha="s",
            eval_split_sha="e",
            n_resamples=200,
        )

        ax = arm.axes["random"]
        assert ax.bag_instab == pytest.approx(0.5)
        assert ax.bag_instab_ci == pytest.approx((0.5, 0.5))
        assert ax.rel_dfert == pytest.approx(0.3)
        assert ax.rel_dfert_ci == pytest.approx((0.3, 0.3))

    def test_empty_split(self) -> None:
        arm = _arm([])
        assert arm.axes == {}
        assert math.isnan(arm.mean_canon_fert)


class TestMatchedPair:
    def _pair(self, bpe: ArmNoncanon, ul: ArmNoncanon):  # noqa: ANN202
        return compute_matched_pair_noncanon(
            bpe,
            ul,
            pair_key="pubchem__v1024_nmb",
            tier="headline",
            corpus="pubchem",
            vocab_size=1024,
            boundary="nmb",
        )

    def test_delta_and_gap(self) -> None:
        bpe = _arm(
            [_pm(10, 12.0, {"random": 0.2}, {"random": 0.40})],
            arm="bpe",
        )
        ul = _arm(
            [_pm(14, 16.0, {"random": 0.1}, {"random": 0.15})],
            arm="unigram",
            cell_id="pubchem__smirk_unigram_v1024_nmb",
        )
        pair = self._pair(bpe, ul)
        # BPE bag higher -> positive delta (BPE the less stable arm).
        assert pair.delta_bag_instab["random"] == pytest.approx(0.25)
        assert pair.gap_canon == pytest.approx(14 / 10)
        assert pair.gap_rand == pytest.approx(16 / 12)
        assert pair.pair_status == "matched"

    def test_absent_axis_delta_is_nan(self) -> None:
        bpe = _arm([_pm(10, 11.0, {"random": 0.2}, {"random": 0.4})], arm="bpe")
        ul = _arm(
            [_pm(10, 11.0, {"random": 0.1}, {"random": 0.2})],
            arm="unigram",
            cell_id="pubchem__smirk_unigram_v1024_nmb",
        )
        pair = self._pair(bpe, ul)
        assert math.isnan(pair.delta_bag_instab["explicitH"])

    def test_arm_tag_validation(self) -> None:
        ul = _arm([], arm="unigram")
        with pytest.raises(ValueError, match="must be the BPE arm"):
            self._pair(ul, ul)

    def test_non_unigram_second_arg(self) -> None:
        # The swap above trips the bpe-side guard first; this pins its sibling.
        bpe = _arm([], arm="bpe")
        with pytest.raises(ValueError, match="must be the Unigram arm"):
            self._pair(bpe, bpe)

    def test_boundary_mismatch(self) -> None:
        bpe = _arm([], arm="bpe")
        ul = compute_arm_noncanon(
            [],
            cell_id="pubchem__smirk_unigram_v1024_mb",
            arm="unigram",
            boundary="mb",
            training_corpus_sha="x",
            eval_split_sha="y",
            n_resamples=8,
        )
        with pytest.raises(ValueError, match="boundaries must match"):
            self._pair(bpe, ul)


class TestUnpaired:
    def test_present_arm_carried(self) -> None:
        present = _arm([_pm(10, 11.0, {"random": 0.2}, {"random": 0.3})], arm="bpe")
        rec = compute_unpaired_noncanon(
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
        payload = rec.as_dict()
        assert "unigram" not in payload
        assert payload["bpe"]["axes"]["random"]["bag_instab"] == pytest.approx(0.3)

    def test_missing_equals_present_rejected(self) -> None:
        present = _arm([], arm="bpe")
        with pytest.raises(ValueError, match="cannot equal present_arm"):
            compute_unpaired_noncanon(
                present,
                pair_key="zinc22__v2048_nmb",
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
