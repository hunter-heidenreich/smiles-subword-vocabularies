"""Tests for ``fg_alignment`` pure math (functional-bond locality).

Covers per-arm aggregation (overall + per-class locality, deterministic CIs,
nan when no functional bond fired), the matched-pair gap with nan propagation
and arm/boundary validation, and the within-arm single-arm wrap.
"""

from __future__ import annotations

import math

import pytest

from smiles_subword.tokenize.measure.fg_alignment import (
    ArmFgAlignment,
    PerMoleculeFgLocality,
    compute_arm_fg_alignment,
    compute_matched_pair_fg_alignment,
    compute_unpaired_fg_alignment,
)


def _arm(
    per_molecule: list[PerMoleculeFgLocality],
    *,
    arm: str = "bpe",
    cell_id: str = "pubchem__smirk_gpe_v256_nmb",
) -> ArmFgAlignment:
    return compute_arm_fg_alignment(
        per_molecule,
        cell_id=cell_id,
        arm=arm,  # type: ignore[arg-type]
        boundary="nmb",
        training_corpus_sha="sha-train",
        eval_split_sha="sha-eval",
        n_resamples=64,
    )


class TestComputeArm:
    def test_overall_and_per_class_locality(self) -> None:
        per_molecule = [
            PerMoleculeFgLocality(
                n_bonds=2,
                n_local=2,
                class_bonds={"C=O": 1, "C#N": 1},
                class_local={"C=O": 1, "C#N": 1},
            ),
            PerMoleculeFgLocality(
                n_bonds=2,
                n_local=1,
                class_bonds={"C=O": 2},
                class_local={"C=O": 1},
            ),
        ]
        arm = _arm(per_molecule)

        assert arm.n_bonds == 4
        assert arm.n_local == 3
        assert arm.locality == pytest.approx(0.75)
        assert arm.class_locality("C=O") == pytest.approx(2 / 3)
        assert arm.class_locality("C#N") == pytest.approx(1.0)
        assert arm.n_molecules == 2

    def test_block_has_all_classes(self) -> None:
        arm = _arm(
            [
                PerMoleculeFgLocality(
                    n_bonds=1, n_local=1, class_bonds={"C=O": 1}, class_local={"C=O": 1}
                )
            ]
        )
        block = arm.as_block()
        # Every canonical class is present in the serialized breakdown.
        for label in ("C=O", "C#N", "C=N", "S=O", "P=O", "N=O", "C=S", "other"):
            assert label in block["class_bonds"]  # type: ignore[operator]

    def test_no_functional_bond_is_nan(self) -> None:
        arm = _arm([PerMoleculeFgLocality(n_bonds=0, n_local=0)])
        assert math.isnan(arm.locality)
        assert math.isnan(arm.class_locality("C=O"))

    def test_empty_split_nan_ci(self) -> None:
        arm = _arm([])
        assert arm.n_molecules == 0
        assert math.isnan(arm.locality_ci[0])
        assert math.isnan(arm.locality_ci[1])

    def test_ci_is_deterministic(self) -> None:
        per_molecule = [
            PerMoleculeFgLocality(
                n_bonds=2,
                n_local=i % 2,
                class_bonds={"C=O": 2},
                class_local={"C=O": i % 2},
            )
            for i in range(40)
        ]
        a = _arm(per_molecule)
        b = _arm(per_molecule)
        assert a.locality_ci == b.locality_ci


class TestMatchedPair:
    def _pair(self, bpe: ArmFgAlignment, ul: ArmFgAlignment):  # noqa: ANN202
        return compute_matched_pair_fg_alignment(
            bpe,
            ul,
            pair_key="pubchem__v256_nmb",
            tier="headline",
            corpus="pubchem",
            vocab_size=256,
            boundary="nmb",
        )

    def test_delta_is_bpe_minus_ul(self) -> None:
        bpe = _arm(
            [
                PerMoleculeFgLocality(
                    n_bonds=4, n_local=4, class_bonds={"C=O": 4}, class_local={"C=O": 4}
                )
            ],
            arm="bpe",
        )
        ul = _arm(
            [
                PerMoleculeFgLocality(
                    n_bonds=4, n_local=0, class_bonds={"C=O": 4}, class_local={}
                )
            ],
            arm="unigram",
            cell_id="pubchem__smirk_unigram_v256_nmb",
        )
        pair = self._pair(bpe, ul)
        assert pair.delta_locality == pytest.approx(1.0)
        assert pair.delta_locality_by_class["C=O"] == pytest.approx(1.0)
        assert pair.pair_status == "matched"

    def test_nan_class_gap_propagates(self) -> None:
        bpe = _arm(
            [
                PerMoleculeFgLocality(
                    n_bonds=1, n_local=1, class_bonds={"C=O": 1}, class_local={"C=O": 1}
                )
            ],
            arm="bpe",
        )
        ul = _arm(
            [
                PerMoleculeFgLocality(
                    n_bonds=1, n_local=0, class_bonds={"C=O": 1}, class_local={}
                )
            ],
            arm="unigram",
            cell_id="pubchem__smirk_unigram_v256_nmb",
        )
        pair = self._pair(bpe, ul)
        # Neither arm saw a nitrile, so its gap is undefined.
        assert math.isnan(pair.delta_locality_by_class["C#N"])

    def test_arm_tag_validation(self) -> None:
        good_ul = _arm([], arm="unigram")
        with pytest.raises(ValueError, match="must be the BPE arm"):
            self._pair(good_ul, good_ul)

    def test_boundary_mismatch_rejected(self) -> None:
        bpe = _arm([], arm="bpe")
        ul = compute_arm_fg_alignment(
            [],
            cell_id="pubchem__smirk_unigram_v256_mb",
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
        present = _arm(
            [
                PerMoleculeFgLocality(
                    n_bonds=2, n_local=2, class_bonds={"C=O": 2}, class_local={"C=O": 2}
                )
            ],
            arm="bpe",
        )
        rec = compute_unpaired_fg_alignment(
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
        assert payload["bpe"]["locality"] == pytest.approx(1.0)  # type: ignore[index]
        assert "unigram" not in payload  # within-arm: only the present arm block

    def test_missing_equals_present_rejected(self) -> None:
        present = _arm([], arm="bpe")
        with pytest.raises(ValueError, match="cannot equal present_arm"):
            compute_unpaired_fg_alignment(
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
