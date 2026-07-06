"""Tests for ``smiles_subword.tokenize.measure.deadzone`` (pure computation)."""

from __future__ import annotations

import pytest

from smiles_subword.tokenize.audit.f95 import F95_GRID, HEADLINE_N, HEADLINE_P
from smiles_subword.tokenize.measure.deadzone import (
    ArmF95Slice,
    compute_matched_pair_deadzone,
    compute_unpaired_deadzone,
)


def _slice(
    *,
    arm: str,
    cell_id: str = "cell",
    f50: float = 1.0,
    f100: float = 1.0,
    f200: float = 1.0,
    unsafe: bool = False,
    sha: str = "sha",
    v: int = 256,
    n_non_atomic: int = 80,
) -> ArmF95Slice:
    return ArmF95Slice(
        cell_id=cell_id,
        arm=arm,  # type: ignore[arg-type]
        clearance_by_n={50: f50, 100: f100, 200: f200},
        headline_clearance=f100,
        embedding_tail_unsafe=unsafe,
        training_corpus_sha=sha,
        v_observed=v,
        n_non_atomic=n_non_atomic,
    )


def _f95_payload(
    *,
    arm: str,
    cell_id: str = "pubchem__smirk_gpe_v256_nmb",
    f50: float = 0.9,
    f100: float = 0.8,
    f200: float = 0.5,
    unsafe: bool = True,
    sha: str = "abc",
) -> dict[str, object]:
    return {
        "arm": arm,
        "cell_id": cell_id,
        "algo": "bpe" if arm == "bpe" else "unigram",
        "vocab_size": 256,
        "corpus": "pubchem",
        "boundary": "nmb",
        "tier": "headline",
        "training_corpus_sha": sha,
        "v_observed": 256,
        "n_non_atomic": 80,
        "n_corpus_tokens": 1000,
        "n_corpus_molecules": 100,
        "fp_thresholds": [],
        "clearance_by_n": {"50": f50, "100": f100, "200": f200},
        "headline_clearance": f100,
        "embedding_tail_unsafe": unsafe,
    }


class TestArmF95SliceFromPayload:
    def test_projects_relevant_fields_from_an_f95_payload(self) -> None:
        payload = _f95_payload(arm="bpe", f100=0.92, unsafe=False, sha="sha123")

        sliced = ArmF95Slice.from_f95_payload(payload)

        assert sliced.arm == "bpe"
        assert sliced.clearance_by_n == {50: 0.9, 100: 0.92, 200: 0.5}
        assert sliced.headline_clearance == 0.92
        assert sliced.embedding_tail_unsafe is False
        assert sliced.training_corpus_sha == "sha123"

    def test_rejects_unknown_arm(self) -> None:
        payload = _f95_payload(arm="garbage")

        with pytest.raises(ValueError, match="arm must be"):
            ArmF95Slice.from_f95_payload(payload)

    def test_f_at_is_p_independent_for_given_n(self) -> None:
        sliced = ArmF95Slice.from_f95_payload(_f95_payload(arm="bpe", f100=0.75))

        assert (
            sliced.f_at(0.90, 100) == sliced.f_at(0.95, 100) == sliced.f_at(0.99, 100)
        )
        assert sliced.f_at(0.95, 100) == 0.75


class TestComputeMatchedPairDeadzone:
    def test_emits_one_delta_per_f95_grid_point(self) -> None:
        bpe = _slice(arm="bpe")
        ul = _slice(arm="unigram")

        pair = compute_matched_pair_deadzone(
            bpe,
            ul,
            pair_key="pubchem__v256_nmb",
            tier="headline",
            corpus="pubchem",
            vocab_size=256,
            boundary="nmb",
        )

        assert len(pair.delta_fp) == len(F95_GRID)
        assert {(d.p, d.n) for d in pair.delta_fp} == set(F95_GRID)

    def test_exactly_one_delta_is_flagged_headline(self) -> None:
        pair = compute_matched_pair_deadzone(
            _slice(arm="bpe"),
            _slice(arm="unigram"),
            pair_key="x",
            tier="headline",
            corpus="pubchem",
            vocab_size=256,
            boundary="nmb",
        )

        headline = [d for d in pair.delta_fp if d.is_headline]
        assert len(headline) == 1
        assert headline[0].p == HEADLINE_P
        assert headline[0].n == HEADLINE_N

    def test_positive_delta_when_bpe_clears_more_than_unigram(self) -> None:
        bpe = _slice(arm="bpe", f50=1.0, f100=0.90, f200=0.5)
        ul = _slice(arm="unigram", f50=1.0, f100=0.60, f200=0.5)

        pair = compute_matched_pair_deadzone(
            bpe,
            ul,
            pair_key="x",
            tier="headline",
            corpus="pubchem",
            vocab_size=256,
            boundary="nmb",
        )

        assert pair.headline_delta_f == pytest.approx(0.30)
        headline = next(d for d in pair.delta_fp if d.is_headline)
        assert headline.f_bpe == 0.90
        assert headline.f_unigram == 0.60

    def test_negative_delta_when_unigram_clears_more_than_bpe(self) -> None:
        # The mirror of the positive case: the sign convention is a true signed
        # difference (f_bpe - f_unigram), so when Unigram clears more the headline
        # ΔF is negative. A bug computing abs(Δ) or clamping at 0 would pass the
        # positive test alone but fail here.
        bpe = _slice(arm="bpe", f50=1.0, f100=0.55, f200=0.4)
        ul = _slice(arm="unigram", f50=1.0, f100=0.85, f200=0.4)

        pair = compute_matched_pair_deadzone(
            bpe,
            ul,
            pair_key="x",
            tier="headline",
            corpus="pubchem",
            vocab_size=256,
            boundary="nmb",
        )

        assert pair.headline_delta_f == pytest.approx(-0.30)
        assert pair.headline_delta_f < 0.0

    def test_propagates_any_and_both_unsafe_flags(self) -> None:
        pair = compute_matched_pair_deadzone(
            _slice(arm="bpe", unsafe=True),
            _slice(arm="unigram", unsafe=False),
            pair_key="x",
            tier="headline",
            corpus="pubchem",
            vocab_size=256,
            boundary="nmb",
        )

        assert pair.any_arm_unsafe is True
        assert pair.both_arms_unsafe is False

    def test_both_unsafe_when_each_arm_failed_headline(self) -> None:
        pair = compute_matched_pair_deadzone(
            _slice(arm="bpe", unsafe=True),
            _slice(arm="unigram", unsafe=True),
            pair_key="x",
            tier="headline",
            corpus="coconut",
            vocab_size=1024,
            boundary="nmb",
        )

        assert pair.both_arms_unsafe is True

    def test_rejects_swapped_arm_arguments(self) -> None:
        with pytest.raises(ValueError, match="BPE arm"):
            compute_matched_pair_deadzone(
                _slice(arm="unigram"),
                _slice(arm="bpe"),
                pair_key="x",
                tier="headline",
                corpus="pubchem",
                vocab_size=256,
                boundary="nmb",
            )

    def test_rejects_a_non_unigram_second_argument(self) -> None:
        # The bpe-side guard fires for swapped args; this pins its sibling — a
        # second argument that is not the Unigram arm.
        with pytest.raises(ValueError, match="Unigram arm"):
            compute_matched_pair_deadzone(
                _slice(arm="bpe"),
                _slice(arm="bpe"),
                pair_key="x",
                tier="headline",
                corpus="pubchem",
                vocab_size=256,
                boundary="nmb",
            )

    def test_serialized_pair_carries_status_and_extras_axes(self) -> None:
        pair = compute_matched_pair_deadzone(
            _slice(arm="bpe"),
            _slice(arm="unigram"),
            pair_key="pubchem__v512_nmb__subsample_r1",
            tier="extras_subsample_redraw",
            corpus="pubchem",
            vocab_size=512,
            boundary="nmb",
            extras_kind="subsample_redraw",
            extras_label="r1",
        )

        payload = pair.as_dict()
        assert payload["pair_status"] == "matched"
        assert payload["extras_kind"] == "subsample_redraw"
        assert payload["extras_label"] == "r1"
        assert payload["missing_arm"] is None
        assert payload["unpaired_reason"] is None


class TestComputeUnpairedDeadzone:
    def test_records_present_and_missing_arms(self) -> None:
        bpe = _slice(arm="bpe", cell_id="zinc22__smirk_gpe_v2048_nmb", unsafe=True)

        record = compute_unpaired_deadzone(
            bpe,
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

        payload = record.as_dict()
        assert payload["pair_status"] == "single_arm"
        assert payload["bpe"] is not None
        assert payload["unigram"] is None
        assert payload["missing_arm"] == "unigram"
        assert payload["headline_delta_f"] is None
        assert payload["delta_fp"] is None
        assert payload["any_arm_unsafe"] is True
        assert payload["both_arms_unsafe"] is False

    def test_single_arm_unigram_serializes_with_unigram_block(self) -> None:
        ul = _slice(
            arm="unigram", cell_id="pubchem__smirk_unigram_v1024_mb_seed_uncapped"
        )

        record = compute_unpaired_deadzone(
            ul,
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

        payload = record.as_dict()
        assert payload["unigram"] is not None
        assert payload["bpe"] is None
        assert payload["unpaired_reason"] == "extras_single_arm_knob"

    def test_rejects_missing_arm_equal_to_present(self) -> None:
        bpe = _slice(arm="bpe")

        with pytest.raises(ValueError, match="cannot equal"):
            compute_unpaired_deadzone(
                bpe,
                pair_key="x",
                tier="conditional",
                corpus="zinc22",
                vocab_size=2048,
                boundary="nmb",
                extras_kind=None,
                extras_label=None,
                missing_arm="bpe",
                unpaired_reason="conditional_negative_branch",
            )
