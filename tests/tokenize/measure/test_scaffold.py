"""Tests for the pure scaffold-token math."""

from __future__ import annotations

import json
from io import StringIO

import pytest

from smiles_subword.tokenize.measure.scaffold import (
    ArmScaffold,
    ScaffoldRecord,
    bucket_by_surface_form,
    classify_scaffolds,
    classify_surface_form,
    compute_bpe_arm_scaffold,
    compute_matched_pair_scaffold,
    compute_unigram_arm_scaffold,
    compute_unpaired_scaffold,
    empty_surface_breakdown,
    end_of_training_standalone,
    parse_scaffold_log,
    scaffold_threshold,
)


def _make_record(
    step: int,
    *,
    new_id: int,
    new_token: str,
    candidate_freq: int,
    standalone: list[tuple[int, int]],
    pair: tuple[int, int] = (0, 0),
) -> ScaffoldRecord:
    return ScaffoldRecord(
        step=step,
        pair=pair,
        new_id=new_id,
        new_token=new_token,
        candidate_freq=candidate_freq,
        standalone=tuple(standalone),
    )


class TestClassifySurfaceForm:
    @pytest.mark.parametrize(
        ("token", "expected"),
        [
            pytest.param("[NH3+]", "bracket_internal", id="bracket_full"),
            pytest.param("[N", "bracket_internal", id="bracket_open_only"),
            pytest.param("()=", "structural", id="all_structural"),
            pytest.param("==", "structural", id="double_eq"),
            pytest.param("123", "structural", id="ring_closures_only"),
            pytest.param("CCO", "atomic", id="atomic_chain"),
            pytest.param("c1cc", "atomic", id="aromatic_with_ring"),
            pytest.param("=O", "atomic", id="atom_with_bond"),
            pytest.param("Br", "atomic", id="two_letter_atom"),
        ],
    )
    def test_classify_known_tokens(self, token: str, expected: str) -> None:
        assert classify_surface_form(token) == expected


class TestParseScaffoldLog:
    def test_parses_header_and_records_from_smirk_jsonl(self) -> None:
        lines = [
            json.dumps(
                {
                    "format": "smirk-scaffold-log/v1",
                    "min_frequency": 0,
                    "vocab_size": 170,
                    "merge_brackets": False,
                    "limit_alphabet": None,
                    "base_alphabet": [[45, "C"], [102, "O"]],
                }
            ),
            json.dumps(
                {
                    "step": 0,
                    "pair": [45, 45],
                    "new_id": 159,
                    "new_token": "CC",
                    "candidate_freq": 10,
                    "standalone": [[45, 4], [159, 9]],
                }
            ),
        ]

        header, records = parse_scaffold_log(StringIO("\n".join(lines) + "\n"))

        assert header.format == "smirk-scaffold-log/v1"
        assert header.vocab_size == 170
        assert header.merge_brackets is False
        assert header.alphabet_dict() == {45: "C", 102: "O"}
        assert records[0].step == 0
        assert records[0].new_id == 159
        assert records[0].candidate_freq == 10
        assert records[0].standalone == ((45, 4), (159, 9))

    def test_raises_on_empty_log(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            parse_scaffold_log(iter([]))

    def test_raises_on_unknown_format(self) -> None:
        with pytest.raises(ValueError, match="unexpected scaffold log format"):
            parse_scaffold_log([json.dumps({"format": "other-format/v2"})])

    def test_skips_blank_lines(self) -> None:
        lines = [
            json.dumps(
                {
                    "format": "smirk-scaffold-log/v1",
                    "min_frequency": 0,
                    "vocab_size": 160,
                    "merge_brackets": False,
                    "base_alphabet": [],
                }
            ),
            "",
            json.dumps(
                {
                    "step": 0,
                    "pair": [0, 0],
                    "new_id": 159,
                    "new_token": "X",
                    "candidate_freq": 5,
                    "standalone": [],
                }
            ),
        ]
        _header, records = parse_scaffold_log(lines)
        assert len(records) == 1


class TestEndOfTrainingStandalone:
    def test_sums_deltas_across_records(self) -> None:
        records = [
            _make_record(
                0,
                new_id=159,
                new_token="CC",
                candidate_freq=10,
                standalone=[(45, 4), (159, 9)],
            ),
            _make_record(
                1,
                new_id=160,
                new_token="CCC",
                candidate_freq=5,
                standalone=[(45, -2), (159, -3), (160, 2)],
            ),
        ]

        totals = end_of_training_standalone(records)

        assert totals[45] == 2
        assert totals[159] == 6
        assert totals[160] == 2

    def test_only_ids_filter_zero_fills_unmentioned(self) -> None:
        records = [
            _make_record(
                0, new_id=159, new_token="CC", candidate_freq=10, standalone=[(45, 4)]
            )
        ]

        totals = end_of_training_standalone(records, only_ids=frozenset({45, 999}))

        assert totals == {45: 4, 999: 0}


class TestScaffoldThreshold:
    def test_threshold_is_last_records_candidate_freq(self) -> None:
        records = [
            _make_record(
                0, new_id=159, new_token="a", candidate_freq=10, standalone=[]
            ),
            _make_record(1, new_id=160, new_token="b", candidate_freq=3, standalone=[]),
        ]

        assert scaffold_threshold(records) == 3

    def test_raises_on_empty(self) -> None:
        with pytest.raises(ValueError, match="zero records"):
            scaffold_threshold([])


class TestClassifyScaffolds:
    """Lian-2024 criterion: standalone < threshold ⇒ scaffold."""

    def test_filters_by_surviving_ids(self) -> None:
        records = [
            _make_record(
                0, new_id=159, new_token="CC", candidate_freq=10, standalone=[(159, 1)]
            ),
            _make_record(
                1,
                new_id=160,
                new_token="CCC",
                candidate_freq=10,
                standalone=[(159, 0), (160, 1)],
            ),
            _make_record(
                2,
                new_id=161,
                new_token="CCCC",
                candidate_freq=5,
                standalone=[(159, 1), (160, 1), (161, 1)],
            ),
        ]

        scaffolds = classify_scaffolds(records, surviving_ids=frozenset({159, 160}))

        assert 161 not in scaffolds

    def test_zero_standalone_below_threshold_is_scaffold(self) -> None:
        records = [
            _make_record(
                0, new_id=159, new_token="A", candidate_freq=10, standalone=[(159, 0)]
            ),
            _make_record(
                1,
                new_id=160,
                new_token="B",
                candidate_freq=3,
                standalone=[(159, 0), (160, 5)],
            ),
        ]

        scaffolds = classify_scaffolds(records, surviving_ids=frozenset({159, 160}))

        assert 159 in scaffolds
        assert 160 not in scaffolds

    def test_at_threshold_is_not_scaffold(self) -> None:
        """Strict inequality: standalone == threshold is NOT a scaffold."""
        records = [
            _make_record(
                0, new_id=159, new_token="A", candidate_freq=5, standalone=[(159, 3)]
            ),
            _make_record(
                1, new_id=160, new_token="B", candidate_freq=3, standalone=[(160, 3)]
            ),
        ]

        scaffolds = classify_scaffolds(records, surviving_ids=frozenset({159, 160}))

        assert 160 not in scaffolds


class TestBucketBySurfaceForm:
    def test_breaks_down_by_three_bins(self) -> None:
        records = [
            _make_record(
                0, new_id=159, new_token="CC", candidate_freq=10, standalone=[]
            ),
            _make_record(
                1, new_id=160, new_token="[NH3+]", candidate_freq=8, standalone=[]
            ),
            _make_record(
                2, new_id=161, new_token="()", candidate_freq=4, standalone=[]
            ),
        ]

        counts = bucket_by_surface_form([159, 160, 161], records)

        assert counts == {"atomic": 1, "bracket_internal": 1, "structural": 1}

    def test_unknown_ids_are_skipped(self) -> None:
        records = [
            _make_record(
                0, new_id=159, new_token="CC", candidate_freq=10, standalone=[]
            )
        ]

        counts = bucket_by_surface_form([159, 999], records)

        assert counts["atomic"] == 1
        assert sum(counts.values()) == 1

    def test_empty_bins_default_to_zero(self) -> None:
        counts = bucket_by_surface_form([], [])
        assert counts == {"bracket_internal": 0, "structural": 0, "atomic": 0}


class TestComputeBpeArmScaffold:
    def test_assembles_arm_record(self) -> None:
        records = [
            _make_record(
                0, new_id=159, new_token="CC", candidate_freq=10, standalone=[(159, 9)]
            ),
            _make_record(
                1,
                new_id=160,
                new_token="CCC",
                candidate_freq=5,
                standalone=[(159, -8), (160, 4)],
            ),
            _make_record(
                2,
                new_id=161,
                new_token="[NH3+]",
                candidate_freq=2,
                standalone=[(161, 1)],
            ),
        ]

        arm = compute_bpe_arm_scaffold(
            records,
            cell_id="pubchem__smirk_gpe_v256_nmb",
            boundary="nmb",
            vocab_size=256,
            n_merges=3,
            atomic_vocab_size=159,
            training_corpus_sha="abc123",
            scaffold_log_sha="def456",
        )

        assert arm.arm == "bpe"
        assert arm.cell_id == "pubchem__smirk_gpe_v256_nmb"
        assert arm.threshold == 2
        assert arm.scaffold_count == 2
        assert arm.scaffold_fraction_of_v == pytest.approx(2 / 256)
        assert arm.surface_form_breakdown["atomic"] == 1
        assert arm.surface_form_breakdown["bracket_internal"] == 1
        assert arm.verified_by_construction is False
        assert arm.scaffold_log_sha == "def456"


class TestComputeUnigramArmScaffold:
    def test_emits_zero_by_construction(self) -> None:
        arm = compute_unigram_arm_scaffold(
            cell_id="pubchem__smirk_unigram_v256_nmb",
            boundary="nmb",
            vocab_size=256,
            training_corpus_sha="abc123",
        )

        assert arm.arm == "unigram"
        assert arm.scaffold_count == 0
        assert arm.scaffold_fraction_of_v == 0.0
        assert arm.verified_by_construction is True
        assert arm.threshold is None
        assert arm.n_merges is None
        assert arm.scaffold_log_sha is None
        assert all(v == 0 for v in arm.surface_form_breakdown.values())


def _make_arm_scaffold(
    arm: str,
    *,
    boundary: str = "nmb",
    scaffold_count: int = 0,
    vocab_size: int = 256,
) -> ArmScaffold:
    fraction = (scaffold_count / vocab_size) if vocab_size > 0 else 0.0
    return ArmScaffold(
        cell_id=f"pubchem__smirk_{arm}_v{vocab_size}_{boundary}",
        arm=arm,  # type: ignore[arg-type]
        boundary=boundary,  # type: ignore[arg-type]
        vocab_size=vocab_size,
        n_merges=(scaffold_count + 2 if arm == "bpe" else None),
        scaffold_count=scaffold_count,
        scaffold_fraction_of_v=fraction,
        surface_form_breakdown=empty_surface_breakdown(),
        threshold=(2 if arm == "bpe" else None),
        verified_by_construction=(arm == "unigram"),
        training_corpus_sha="abc",
        scaffold_log_sha=("xyz" if arm == "bpe" else None),
    )


class TestComputeMatchedPairScaffold:
    def test_delta_is_bpe_minus_unigram(self) -> None:
        bpe = _make_arm_scaffold("bpe", scaffold_count=10, vocab_size=256)
        unigram = _make_arm_scaffold("unigram", scaffold_count=0, vocab_size=256)

        pair = compute_matched_pair_scaffold(
            bpe,
            unigram,
            pair_key="pubchem__v256_nmb",
            tier="headline",
            corpus="pubchem",
            vocab_size=256,
            boundary="nmb",
        )

        assert pair.delta_scaffold_fraction == pytest.approx(10 / 256)
        assert pair.pair_status == "matched"

    def test_raises_on_arm_tag_swap(self) -> None:
        bpe = _make_arm_scaffold("bpe")
        unigram = _make_arm_scaffold("unigram")

        with pytest.raises(ValueError, match="first argument must be the BPE arm"):
            compute_matched_pair_scaffold(
                unigram,
                bpe,
                pair_key="x",
                tier="t",
                corpus="c",
                vocab_size=256,
                boundary="nmb",
            )

    def test_raises_on_non_unigram_second_arg(self) -> None:
        # The swap test trips the bpe-side guard first; this pins its sibling.
        bpe = _make_arm_scaffold("bpe")

        with pytest.raises(ValueError, match="must be the Unigram arm"):
            compute_matched_pair_scaffold(
                bpe,
                _make_arm_scaffold("bpe"),
                pair_key="x",
                tier="t",
                corpus="c",
                vocab_size=256,
                boundary="nmb",
            )

    def test_raises_on_boundary_mismatch(self) -> None:
        bpe = _make_arm_scaffold("bpe", boundary="nmb")
        unigram = _make_arm_scaffold("unigram", boundary="mb")

        with pytest.raises(ValueError, match="boundaries must match"):
            compute_matched_pair_scaffold(
                bpe,
                unigram,
                pair_key="x",
                tier="t",
                corpus="c",
                vocab_size=256,
                boundary="nmb",
            )


class TestComputeUnpairedScaffold:
    def test_wraps_present_arm(self) -> None:
        bpe = _make_arm_scaffold("bpe", scaffold_count=4, vocab_size=2048)

        unpaired = compute_unpaired_scaffold(
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

        assert unpaired.pair_status == "single_arm"
        assert unpaired.present_arm.arm == "bpe"
        assert unpaired.missing_arm == "unigram"

    def test_raises_on_boundary_mismatch(self) -> None:
        bpe = _make_arm_scaffold("bpe", boundary="mb")

        with pytest.raises(ValueError, match="disagrees with pair"):
            compute_unpaired_scaffold(
                bpe,
                pair_key="x",
                tier="t",
                corpus="c",
                vocab_size=256,
                boundary="nmb",
                extras_kind=None,
                extras_label=None,
                missing_arm="unigram",
                unpaired_reason="conditional_negative_branch",
            )

    def test_raises_when_missing_arm_equals_present(self) -> None:
        bpe = _make_arm_scaffold("bpe")

        with pytest.raises(ValueError, match="cannot equal the present arm"):
            compute_unpaired_scaffold(
                bpe,
                pair_key="x",
                tier="t",
                corpus="c",
                vocab_size=256,
                boundary="nmb",
                extras_kind=None,
                extras_label=None,
                missing_arm="bpe",
                unpaired_reason="conditional_negative_branch",
            )
